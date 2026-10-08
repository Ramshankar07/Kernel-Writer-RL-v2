"""Unit tests for sft_decontam.py.   python modal_app/test_sft_decontam.py   (or pytest)"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sft_decontam as D  # noqa: E402

_IDX = None


def idx():
    global _IDX
    if _IDX is None:
        _IDX = D.Index()
    return _IDX


def kb(pid):
    for line in open(D.KB_PATH):
        r = json.loads(line)
        if r["problem_id"] == pid:
            return r["code"]
    raise KeyError(pid)


def rename(src, mapping):
    for a, b in mapping.items():
        src = re.sub(rf"\b{re.escape(a)}\b", b, src)
    return src


def row(ref, rid="t:0"):
    return {"id": rid, "source": "test", "format": "modelnew", "reference": ref, "family": []}


def test_renamed_kb_copy_flagged():
    # KB L1 #40 (LayerNorm): rename class, args, globals, attribute, alias nn, strip docstrings
    src = kb(40)
    src = rename(src, {"Model": "MyLayerNormNet", "normalized_shape": "nshape", "x": "inp",
                       "batch_size": "B", "features": "C", "dim1": "H", "dim2": "W", "ln": "norm"})
    src = src.replace("import torch.nn as nn", "import torch.nn as tnn").replace("nn.", "tnn.")
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    r = idx().check(row(src))
    assert r["flag"] and r["match"] == "kb_l1/40" and r["near_dup_kb"], r
    print("renamed KB#40 copy ->", {k: r[k] for k in ("flag", "match", "overlap", "shared_lines", "ast_equal")})


def test_renamed_kb_copy_flagged_by_lines_only():
    # KB L1 #97 (SDPA) with renamed variables AND a changed class body (extra no-op statement),
    # so the AST-equality path can't fire: the distinctive-line overlap must catch it.
    src = kb(97)
    src = rename(src, {"Model": "Attn", "Q": "q", "K": "k", "V": "v", "out": "o",
                       "batch_size": "bs", "num_heads": "nh", "sequence_length": "seq", "embedding_dimension": "hd"})
    src = src.replace("        return o\n", "        o = o.contiguous()\n        return o\n")
    r = idx().check(row(src))
    assert r["flag"] and r["match"] == "kb_l1/97" and not r["ast_equal"] and r["overlap"] >= 0.5, r
    print("renamed+edited KB#97 ->", {k: r[k] for k in ("flag", "match", "overlap", "shared_lines", "ast_equal")})


def test_unrelated_module_not_flagged():
    src = '''import torch
import torch.nn as nn

class GatedResidualShift(nn.Module):
    def __init__(self, channels, shift):
        super().__init__()
        self.shift = shift
        self.gate = nn.Parameter(torch.randn(channels))

    def forward(self, x, y):
        z = torch.roll(x, shifts=self.shift, dims=-1)
        g = torch.sigmoid(self.gate)[None, :, None]
        return torch.where(y > 0, z * g, y.flip(-1) - z)

def get_inputs():
    return [torch.rand([4, 12, 33]), torch.randn([4, 12, 33])]

def get_init_inputs():
    return [12, 3]
'''
    r = idx().check(row(src))
    assert not r["flag"], r
    print("unrelated module ->", {k: r[k] for k in ("flag", "match", "overlap", "op_overlap")})


def test_same_op_other_shapes_is_op_overlap_not_dup():
    # KernelBook-style softmax with tiny literal shapes: same op as KB#23 but not a copy of it.
    src = '''import torch
import torch.nn as nn

class SoftmaxLayer(nn.Module):
    def forward(self, x):
        y = torch.softmax(x, dim=1)
        return y * 2.0 + 1.0

def get_inputs():
    return [torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return []
'''
    r = idx().check(row(src))
    assert not r["flag"] and r["op_overlap"]["problem"] == "kb_l1/23" and r["op_overlap_flag"], r
    print("same-op/other-shape softmax ->", {k: r[k] for k in ("flag", "match", "overlap", "op_overlap")})


def test_val_reference_copy_flagged():
    # val problem fused_mlp/gelu_tanh wrapped into a Model (args renamed)
    src = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def forward(self, a, wg, wu, wd):
        g = F.gelu(a @ wg.T, approximate="tanh")
        u = a @ wu.T
        return (g * u) @ wd.T

def get_inputs():
    return [torch.randn(8, 64), torch.randn(128, 64), torch.randn(128, 64), torch.randn(64, 128)]
'''
    r = idx().check(row(src))
    assert r["flag"] and r["near_dup_val"] and "gelu_tanh" in r["match"], r
    print("val gelu_tanh copy ->", {k: r[k] for k in ("flag", "match", "overlap", "ast_equal")})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all decontam tests passed")
