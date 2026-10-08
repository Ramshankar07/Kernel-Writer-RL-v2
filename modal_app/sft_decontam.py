"""
SFT decontamination (Phase 1 step 4, local CPU): flag candidate rows whose PyTorch module is a
near-duplicate of a KernelBench Level 1 problem (results/kernelbench_l1_problems.jsonl) or of a
validation-set problem (results/v3/val_set/*/reference.py, ids from results/v3/val_set.json).

    python modal_app/sft_decontam.py results/v3/sft/candidates_drkernel.jsonl
    # -> results/v3/sft/decontam_drkernel.jsonl  (one line per input row, same order)
    python modal_app/test_sft_decontam.py          # unit tests

Method (research note section 3, step 3):
  1. Normalized AST: parse, drop docstrings / imports / type annotations, canonicalize module
     aliases (torch.nn.functional -> F, torch.nn -> nn), `super(X, self)` -> `super()`, rename
     identifiers by first appearance: module-level names g0.., function arguments/locals v0..
     (reset per function), `self.<attr>` a0.. (per class). Numeric literals are kept (so a module
     with other shapes is "same op", not "same problem").
  2. Distinctive lines: the normalized source lines minus structural boilerplate (def/class/
     return []/super().__init__()/pass) and minus lines found in >= 5% of the reference corpus
     (KB L1 + val set), e.g. `v0 = torch.rand(g0, g1)`.
  3. overlap(candidate, problem) = |D_cand & D_prob| / |D_prob|.
     near_dup  <=> overlap >= 0.5 with >= 2 shared distinctive lines, or the normalized AST of the
               problem's Model class (KB) / reference function body (val) equals the candidate's
               Model class / forward body.
     op_overlap = Jaccard of the torch ops called (torch.*/F.*/nn.* constructors/tensor methods)
               vs the best-matching problem; reported, not a drop criterion, so KB-L1 results can
               be split into ops seen / unseen in SFT (op_overlap_flag: score >= 0.5).
Rows with format `autokernel` have no module (their reference is upstream reference.py), so they
get flag=false with `skipped: "autokernel format"`; their family tags are still reported.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "results" / "kernelbench_l1_problems.jsonl"
VAL_DIR = ROOT / "results" / "v3" / "val_set"
VAL_JSON = ROOT / "results" / "v3" / "val_set.json"
THRESH = 0.5
MIN_SHARED = 2
DF_FRAC = 0.05

KEEP = set(dir(builtins)) | {"self", "torch", "nn", "F", "math", "np", "numpy", "triton", "tl",
                             "get_inputs", "get_init_inputs", "forward", "__init__", "cls"}
MOD_ALIAS = {"torch.nn.functional": "F", "torch.nn": "nn", "torch": "torch", "numpy": "np",
             "math": "math", "triton.language": "tl", "triton": "triton"}
BOILER = re.compile(r"^(def |class |@|pass$|return \[\]$|return$|super\(\)\.__init__\(\)$|"
                    r"return \[v0\]$|return v0$|else:$|try:$|except)")


class _Norm(ast.NodeTransformer):
    def __init__(self, numbers: bool):
        self.numbers = numbers
        self.alias = {}
        self.g = {}
        self.scope = None      # per-function name map
        self.attrs = None      # per-class self-attr map

    # ---- helpers
    def _gname(self, n):
        if n not in self.g:
            self.g[n] = f"g{len(self.g)}"
        return self.g[n]

    def _vname(self, n):
        if n in KEEP:
            return n
        if self.scope is None:
            return self._gname(n)
        if n in self.scope:
            return self.scope[n]
        if n in self.g:
            return self.g[n]
        self.scope[n] = f"v{len(self.scope)}"
        return self.scope[n]

    # ---- module
    def visit_Module(self, node):
        body = []
        for st in node.body:
            if isinstance(st, ast.Import):
                for a in st.names:
                    if a.name in MOD_ALIAS:
                        self.alias[a.asname or a.name.split(".")[0]] = MOD_ALIAS[a.name] if a.asname else a.name.split(".")[0]
                continue
            if isinstance(st, ast.ImportFrom):
                for a in st.names:
                    full = f"{st.module}.{a.name}" if st.module else a.name
                    if full in MOD_ALIAS:
                        self.alias[a.asname or a.name] = MOD_ALIAS[full]
                continue
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                continue
            body.append(st)
        # pre-register module-level names in order
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and st.name not in KEEP:
                self._gname(st.name)
            elif isinstance(st, ast.Assign):
                for t in st.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name) and n.id not in KEEP:
                            self._gname(n.id)
        node.body = [b for b in (self.visit(st) for st in body) if b is not None] or [ast.Pass()]
        return node

    def _strip_doc(self, body):
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        return body or [ast.Pass()]

    def visit_ClassDef(self, node):
        if node.name not in KEEP:
            node.name = self._gname(node.name)
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = []
        node.decorator_list = []
        prev = self.attrs
        self.attrs = {}
        node.body = [b for b in (self.visit(st) for st in self._strip_doc(node.body)) if b is not None] or [ast.Pass()]
        self.attrs = prev
        return node

    def visit_FunctionDef(self, node):
        if self.scope is None and self.attrs is None and node.name not in KEEP:
            node.name = self._gname(node.name)
        prev = self.scope
        self.scope = {}
        node.returns = None
        node.decorator_list = []
        args = node.args
        for a in args.posonlyargs + args.args + args.kwonlyargs + [x for x in (args.vararg, args.kwarg) if x]:
            a.annotation = None
            a.arg = self._vname(a.arg)
        args.defaults = [self.visit(d) for d in args.defaults]
        args.kw_defaults = [self.visit(d) if d is not None else None for d in args.kw_defaults]
        node.body = [b for b in (self.visit(st) for st in self._strip_doc(node.body)) if b is not None] or [ast.Pass()]
        self.scope = prev
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[self.visit(node.target)], value=self.visit(node.value)), node)

    def visit_Name(self, node):
        if node.id in self.alias:
            node.id = self.alias[node.id]
        else:
            node.id = self._vname(node.id)
        return node

    def visit_Attribute(self, node):
        # torch.nn.functional.x -> F.x ; torch.nn.x -> nn.x
        dotted = []
        n = node
        while isinstance(n, ast.Attribute):
            dotted.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            full = ".".join([self.alias.get(n.id, n.id)] + list(reversed(dotted)))
            for long, short in (("torch.nn.functional.", "F."), ("nn.functional.", "F."), ("torch.nn.", "nn.")):
                if full.startswith(long) and n.id not in (self.scope or {}):
                    rest = full[len(long):].split(".")
                    expr = ast.Name(id=short[:-1], ctx=ast.Load())
                    for r in rest:
                        expr = ast.Attribute(value=expr, attr=r, ctx=ast.Load())
                    return ast.copy_location(expr, node)
            if n.id == "self" and len(dotted) >= 1 and self.attrs is not None:
                first = dotted[-1]
                if first not in self.attrs:
                    self.attrs[first] = f"a{len(self.attrs)}"
                # rebuild self.<a_i>.rest
                expr = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=self.attrs[first], ctx=node.ctx)
                for r in reversed(dotted[:-1]):
                    expr = ast.Attribute(value=expr, attr=r, ctx=node.ctx)
                return ast.copy_location(expr, node)
        node.value = self.visit(node.value)
        return node

    def visit_Call(self, node):
        # super(X, self) -> super()
        if isinstance(node.func, ast.Name) and node.func.id == "super":
            node.args = []
            return node
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        if not self.numbers and isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=0), node)
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        self.generic_visit(node)
        return node


class View:
    """Normalized view of one source: lines, Model-class dump, forward/reference body dump."""

    def __init__(self, src: str, numbers: bool):
        self.ok = True
        try:
            tree = ast.parse(src)
        except SyntaxError:
            self.ok, self.lines, self.cls_dumps, self.body_dumps = False, [], set(), set()
            return
        tree = _Norm(numbers).visit(tree)
        ast.fix_missing_locations(tree)
        txt = ast.unparse(tree)
        self.lines = [l.strip() for l in txt.splitlines() if l.strip()]
        self.cls_dumps, self.body_dumps = set(), set()
        self.calls = set()
        for st in tree.body:
            if isinstance(st, (ast.ClassDef, ast.FunctionDef)) and getattr(st, "name", "") not in (
                    "get_inputs", "get_init_inputs"):
                for n in ast.walk(st):
                    if isinstance(n, ast.Call):
                        f = n.func
                        if isinstance(f, ast.Attribute):
                            base = f.value
                            if isinstance(base, ast.Name) and base.id in ("torch", "F", "nn"):
                                self.calls.add(f"{base.id}.{f.attr}")
                            elif f.attr not in ("__init__",) and not (isinstance(base, ast.Name) and base.id == "self"):
                                self.calls.add("." + f.attr)
                    elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
                        self.calls.add("@")
        for st in tree.body:
            if isinstance(st, ast.ClassDef):
                d = ast.dump(ast.Module(body=st.body, type_ignores=[]))
                self.cls_dumps.add(d)
                for f in st.body:
                    if isinstance(f, ast.FunctionDef) and f.name == "forward":
                        self.body_dumps.add(self._body(f, drop_self=True))
            elif isinstance(st, ast.FunctionDef) and st.name not in ("get_inputs", "get_init_inputs"):
                self.body_dumps.add(self._body(st, drop_self=False))

    @staticmethod
    def _body(f, drop_self):
        # args renamed by position so forward(self, x) and reference(x) agree
        names = [a.arg for a in f.args.args]
        if drop_self and names and names[0] == "self":
            names = names[1:]
        src = ast.unparse(ast.Module(body=f.body, type_ignores=[]))
        for i, n in enumerate(names):
            src = re.sub(rf"\b{re.escape(n)}\b", f"p{i}", src)
        return src


def _distinct(lines, common):
    return {l for l in lines if not BOILER.match(l) and l not in common}


def load_refs():
    refs = []
    for line in open(KB_PATH):
        r = json.loads(line)
        refs.append({"kind": "kb_l1", "problem": f"kb_l1/{r['problem_id']}", "src": r["code"]})
    vj = json.load(open(VAL_JSON))
    dirs = {p["dir"]: p["id"] for p in vj["problems"]}
    for d in sorted(VAL_DIR.iterdir()):
        f = d / "reference.py"
        if f.exists():
            refs.append({"kind": "val", "problem": f"val/{dirs.get(d.name, d.name)}", "src": f.read_text()})
    val_fams = sorted({p["family"] for p in vj["problems"]})
    return refs, val_fams


class Index:
    def __init__(self):
        self.refs, self.val_fams = load_refs()
        for r in self.refs:
            r["v"] = View(r["src"], numbers=True)
        n = len(self.refs)
        df = Counter()
        for r in self.refs:
            df.update(set(r["v"].lines))
        self.common = {l for l, c in df.items() if c >= max(2, DF_FRAC * n)}
        for r in self.refs:
            r["D"] = _distinct(r["v"].lines, self.common)

    def check(self, row):
        out = {"id": row.get("id"), "source": row.get("source"), "format": row.get("format")}
        fam = row.get("family") or []
        out["families_in_val"] = sorted(set(fam) & set(self.val_fams) |
                                        ({"rotary_embedding"} if "rotary" in fam and "rotary_embedding" in self.val_fams else set()))
        if row.get("format") == "autokernel":
            out.update(flag=False, skipped="autokernel format", overlap=0.0, match=None)
            return out
        v = View(row.get("reference", ""), True)
        if not v.ok:
            out.update(flag=False, skipped="reference does not parse", overlap=0.0, match=None)
            return out
        D = _distinct(v.lines, self.common)
        best, best_op = None, None
        for r in self.refs:
            shared = len(D & r["D"])
            ov = shared / len(r["D"]) if r["D"] else 0.0
            ast_eq = bool((r["v"].cls_dumps & v.cls_dumps) if r["kind"] == "kb_l1"
                          else (r["v"].body_dumps & v.body_dumps))
            near = ast_eq or (ov >= THRESH and shared >= MIN_SHARED)
            cand = (near, ast_eq, ov, shared)
            if best is None or cand > best[0]:
                best = (cand, r)
            u = v.calls | r["v"].calls
            ov0 = len(v.calls & r["v"].calls) / len(u) if u else 0.0
            if best_op is None or ov0 > best_op[0]:
                best_op = (ov0, r)
        (near, ast_eq, ov, shared), r = best
        out.update(flag=bool(near), match=r["problem"] if (near or ov > 0) else None, match_kind=r["kind"],
                   overlap=round(ov, 3), shared_lines=shared, problem_distinct_lines=len(r["D"]),
                   ast_equal=ast_eq, near_dup_kb=bool(near and r["kind"] == "kb_l1"),
                   near_dup_val=bool(near and r["kind"] == "val"),
                   op_overlap={"problem": best_op[1]["problem"], "score": round(best_op[0], 3)},
                   op_overlap_flag=best_op[0] >= THRESH)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default=None)
    a = ap.parse_args(argv)
    p = pathlib.Path(a.inp)
    src = a.source or re.sub(r"^(candidates|verified)_", "", p.stem)
    out = pathlib.Path(a.out) if a.out else p.parent / f"decontam_{src}.jsonl"
    idx = Index()
    n, flags, kinds = 0, 0, Counter()
    with out.open("w") as f:
        for line in open(p):
            row = json.loads(line)
            r = idx.check(row)
            n += 1
            if r["flag"]:
                flags += 1
                kinds[r["match_kind"]] += 1
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"input": str(p), "out": str(out), "n": n, "flagged": flags, "by_kind": dict(kinds)}))


if __name__ == "__main__":
    main(sys.argv[1:])
