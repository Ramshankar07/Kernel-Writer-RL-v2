"""
Format test for sft_assemble.py outputs (Phase 1 step 6).

    python modal_app/test_sft_assemble.py [results/v3/sft/sft_train.jsonl] [--n 5]

Renders N random examples (fixed seed) with the Qwen2.5-Coder-7B-Instruct tokenizer's
apply_chat_template and asserts, per example:
  * roles are system, user, assistant[, user, assistant] and loss_mask/train_on mark only the
    final assistant message;
  * the rendered text ends with that final assistant content + <|im_end|>;
  * the final assistant content ends with a closing ``` and holds exactly one ```python block,
    whose code parses with ast (and, for autokernel rows, defines kernel_fn;
    for modelnew rows, class ModelNew);
  * n_tokens matches a fresh tokenization and is <= 4096.
Also runs a synthetic repair row through build_messages (no data needed).
"""
import ast
import json
import pathlib
import random
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sft_assemble as sa  # noqa: E402

BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def check_example(tok, r):
    msgs = r["messages"]
    roles = [m["role"] for m in msgs]
    assert roles in (["system", "user", "assistant"],
                     ["system", "user", "assistant", "user", "assistant"]), roles
    assert r["is_repair"] == (len(msgs) == 5), r["id"]
    assert r["loss_mask"] == [0] * (len(msgs) - 1) + [1] and r["train_on"] == len(msgs) - 1, r["id"]
    final = msgs[-1]["content"]
    assert final.rstrip().endswith("```"), f"{r['id']}: final assistant does not end with ```"
    blocks = BLOCK.findall(final)
    assert len(blocks) == 1, f"{r['id']}: {len(blocks)} python blocks"
    tree = ast.parse(blocks[0])
    defs = {t.name for t in ast.walk(tree) if isinstance(t, (ast.FunctionDef, ast.ClassDef))}
    if r["format"] == "autokernel":
        assert "kernel_fn" in defs, r["id"]  # KERNEL_TYPE is optional: bench v2 only imports kernel_fn
    else:
        assert "ModelNew" in defs, r["id"]
    text = tok.apply_chat_template(msgs, tokenize=False)
    assert text.rstrip("\n").endswith(final + "<|im_end|>"), f"{r['id']}: rendered text does not end with final turn"
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    assert n == r["n_tokens"] and n <= sa.MAX_TOKENS, (r["id"], n, r["n_tokens"])
    return text


def test_synthetic_repair():
    row = {"id": "x:1", "format": "modelnew", "family": ["softmax"], "kernel_type": None,
           "reference": "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x.softmax(-1)\n",
           "target": "import triton\n@triton.jit\ndef k(x):\n    pass\nclass ModelNew:\n    pass\n",
           "repair": {"failed_code": "class ModelNew:\n    pass\n", "feedback": '{"correctness": false}'}}
    m = sa.build_messages(row)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant"]
    assert m[3]["content"].startswith("bench result:\n")
    assert m[4]["content"].endswith("```") and "(1 @triton.jit kernel)" in m[4]["content"]
    row["repair"]["feedback"] = "bench result:\n{}"
    assert sa.build_messages(row)[3]["content"].count("bench result:") == 1


def main(argv):
    path = pathlib.Path(argv[0]) if argv and not argv[0].startswith("--") else sa.SFT / "sft_train.jsonl"
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 5
    test_synthetic_repair()
    rows = [json.loads(l) for l in open(path)]
    assert rows, f"{path} is empty"
    tok = sa.load_tokenizer()
    rng = random.Random(0)
    picks = rng.sample(rows, min(n, len(rows)))
    if not any(r["is_repair"] for r in picks):  # always cover the repair format when present
        rep = [r for r in rows if r["is_repair"]]
        if rep:
            picks.append(rng.choice(rep))
    for r in picks:
        check_example(tok, r)
        print(f"ok  {r['id']:<40} {r['format']:<10} {r['primary_family']:<14} repair={r['is_repair']!s:<5} tokens={r['n_tokens']}")
    print(f"PASS: {len(picks)} examples from {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
