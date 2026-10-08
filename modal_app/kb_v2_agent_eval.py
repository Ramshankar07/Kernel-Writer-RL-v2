"""
Phase 3: multi-turn KernelBench L1 eval under the fixed harness (kb v2, all 100 valid problems),
base vs SFT LoRA, with the ModelNew prompt the SFT data used (prompts_sft.py) for both.

Reuses agent_eval.run_episodes unchanged through a small adapter around KBV2.run, which:
  * maps kb v2's `status` -> `correctness`, `speedup` -> `speedup_vs_pytorch`, stage/reason -> fail_lines;
  * adds a static launch check, because kb v2 has none: PASS also requires an @triton.jit kernel
    launched as name[grid](...) (reported as `pass_triton`; raw PASS is kept too).

    modal deploy modal_app/kernelbench_v2_modal.py
    modal run modal_app/kb_v2_agent_eval.py --tag kb2_base
    modal run modal_app/kb_v2_agent_eval.py --tag kb2_sft_v1 --lora /vol/sft/sft_v1/step_378
"""
import ast
import json
import pathlib
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent_eval as AE  # noqa: E402
from prompts_sft import SYSTEM_PROMPT_MODELNEW, user_prompt_modelnew  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results" / "v3" / "phase3_runs"
app = modal.App("autokernel-kb-v2-agent-eval")


def launches_triton(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    jit = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
           and any("jit" in ast.unparse(d) for d in f.decorator_list)}
    for node in ast.walk(tree):   # name[grid](...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in jit:
            return True
    return False


class _Run:
    def __init__(self, kb):
        self.kb = kb

    def starmap(self, args, return_exceptions=True, wrap_returned_exceptions=False):
        calls = [(a[0], a[1], True) for a in args]
        for (pid, code, _), r in zip(calls, self.kb.run.starmap(
                calls, return_exceptions=return_exceptions, wrap_returned_exceptions=wrap_returned_exceptions)):
            if isinstance(r, Exception):
                yield r
                continue
            st = r.get("status", "CRASH")
            r["correctness"] = st
            r["speedup_vs_pytorch"] = float(r.get("speedup") or 0.0) if st == "PASS" else 0.0
            if st != "PASS":
                r["fail_lines"] = [f"{r.get('stage', '')}: {r.get('reason', '')}"[:300]]
            r["pass_triton"] = st == "PASS" and launches_triton(code)
            yield r


class KBAdapter:
    def __init__(self, kb):
        self.run = _Run(kb)


def kb_v2_tasks():
    rows = [json.loads(l) for l in (ROOT / "results" / "kernelbench_l1_problems.jsonl").read_text().splitlines() if l]
    ok = set(json.loads((ROOT / "results" / "v3" / "kb_v2_valid_problems.json").read_text())["valid"])
    return [{"task_id": f"L1_{p['problem_id']}", "suite": "kb", "problem_id": p["problem_id"], "messages": [
        {"role": "system", "content": SYSTEM_PROMPT_MODELNEW},
        {"role": "user", "content": user_prompt_modelnew(p["code"])}]}
        for p in rows if p["problem_id"] in ok]


@app.local_entrypoint()
def main(tag: str, lora: str = "", n: int = 1, max_turns: int = 3, temperature: float = 0.8,
         model: str = "Qwen/Qwen2.5-Coder-7B-Instruct", policy_gpu: str = "L40S", limit: int = 0):
    tasks = kb_v2_tasks()[:limit or None]
    kb = modal.Cls.from_name("autokernel-kernelbench-v2", "KBV2")()
    pol = modal.Cls.from_name("autokernel-policy", "Policy").with_options(gpu=policy_gpu)(
        model=model, enable_lora=bool(lora))

    def gen_fn(convs):
        outs = pol.generate.remote(convs, n=1, temperature=temperature, max_tokens=AE.MAX_NEW,
                                   lora_path=lora, lora_id=1 if lora else 0)
        return [o[0] for o in outs]

    t0 = time.time()
    trajs = AE.run_episodes(tasks, n, max_turns, gen_fn, KBAdapter(kb), "kb")
    for t in trajs:   # pass_triton is per turn in the bench result; lift it to the episode
        t["any_pass_triton"] = any(x.get("correctness") == "PASS" and AE_launch(x) for x in t["turns"])
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{tag}.jsonl"
    AE.write_trajs(trajs, out, keep_messages=True)
    n_t = len(trajs)
    meta = {"tag": tag, "model": model, "lora": lora or None, "n": n, "max_turns": max_turns,
            "temperature": temperature, "tasks": len(tasks), "harness": "kb v2 (kb_v2_eval.py)",
            "prompt": "prompts_sft.SYSTEM_PROMPT_MODELNEW", "wall_s": round(time.time() - t0),
            "fast_0_raw": sum(t["any_pass"] for t in trajs) / n_t,
            "fast_0_triton": sum(t["any_pass_triton"] for t in trajs) / n_t,
            "fast_1_triton": sum(any(x.get("correctness") == "PASS" and AE_launch(x) and x["speedup"] > 1.0
                                     for x in t["turns"]) for t in trajs) / n_t}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, json.dumps(meta), flush=True)


def AE_launch(turn):
    return bool(turn.get("code")) and launches_triton(turn["code"])
