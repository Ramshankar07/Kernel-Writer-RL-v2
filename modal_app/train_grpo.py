"""
GRPO for the AutoKernel agent on Modal (LoRA, 1 trainer H100).

Replaces the verl job in src/autokernel-rlvr/skypilot/trainer.yaml with a small
loop that keeps verl's GRPO semantics:
  * adv_estimator=grpo: A = (r - mean_group) / (std_group + 1e-6), one scalar per
    trajectory, broadcast to every assistant token (response_mask: tool/observation
    tokens get no gradient)
  * one on-policy update per batch (ppo ratio == 1, so clipping is inactive),
    loss_agg_mode=token-mean, no KL term (verl GRPO default use_kl_loss=False)
  * reward = src/autokernel-rlvr/agent_loop/reward.py:autokernel_reward, unchanged

group=8 (not verl's 4-8 low end): in the base-model eval 78% of 4-sample groups had
identical rewards (zero advantage, no gradient) vs 67% at 8. max_turns=4: pass@turn was
flat after turn 1 for the base model, so 8 turns mostly buys context growth.

Topology (account cap 10 GPUs): trainer 1xH100 + policy (vLLM) 1xH100 + bench <=8xL4.
The whole loop runs inside the trainer container, so the laptop can disconnect.

    modal run --detach modal_app/train_grpo.py --run grpo_v1 --steps 30
"""
import json
import os
import pathlib
import time

import modal

from common import VOL, volume

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "peft==0.15.2", "accelerate",
                 "hf_transfer", "pandas", "httpx")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{VOL}/hf", "AK_ROOT": "/ak",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(ROOT / "src" / "autokernel-rlvr", "/ak/src/autokernel-rlvr")
    .add_local_dir(ROOT / "results" / "autokernel_src", "/ak/results/autokernel_src")
    .add_local_python_source("common", "agent_eval", "reward_v2")
)

app = modal.App("autokernel-grpo")


def build_masked(tok, messages):
    """Token ids of the conversation up to the last assistant turn + assistant-token mask."""
    last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    msgs = messages[: last + 1]
    ids = tok.apply_chat_template(msgs, tokenize=True)
    mask = [0] * len(ids)
    mismatches = 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        pre = tok.apply_chat_template(msgs[:i], tokenize=True, add_generation_prompt=True)
        upto = tok.apply_chat_template(msgs[: i + 1], tokenize=True)
        if ids[: len(pre)] != pre:
            mismatches += 1
        # assistant span incl. <|im_end|>, excluding the template's trailing "\n"
        for p in range(len(pre), len(upto) - 1):
            mask[p] = 1
    return ids, mask, mismatches


def token_logps(model, ids, mask, chunk=2048):
    """log p(token) at assistant positions; lm_head applied in chunks (152k vocab)."""
    import torch
    base = model.get_base_model()
    x = torch.tensor([ids], device="cuda")
    h = base.model(input_ids=x, use_cache=False).last_hidden_state[0]
    m = torch.tensor(mask, device="cuda").bool()
    pos = m[1:].nonzero().squeeze(-1)          # h[p] predicts token p+1
    tgt = x[0, pos + 1]
    out = []
    for c in range(0, len(pos), chunk):
        logits = base.lm_head(h[pos[c:c + chunk]]).float()
        out.append(torch.log_softmax(logits, -1).gather(-1, tgt[c:c + chunk, None]).squeeze(-1))
    return torch.cat(out)


def grpo_advantages(trajs, min_std: float = 0.0):
    """min_std: groups whose reward std is below it count as ties (advantage 0). v2 uses
    0.02 ~ bench timing noise (stage1: <=1.9% CV on speedup ~ 0.027 in log2 units), so
    std-normalisation can't turn measurement noise into full-size advantages."""
    import statistics as st
    groups = {}
    for t in trajs:
        groups.setdefault(t["task"]["task_id"], []).append(t)
    zero_groups = 0
    for g in groups.values():
        rs = [t["train_reward"] for t in g]
        mu, sd = st.mean(rs), st.pstdev(rs)
        tie = sd <= min_std or sd == 0
        zero_groups += tie
        for t in g:
            t["advantage"] = 0.0 if tie else (t["train_reward"] - mu) / (sd + 1e-6)
    return zero_groups / len(groups)


@app.function(image=train_image, gpu="H100", timeout=24 * 3600, memory=65536,
              volumes={VOL: volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def train(run: str = "grpo_v1", steps: int = 30, group: int = 8, max_turns: int = 4,
          lr: float = 1e-5, lora_r: int = 64, max_new: int = 3072, temperature: float = 1.0,
          bench_gpu: str = "L4", save_opt_every: int = 2, reward: str = "v1",
          adv_min_std: float = 0.0):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agent_eval import autokernel_tasks, run_episodes, write_trajs
    from reward_v2 import reward_v2

    run_dir = pathlib.Path(VOL) / "grpo" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "train.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    cfg = dict(run=run, steps=steps, group=group, max_turns=max_turns, lr=lr, lora_r=lora_r,
               max_new=max_new, temperature=temperature, bench_gpu=bench_gpu, model=MODEL,
               reward=reward, adv_min_std=adv_min_std)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()

    ckpts = sorted((p for p in run_dir.glob("step_*") if (p / "adapter_config.json").exists()),
                   key=lambda p: int(p.name.split("_")[1]))
    start, adapter_path = 0, ""
    if ckpts:
        model = PeftModel.from_pretrained(base, str(ckpts[-1]), is_trainable=True)
        start, adapter_path = int(ckpts[-1].name.split("_")[1]), str(ckpts[-1])
        log(f"resumed from {ckpts[-1]}")
    else:
        model = get_peft_model(base, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    opt_path = run_dir / "optimizer.pt"
    if start and opt_path.exists():
        opt.load_state_dict(torch.load(opt_path))
        log("optimizer state restored")
    log(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M, start step {start}")

    pol = modal.Cls.from_name("autokernel-policy", "Policy")(model=MODEL, enable_lora=True)
    bench = modal.Cls.from_name("autokernel-bench", "Bencher").with_options(gpu=bench_gpu)()
    tasks = autokernel_tasks(max_turns)

    for step in range(start, steps):
        t0 = time.time()
        lora_id = step + 1

        def gen_fn(convs):
            outs = pol.generate.remote(convs, n=1, temperature=temperature, max_tokens=max_new,
                                       lora_path=adapter_path, lora_id=lora_id if adapter_path else 0)
            return [o[0] for o in outs]

        trajs = run_episodes(tasks, group, max_turns, gen_fn, bench, "autokernel",
                             max_new=max_new, log=log)
        t_roll = time.time() - t0
        for t in trajs:   # t["reward"] is always the original reward.py value
            t["reward_v2"] = reward_v2(t["turns"])
            t["train_reward"] = t["reward_v2"] if reward == "v2" else t["reward"]
        zero_frac = grpo_advantages(trajs, adv_min_std)

        # ---- policy-gradient update (token-mean over the whole batch) ----
        t1 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        items = [t for t in trajs if abs(t["advantage"]) > 1e-8]
        built = [(t, *build_masked(tok, t["messages"])) for t in items]
        total_tok = max(1, sum(sum(m) for _, _, m, _ in built))
        loss_sum, logp_sum, n_tok, mism, max_len = 0.0, 0.0, 0, 0, 0
        for t, ids, mask, mm in built:
            mism += mm
            max_len = max(max_len, len(ids))
            lp = token_logps(model, ids, mask)
            loss = -(t["advantage"] * lp.sum()) / total_tok
            loss.backward()
            loss_sum += loss.item()
            logp_sum += lp.detach().sum().item()
            n_tok += lp.numel()
            del lp, loss
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if built else 0.0
        if built:
            opt.step()
        t_train = time.time() - t1

        ck = run_dir / f"step_{step + 1}"
        model.save_pretrained(str(ck))
        if (step + 1) % save_opt_every == 0 or step + 1 == steps:
            torch.save(opt.state_dict(), opt_path)
        for old in run_dir.glob("step_*"):   # keep last 2 + every 10th
            k = int(old.name.split("_")[1])
            if k < step and k % 10 != 0:
                import shutil
                shutil.rmtree(old, ignore_errors=True)
        adapter_path = str(ck)

        turns = [x for t in trajs for x in t["turns"]]
        per_kernel = {}
        for t in trajs:
            per_kernel.setdefault(t["task"]["kernel_type"], []).append(t)
        m = {
            "step": step + 1, "reward_mean": sum(t["reward"] for t in trajs) / len(trajs),
            "reward_v2_mean": sum(t["reward_v2"] for t in trajs) / len(trajs),
            "pass_rate": sum(t["any_pass"] for t in trajs) / len(trajs),
            "turn_pass_rate": sum(x["correctness"] == "PASS" for x in turns) / len(turns),
            "crash_rate": sum(x["correctness"] == "CRASH" for x in turns) / len(turns),
            "best_speedup_mean": sum(t["best_speedup"] for t in trajs) / len(trajs),
            "best_speedup_max": max(t["best_speedup"] for t in trajs),
            "zero_adv_group_frac": zero_frac, "n_trained_trajs": len(built),
            "assistant_tokens": total_tok, "max_seq_len": max_len, "template_mismatch": mism,
            "mean_token_logp": logp_sum / max(1, n_tok), "loss": loss_sum,
            "grad_norm": grad_norm, "cache_hit_rate": sum(bool(x["cached"]) for x in turns) / len(turns),
            "completion_tokens_mean": sum(x["completion_tokens"] for x in turns) / len(turns),
            "rollout_s": t_roll, "train_s": t_train,
            "per_kernel_reward": {k: sum(t["reward"] for t in v) / len(v) for k, v in per_kernel.items()},
            "per_kernel_best": {k: max(t["best_speedup"] for t in v) for k, v in per_kernel.items()},
        }
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        write_trajs(trajs, run_dir / "rollouts" / f"step_{step + 1:03d}.jsonl")
        volume.commit()
        log(f"step {step + 1}/{steps} reward {m['reward_mean']:.3f} pass {m['pass_rate']:.2f} "
            f"best {m['best_speedup_max']:.2f}x zero_adv {zero_frac:.2f} "
            f"roll {t_roll:.0f}s train {t_train:.0f}s")


@app.local_entrypoint()
def main(run: str = "grpo_v1", steps: int = 30, group: int = 8, max_turns: int = 4,
         lr: float = 1e-5, bench_gpu: str = "L4", reward: str = "v1", adv_min_std: float = 0.0):
    train.remote(run=run, steps=steps, group=group, max_turns=max_turns, lr=lr,
                 bench_gpu=bench_gpu, reward=reward, adv_min_std=adv_min_std)
