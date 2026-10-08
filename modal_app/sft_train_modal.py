"""
SFT Phase 2 (results/v3/sft_plan.md): LoRA SFT of Qwen2.5-Coder-7B-Instruct on results/v3/sft/sft_train.jsonl.

Recipe (plan + sft_data_research.md §3): LoRA r=64, alpha=128, all linear layers; lr 1e-4 cosine with
3% warmup; 2 epochs; max length 4096; loss only on the `train_on` assistant message (the last one;
repair rows mask the failed attempt and the feedback). Token-mean loss over each optimizer step.

The pilot is the first `max_steps` optimizer steps of the full 2-epoch schedule. Every checkpoint
stores adapter + optimizer + scheduler + data position, so the full run resumes from the pilot.

    cd modal_app && modal deploy sft_train_modal.py
    python3 -c "import modal; modal.Function.from_name('autokernel-sft-train','train').spawn(run='sft_v1', max_steps=100)"
    # later, same run name, no max_steps -> resumes and finishes 2 epochs
"""
import json
import math
import os
import pathlib
import random
import time

import modal

from common import VOL, volume

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "peft==0.15.2", "accelerate", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{VOL}/hf",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(ROOT / "results/v3/sft/sft_train.jsonl", "/data/sft_train.jsonl")
    .add_local_file(ROOT / "results/v3/sft/sft_dev.jsonl", "/data/sft_dev.jsonl")
    .add_local_python_source("common")
)
app = modal.App("autokernel-sft-train")

MAX_LEN = 4096
MICRO_BS = 4          # sequences per forward pass (length-bucketed to limit padding)
ACCUM = 4             # micro-batches per optimizer step -> 16 examples/step
SEED = 0


def encode(tok, ex):
    """input ids + labels; labels are -100 except on the train_on assistant message
    (its content + <|im_end|>, not the template's trailing newline)."""
    msgs, k = ex["messages"], ex["train_on"]
    prefix = tok.apply_chat_template(msgs[:k], tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs[:k + 1], tokenize=True)
    assert full[:len(prefix)] == prefix, ex["id"]
    end = len(full) - 1 if tok.decode(full[-1:]) == "\n" else len(full)
    ids = full[:end]
    labels = [-100] * len(prefix) + ids[len(prefix):]
    return ids[:MAX_LEN], labels[:MAX_LEN]


def batches(examples, epoch):
    """Length-bucketed micro-batches in a seeded random order (same order on resume)."""
    rng = random.Random(SEED + epoch)
    idx = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    mbs = [idx[i:i + MICRO_BS] for i in range(0, len(idx), MICRO_BS)]
    rng.shuffle(mbs)
    return mbs


def collate(examples, mb, pad_id):
    import torch
    L = max(len(examples[i][0]) for i in mb)
    ids = torch.full((len(mb), L), pad_id, dtype=torch.long)
    lab = torch.full((len(mb), L), -100, dtype=torch.long)
    att = torch.zeros((len(mb), L), dtype=torch.long)
    for r, i in enumerate(mb):
        x, y = examples[i]
        ids[r, :len(x)] = torch.tensor(x)
        lab[r, :len(y)] = torch.tensor(y)
        att[r, :len(x)] = 1
    return ids, lab, att


def sum_nll(model, ids, lab, att, chunk=4096):
    """Summed NLL over labelled tokens; lm_head applied in chunks (152k vocab)."""
    import torch
    base = model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    h, tgt = h[:, :-1], lab[:, 1:]
    sel = tgt != -100
    hs, ts = h[sel], tgt[sel]
    total = h.new_zeros((), dtype=torch.float32)
    for c in range(0, hs.shape[0], chunk):
        logits = base.lm_head(hs[c:c + chunk]).float()
        total = total + torch.nn.functional.cross_entropy(logits, ts[c:c + chunk], reduction="sum")
    return total, int(sel.sum())


@app.function(image=image, gpu="H100", timeout=24 * 3600, memory=65536,
              volumes={VOL: volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def train(run: str = "sft_v1", max_steps: int = 0, epochs: int = 2, lr: float = 1e-4,
          lora_r: int = 64, warmup_frac: float = 0.03, ckpt_every_frac: float = 0.5, log_every: int = 5):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    out = pathlib.Path(VOL) / "sft" / run
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    train_rows = [json.loads(l) for l in open("/data/sft_train.jsonl")]
    dev_rows = [json.loads(l) for l in open("/data/sft_dev.jsonl")]
    train_ex = [encode(tok, r) for r in train_rows]
    dev_ex = [encode(tok, r) for r in dev_rows]
    log(f"tokenized {len(train_ex)} train / {len(dev_ex)} dev in {time.time() - t0:.0f}s; "
        f"train tokens {sum(len(x) for x, _ in train_ex)}, trained {sum(sum(t != -100 for t in y) for _, y in train_ex)}")

    steps_per_epoch = math.ceil(math.ceil(len(train_ex) / MICRO_BS) / ACCUM)
    total_steps = steps_per_epoch * epochs
    ckpt_every = max(1, int(steps_per_epoch * ckpt_every_frac))
    cfg = dict(run=run, model=MODEL, epochs=epochs, lr=lr, lora_r=lora_r, lora_alpha=2 * lora_r,
               warmup_frac=warmup_frac, micro_bs=MICRO_BS, accum=ACCUM, max_len=MAX_LEN,
               steps_per_epoch=steps_per_epoch, total_steps=total_steps, ckpt_every=ckpt_every,
               n_train=len(train_ex), n_dev=len(dev_ex), seed=SEED)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    ckpts = sorted((p for p in out.glob("step_*") if (p / "trainer_state.json").exists()),
                   key=lambda p: int(p.name.split("_")[1]))
    if ckpts:
        model = PeftModel.from_pretrained(base, str(ckpts[-1]), is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(warmup_frac * total_steps)), total_steps)
    step = 0
    if ckpts:
        st = json.loads((ckpts[-1] / "trainer_state.json").read_text())
        opt.load_state_dict(torch.load(ckpts[-1] / "optimizer.pt"))
        sched.load_state_dict(torch.load(ckpts[-1] / "scheduler.pt"))
        step = st["step"]
        log(f"resumed from {ckpts[-1].name} (step {step}/{total_steps})")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def dev_loss():
        model.eval()
        nll, n = 0.0, 0
        for mb in [list(range(i, min(i + MICRO_BS, len(dev_ex)))) for i in range(0, len(dev_ex), MICRO_BS)]:
            ids, lab, att = (t.cuda() for t in collate(dev_ex, mb, pad_id))
            s, k = sum_nll(model, ids, lab, att)
            nll, n = nll + float(s), n + k
        model.train()
        return nll / max(1, n)

    def save(tag_step):
        ck = out / f"step_{tag_step}"
        model.save_pretrained(str(ck))
        torch.save(opt.state_dict(), ck / "optimizer.pt")
        torch.save(sched.state_dict(), ck / "scheduler.pt")
        (ck / "trainer_state.json").write_text(json.dumps({"step": tag_step}))
        for old in out.glob("step_*"):   # keep optimizer state only on the latest checkpoint
            if old != ck and (old / "optimizer.pt").exists():
                (old / "optimizer.pt").unlink()
        volume.commit()

    if step == 0:
        d0 = dev_loss()
        log(f"step 0 dev_loss {d0:.4f}")
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": 0, "dev_loss": d0}) + "\n")

    model.train()
    stop_at = min(total_steps, step + max_steps) if max_steps else total_steps
    win_tok, win_real, win_t = 0, 0, time.time()
    while step < stop_at:
        epoch = step // steps_per_epoch
        mbs = batches(train_ex, epoch)
        k0 = (step % steps_per_epoch) * ACCUM
        for s_in_epoch in range(k0, len(mbs), ACCUM):
            if step >= stop_at:
                break
            group = mbs[s_in_epoch:s_in_epoch + ACCUM]
            n_tr = sum(sum(t != -100 for t in train_ex[i][1][1:]) for mb in group for i in mb)
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for mb in group:
                ids, lab, att = (t.cuda() for t in collate(train_ex, mb, pad_id))
                s, _ = sum_nll(model, ids, lab, att)
                (s / max(1, n_tr)).backward()
                loss_sum += float(s)
                win_tok += ids.numel()
                win_real += int(att.sum())
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            sched.step()
            step += 1
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": loss_sum / max(1, n_tr),
                   "lr": sched.get_last_lr()[0], "grad_norm": gn}
            if step % log_every == 0 or step == stop_at:
                dt = time.time() - win_t
                rec.update(tok_per_s=round(win_real / dt), padded_tok_per_s=round(win_tok / dt),
                           s_per_step=round(dt / log_every, 2),
                           gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
                log(f"step {step}/{total_steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {gn:.2f} "
                    f"{rec['tok_per_s']} tok/s ({rec['s_per_step']} s/step) mem {rec['gpu_mem_gb']} GB")
                win_tok, win_real, win_t = 0, 0, time.time()
            if step % ckpt_every == 0 or step == stop_at:
                rec["dev_loss"] = dev_loss()
                log(f"step {step} dev_loss {rec['dev_loss']:.4f}; checkpoint")
                save(step)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
    remaining = total_steps - step
    log(f"stopped at step {step}/{total_steps}; remaining {remaining} steps")
    return {"step": step, "total_steps": total_steps}
