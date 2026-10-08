"""
vLLM policy on 1xH100: batched chat generation for Qwen2.5-Coder-*-Instruct.

LoRA adapters are passed per call (path on the Volume + a unique int id), so
GRPO can hot-swap the adapter every step without restarting the container.
"""
import modal

from common import VOL, volume

MODEL_DIR = f"{VOL}/hf"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.8.5", "transformers==4.51.3", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": MODEL_DIR})
    .add_local_python_source("common")
)

app = modal.App("autokernel-policy")


@app.cls(image=vllm_image, gpu="H100", timeout=6 * 3600, max_containers=1,
         volumes={VOL: volume}, secrets=[modal.Secret.from_name("huggingface-secret")],
         scaledown_window=600)
class Policy:
    model: str = modal.parameter(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    enable_lora: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model=self.model, max_model_len=32768, gpu_memory_utilization=0.9,
                       enable_lora=self.enable_lora, max_lora_rank=64, max_loras=1,
                       enable_prefix_caching=True)

    @modal.method()
    def generate(self, conversations: list, n: int = 1, temperature: float = 0.8,
                 max_tokens: int = 4096, seed: int | None = None,
                 lora_path: str = "", lora_id: int = 0) -> list:
        """Returns, per conversation, n dicts {text, prompt_tokens, completion_tokens}."""
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        sp = SamplingParams(n=n, temperature=temperature, top_p=0.95,
                            max_tokens=max_tokens, seed=seed)
        lora = None
        if lora_path:
            volume.reload()   # adapter was just written by the trainer container
            lora = LoRARequest(f"ckpt{lora_id}", lora_id, lora_path)
        outs = self.llm.chat(conversations, sp, lora_request=lora, use_tqdm=False)
        return [[{"text": c.text, "prompt_tokens": len(o.prompt_token_ids),
                  "completion_tokens": len(c.token_ids), "finish": c.finish_reason}
                 for c in o.outputs] for o in outs]
