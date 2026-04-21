"""
AutoKernel agent loop for verl.

Two pieces:

  1. BenchHTTPTool — the tool verl invokes when the model emits a
     `autokernel_bench(kernel_type=..., code=...)` call. Posts to the
     queue service, long-polls for the reward, returns an observation
     the model can read.

  2. AutoKernelAgentLoop — subclass of verl's AgentLoopBase. Manages the
     per-episode state (best-speedup-so-far, PASS count) and writes it
     into extra_info on the final AgentLoopOutput so reward.py can pull it.

Notes on design:
  * We intentionally do NOT return the reward as a scalar inside the tool
    response. The reward is computed in reward.py from the trajectory's
    record of all bench results. The tool only returns metrics.
  * The agent sees ALL bench responses in-context so it can learn to
    read them and revise. This is the whole point of multi-turn RLVR.
  * `response_mask` = 0 on observation tokens (tool output), 1 on model
    tokens — verl handles this when we return AgentLoopOutput.
"""
import asyncio
import json
import os
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel

# verl imports — these are what the verl docs / verl-tool use. If you're on
# a different verl version, the class names may have drifted; adjust.
from verl.workers.agent.agent_loop.base import AgentLoopBase, AgentLoopOutput
from verl.workers.agent.tool_agent_loop import ToolAgentLoop


class BenchHTTPTool:
    """Invoked by verl's tool-calling machinery. Async: long-polls the queue."""

    def __init__(self, config: dict, tool_schema: dict):
        env_var = config.get("bench_url_env", "BENCH_URL")
        self.url = os.environ.get(env_var, "").rstrip("/")
        if not self.url:
            raise RuntimeError(
                f"BenchHTTPTool: env var {env_var} is not set. "
                f"Pass --env {env_var}=http://<bench-ip>:8000 on sky launch."
            )
        self.poll = float(config.get("poll_interval_sec", 2.0))
        self.max_wait = float(config.get("max_wait_sec", 240))
        self._client = httpx.AsyncClient(timeout=30)

    async def __call__(self, args: dict) -> str:
        """Returns a JSON-serializable string that becomes the observation."""
        kernel_type = args.get("kernel_type", "").strip()
        code = args.get("code", "")
        if not kernel_type or not code:
            return json.dumps({
                "correctness": "CRASH",
                "error": "missing kernel_type or code",
            })

        try:
            r = await self._client.post(
                f"{self.url}/bench",
                json={"kernel_type": kernel_type, "code": code},
            )
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            return json.dumps({"correctness": "CRASH", "error": f"enqueue failed: {e}"})

        if body.get("cached"):
            return json.dumps(body["result"])

        job_id = body["job_id"]
        deadline = time.time() + self.max_wait
        while time.time() < deadline:
            await asyncio.sleep(self.poll)
            try:
                p = await self._client.get(f"{self.url}/bench/{job_id}")
                if p.status_code == 404:
                    return json.dumps({"correctness": "CRASH", "error": "job expired"})
                data = p.json()
                if data.get("status") == "done":
                    return json.dumps(data["result"])
            except Exception:
                # transient — keep polling
                continue
        return json.dumps({"correctness": "TIMEOUT", "error": "queue wait exceeded"})


class AutoKernelAgentLoop(ToolAgentLoop):
    """Episode = up to MAX_TURNS proposals for one kernel_type prompt.

    Subclasses verl's ToolAgentLoop (which already handles the
    assistant-message / tool-call / tool-response token bookkeeping).
    We only override run() to stamp per-trajectory aggregates into
    extra_info so the reward function can consume them.
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        out: AgentLoopOutput = await super().run(sampling_params, **kwargs)

        # Walk the trajectory's tool responses and extract best speedup.
        # verl exposes them via out.extra_info["tool_responses"] in recent
        # versions. If your version differs, you can parse the decoded
        # response text for JSON blocks.
        tool_responses = (out.extra_info or {}).get("tool_responses", [])
        best_speedup = 0.0
        pass_count = 0
        crash_count = 0
        any_pass = False
        for tr in tool_responses:
            try:
                body = json.loads(tr if isinstance(tr, str) else tr.get("content", ""))
            except (json.JSONDecodeError, TypeError):
                continue
            corr = body.get("correctness", "CRASH")
            if corr == "PASS":
                any_pass = True
                pass_count += 1
                best_speedup = max(best_speedup, body.get("speedup_vs_pytorch", 0.0))
            elif corr == "CRASH":
                crash_count += 1

        out.extra_info = out.extra_info or {}

        # Preserve kernel_type from the prompt's extra_info so reward_amd.py
        # can select the correct per-kernel log2 clip ceiling (FP8/MXFP4
        # kernels allow up to log2(16); collective kernels log2(11.3); etc.).
        # verl may merge data extra_info with loop output but we write it
        # explicitly here to guarantee it survives.
        kernel_type = out.extra_info.get("kernel_type") or kwargs.get("extra_info", {}).get("kernel_type", "")

        out.extra_info.update({
            "best_speedup": best_speedup,
            "pass_count": pass_count,
            "crash_count": crash_count,
            "any_pass": any_pass,
            "num_turns": len(tool_responses),
            "kernel_type": kernel_type,
        })
        return out
