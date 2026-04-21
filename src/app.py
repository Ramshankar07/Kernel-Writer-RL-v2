"""
AutoKernel RLVR — bench queue service.

Endpoints:
    POST /bench                 enqueue a job  -> {job_id, cached: bool, reward?: float}
    GET  /bench/{job_id}        poll         -> {status: "pending|done|error", reward?, raw?}
    POST /bench/{job_id}/result worker posts  -> {ok: true}
    GET  /bench/next            worker pulls  -> {job_id, kernel_type, code} | 204
    GET  /stats                               -> queue depth, cache hit rate
    GET  /healthz                             -> "ok"

Reward semantics live in the client (verl agent loop), NOT here. This service
only shuttles { correctness, speedup_vs_pytorch, pct_peak, latency_us } back.
The agent-loop turns that into a scalar. Keeping the shaping on the trainer
side means you can change the reward without restarting bench workers.
"""
import hashlib
import json
import os
import time
import uuid
from typing import Optional

import redis
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)

# Keys:
#   cache:{hash}            -> json result (persistent; same code => same reward)
#   job:{id}                -> json {status, kernel_type, code, hash, result?}
#   queue                   -> list of job_ids awaiting a worker
#   stats:enq, stats:hit    -> counters

CACHE_TTL = 7 * 24 * 3600   # evict week-old results so the cache doesn't balloon
JOB_TTL = 6 * 3600          # jobs live 6h in redis

app = FastAPI()


class BenchRequest(BaseModel):
    kernel_type: str          # e.g. "matmul", "flash_attention"
    code: str                 # the Triton kernel source to evaluate
    # Optional: a trajectory-level tag so you can filter metrics later.
    meta: Optional[dict] = None


class BenchResult(BaseModel):
    correctness: str          # "PASS" | "FAIL" | "TIMEOUT" | "CRASH"
    speedup_vs_pytorch: float = 0.0
    pct_peak: float = 0.0
    latency_us: float = 0.0
    throughput_tflops: float = 0.0
    raw: Optional[str] = None  # stderr/stdout for debugging


def _hash(kernel_type: str, code: str) -> str:
    h = hashlib.sha256()
    h.update(kernel_type.encode())
    h.update(b"\x00")
    h.update(code.encode())
    return h.hexdigest()


@app.get("/healthz")
def healthz():
    return "ok"


@app.post("/bench")
def enqueue(req: BenchRequest):
    r.incr("stats:enq")
    h = _hash(req.kernel_type, req.code)

    cached = r.get(f"cache:{h}")
    if cached:
        r.incr("stats:hit")
        return {"job_id": None, "cached": True, "result": json.loads(cached)}

    job_id = uuid.uuid4().hex
    payload = {
        "status": "pending",
        "kernel_type": req.kernel_type,
        "code": req.code,
        "hash": h,
        "meta": req.meta or {},
        "created_at": time.time(),
    }
    r.setex(f"job:{job_id}", JOB_TTL, json.dumps(payload))
    r.rpush("queue", job_id)
    return {"job_id": job_id, "cached": False}


@app.get("/bench/next")
def pull_job():
    # Blocking pop with a short timeout would be nicer; keep it simple.
    job_id = r.lpop("queue")
    if not job_id:
        return Response(status_code=204)
    raw = r.get(f"job:{job_id}")
    if not raw:
        # expired between pop and read — skip
        return Response(status_code=204)
    payload = json.loads(raw)
    return {
        "job_id": job_id,
        "kernel_type": payload["kernel_type"],
        "code": payload["code"],
    }


@app.post("/bench/{job_id}/result")
def post_result(job_id: str, result: BenchResult):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(404, "job expired or unknown")
    payload = json.loads(raw)
    payload["status"] = "done"
    payload["result"] = result.dict()
    r.setex(f"job:{job_id}", JOB_TTL, json.dumps(payload))
    # Cache by content hash so future identical proposals are free.
    r.setex(f"cache:{payload['hash']}", CACHE_TTL, json.dumps(result.dict()))
    return {"ok": True}


@app.get("/bench/{job_id}")
def poll(job_id: str):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(404, "unknown job_id")
    payload = json.loads(raw)
    return {
        "status": payload["status"],
        "result": payload.get("result"),
    }


@app.get("/stats")
def stats():
    enq = int(r.get("stats:enq") or 0)
    hit = int(r.get("stats:hit") or 0)
    depth = r.llen("queue")
    return {
        "enqueued_total": enq,
        "cache_hits": hit,
        "hit_rate": (hit / enq) if enq else 0.0,
        "queue_depth": depth,
    }
