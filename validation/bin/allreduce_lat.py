#!/usr/bin/env python3
"""Cross-node all-reduce latency at decode-sized messages.

Run one rank per node with torchrun, no vLLM server up:

  # spark1 (node 0) and spark2 (node 1), same NCCL/Gloo env as VALIDATION.md T2
  torchrun --nnodes 2 --nproc-per-node 1 --node-rank <0|1> \
    --master-addr <spark1 link address> --master-port 29600 \
    allreduce_lat.py --sizes 10240 16384 --out allreduce.json

--sizes are message sizes in bytes (hidden_size x 2 for one BF16 token).
For each size it reports mean latency per all-reduce, eager (one launch per
op) and graph (K ops captured in one CUDA graph and replayed, as vLLM decode
runs them). Rank 0 writes the JSON. --cpu runs Gloo on CPU for a local
logic test only; it skips the graph measurement.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def eager_us(x, iters, sync):
    for _ in range(20):
        dist.all_reduce(x)
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    sync()
    return (time.perf_counter() - t) / iters * 1e6


def graph_us(x, k, replays):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            dist.all_reduce(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k):
            dist.all_reduce(x)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (replays * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10240, 16384])
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--k", type=int, default=160, help="all-reduces per captured graph")
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="allreduce.json")
    a = ap.parse_args()

    dist.init_process_group("gloo" if a.cpu else "nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if a.cpu:
        dev, dtype, sync = "cpu", torch.float32, (lambda: None)
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dev, dtype, sync = "cuda", torch.bfloat16, torch.cuda.synchronize

    rows = []
    for size in a.sizes:
        n = size // torch.tensor([], dtype=dtype).element_size()
        x = torch.ones(n, dtype=dtype, device=dev)
        dist.all_reduce(x)
        sync()
        ok = bool(torch.all(x == world).item())
        x.fill_(0)
        row = {"size_bytes": size, "correct": ok, "eager_us": round(eager_us(x, a.iters, sync), 2)}
        if not a.cpu:
            try:
                row["graph_us"] = round(graph_us(x, a.k, a.replays), 2)
                row["graph_k"] = a.k
            except Exception as e:  # keep the eager number if capture fails
                row["graph_error"] = f"{type(e).__name__}: {e}"[:300]
                torch.cuda.synchronize()
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)

    if rank == 0:
        meta = {"world": world, "backend": dist.get_backend(), "torch": torch.__version__,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())) if not a.cpu else None,
                "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "GLOO_"))}}
        with open(a.out, "w") as f:
            json.dump({"meta": meta, "results": rows}, f, indent=2)
        print("wrote", a.out)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
