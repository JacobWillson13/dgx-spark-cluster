# Cluster Validation Suite

Automated functional validation for the two-node DGX Spark cluster. A coding
agent with SSH access to `spark1` and `spark2` runs the suite end to end from
the MacBook. It certifies that a serving configuration produces correct output
and uses the interconnect as intended before any throughput number from that
configuration is trusted.

Run the full suite after any change to driver, kernel, vLLM, NCCL, FlashInfer,
network configuration, or after any node has been idle for several days. Run a
single tier when only that configuration is about to be benchmarked.

Every command block is labeled with the machine it runs on. Addresses use the
variables defined at the top of [SETUP.md](SETUP.md) (`$SPARK1_CX7_A` and so on).

---

# 1. One-time prerequisites

## 1.1 Non-interactive cache drop

The suite drops the page cache before every launch. Allow exactly that one
command without a password.

```bash
# spark1 and spark2, interactive session
which tee    # must print /usr/bin/tee
echo "$USER ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" | sudo tee /etc/sudoers.d/90-drop-caches
sudo chmod 440 /etc/sudoers.d/90-drop-caches
sudo visudo -cf /etc/sudoers.d/90-drop-caches
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches    # prints 3, no prompt
```

Every sudo call in the suite uses `sudo -n`, so a missing permission fails
immediately instead of hanging on a prompt.

## 1.2 Environments

- `spark1`, `spark2`: `~/venvs/vllm029` with vLLM 0.29.0 and Ray 2.58.0.
- `mac`: AIPerf 0.13.0 in `.venv-aiperf`. `spark2`: AIPerf in
  `~/venvs/aiperf-uv`, built from uv's managed CPython because system Python
  lacks `Python.h`.
- `spark1`, `spark2`: FlashInfer JIT cache warm at
  `~/.cache/flashinfer/0.6.18/121a/cached_ops/`, including `gemm_sm120` and
  `fp4_gemm_cutlass_sm120`. Cache readiness is per module and compiler flags;
  cold MoE JIT can OOM despite warm dense caches. See 6.7 for bounded offline
  prewarm and the process-group watchdog.

## 1.3 Scripts

Install the scripts in section 6 at `~/validation/bin/` on both Sparks.

---

# 2. Run protocol

Every tier follows the same five steps. The agent does not deviate from them.

1. **Prepare.** Create a run directory, drop caches, confirm no Ray or vLLM
   process is running.
2. **Launch.** Use `~/validation/bin/launch.sh` for every single-node and DP
   server (and the TP2 head server). It activates the venv, sets CUDA PATH,
   drops caches, bounds FlashInfer jobs, verifies ninja/setsid, and starts
   under nohup/setsid with a PID file and independent memory watchdog.
   For TP2, activate the venv and export section 2.2 in both Ray start shells.
3. **Wait.** Poll `/health` every 10 seconds with a 10-minute timeout. On
   timeout, capture the last 50 log lines and stop. Do not relaunch.
4. **Verify.** Scan the log, scrape the manifest, run the correctness checks,
   and collect counters where the tier requires them.
5. **Tear down.** Stop the server, stop Ray, and confirm no process remains.

Run directory layout:

```text
~/validation/runs/<YYYYMMDD-HHMM>-<tier>/
  server.log            server output (spark2.log for the second node where used)
  manifest.txt          /metrics cache_config_info scrape at launch
  smoke.json            single-request correctness pair
  echo-c1.json          reusable per-model+precision C1 reference
  echo.json             ownership echo check with non-gating C1 comparison at N=max(64, max-num-seqs)
  distinct.json         arithmetic accuracy, recorded but non-gating
  rdma-before.txt       mlx5 RDMA counters, both rails (TP2 only)
  rdma-after.txt
  summary.txt           PASS/FAIL per criterion, written last
```

## 2.1 Standard launch flags

```text
--generation-config vllm
--load-format fastsafetensors
--max-model-len 8192
--served-model-name qwen
```

CUDA graphs stay on. `--enforce-eager` is never used in the suite.

**G0:** unchanged smoke pair, then echo N=max(64, configuration max-num-seqs).
All modes use temperature 0, no stop field, and common ownership/validity/gate
rules. Prompt version 3 is retained and every echo report records echo_mode.

| Mode | Request | max_tokens |
|---|---|---:|
| completions | /v1/completions, v3 prompt `Repeat this number exactly, and nothing else: <number>\nAnswer:` (actual newline) |16 |
| chat-nothink | /v1/chat/completions; single user message `Repeat this number exactly, and nothing else: <number>`; chat_template_kwargs={"enable_thinking":false} |16 |
| completions-long | Same v3 completions prompt/endpoint |256 |

Pre-flight: `--preflight --smoke-report "$R/smoke.json"` automatically selects
per model. If smoke or initial completions pre-flight shows <think>, try
chat-nothink first; select it if 8/8 ownership. Otherwise try completions-long
and select it if 8/8. If both fail, STOP/report every string from both attempts.
Dense non-thinking models retain completions; a failed ordinary pre-flight
without <think> stops. preflight_attempts preserves each request set/result.
N-sized runs inherit mode from --preflight-report; C1 reference and C=N must
use that same mode. Explicit/reference mismatches reject before requests.
Legacy v3 reports without echo_mode describe completions only.

Ownership requires own-number substring, no other request's number substring,
and non-empty text containing digits.

First run 8 items at C1 (`--preflight`), requiring 8/8 ownership after the mode-selection
sequence above. If no mode passes, STOP/report all attempted strings as a
prompt problem. Every N-sized echo command
requires the saved `--preflight-report`. Collect one C1 reference per
model+precision, same seed/N. C1 ownership must be >=95% of N; otherwise STOP:
test cannot measure the server (prompt problem, not gate failure).
At C=N with `--reference`, require every C1-passed item to pass ownership.
Exclude C1-failed items and report their C1/C=N strings verbatim. No C=N
response anywhere (including exclusions) may contain another request's number.
Report eligible-pass/eligible-total, exclusions, parsed-reference differences,
and first-integer accuracy; the latter two remain non-gating. Transport errors
still fail. Preserve all responses, seeds, reference paths and max-num-seqs.

**Why no stop sequence:** an earlier version used stop=["\n"]. Generations
began with "\n\n", so every response ended before any content. That was a
test-design error, not a server fault. Version 3 removes the stop field.

vLLM output is not batch-invariant at temperature 0; exact cross-concurrency
text equality is not a valid correctness test. Replaying 43 responses from a
256-request run one at a time, 39 matched verbatim and 4 changed; all four
still contained their own number and no other request's. Batch-size changes
alter floating-point reduction order and can flip near-tie greedy tokens.
Arithmetic `distinct` is recorded but non-gating; its changed-answer
cross-product comparisons are diagnostic.

Two-node runs use `--gpu-memory-utilization 0.80`. Single-node runs use `0.85`.

## 2.2 Two-node environment

Export on both nodes before `ray start`, with each node's own `VLLM_HOST_IP`.

```bash
# spark1 (spark2 identical except VLLM_HOST_IP=$SPARK2_CX7_A)
source ~/venvs/vllm029/bin/activate
export VLLM_HOST_IP=$SPARK1_CX7_A
export NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
export NCCL_IB_MERGE_NICS=1
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export NCCL_DEBUG=INFO
export MAX_JOBS=2
export FLASHINFER_NVCC_THREADS=2
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

`GLOO_SOCKET_IFNAME` is mandatory. Without it, Gloo binds one rank to loopback
and TP2 fails with `Gloo connectFullMesh failed ... remote=[<loopback>]`.

## 2.3 Stop conditions

The agent stops and reports to a human when any of these occur:

- `nvidia-smi` shows `ERR!` on either node. Recovery requires a power cycle
  with physical access.
- A server fails to reach `/health` within the timeout.
- `OOM_STOP` or `WATCHDOG_ERROR` appears in watchdog.log; a watchdog kill
  is an OOM stop even without a kernel OOM. No automatic retry.
- A two-node launch hangs rather than exiting with an error.
- Smoke or the echo eligible-item gate fails, or an arithmetic-reference diagnostic shows a system
  fault or an ambiguous result requiring user review. Arithmetic accuracy
  alone is non-gating.
- Echo pre-flight below 8/8 after all prescribed mode attempts, or C1 ownership below 95%: stop as a prompt problem,
  not a server gate failure.
- A launch with a known risk of hanging the machine is requested without a person
  with physical access standing by.

**One-retry exception:** if a launch fails with an unambiguous environment or
operator error (missing PATH entry, typo in a flag, wrong path), name the error,
fix it, and retry once without asking. Preserve separate run directories and
record both attempts. Stop and ask if that retry fails. OOM, kernel/CUDA
errors, hangs, NCCL/Gloo errors, and uncertain causes always require stopping
and asking; the one-retry exception does not apply.

---

# 3. Tiers

## T0. Node health

```bash
# spark1 and spark2
nvidia-smi | grep -c 'ERR!'          # must be 0
source ~/venvs/vllm029/bin/activate
python ~/validation/bin/burn.py      # first burn: warmup, fp16 8192^3, 15 s
python ~/validation/bin/burn.py      # second burn: use this result
# dmesg is not readable without root; use the session start time:
journalctl -k --since "<session start, YYYY-MM-DD HH:MM:SS>" --no-pager | grep -iE "xid|nvrm"
```

| Criterion | Pass |
|---|---|
| `ERR!` fields | 0 on both nodes |
| Burn throughput | Second burn >=88 TFLOP/s per node; no upper bound |

Run the burn twice and use the second result. A high reading is not a fault;
there is no upper bound. Observed range on 2026-10-15: 93.0–95.3 TFLOP/s.
`dmesg` is not readable without root. Use `journalctl -k` for Xid/NVRM checks,
compare against existing entries and stop on a new Xid. ERR! and CUDA
initialization failures still stop.

## T1. Single-node serving, per precision

Run once per model directory: `Qwen3.8-27B`, `Qwen3.8-27B-FP8`,
`Qwen3.8-27B-NVFP4`. Runs on either node; run on both nodes in parallel to
halve wall time.

```bash
# spark1 or spark2
R=~/validation/runs/$(date +%Y%m%d-%H%M)-t1-$(hostname)-bf16; mkdir -p $R
source ~/venvs/vllm029/bin/activate
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches
~/validation/bin/launch.sh "$R" ~/models/Qwen3.8-27B --served-model-name qwen \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --generation-config vllm --load-format fastsafetensors \
  --max-num-seqs 64 --port 8000
~/validation/bin/wait_healthy.sh http://localhost:8000 600 $R/server.log
~/validation/bin/scan_log.sh $R/server.log
curl -s http://localhost:8000/metrics | grep cache_config_info > $R/manifest.txt
python ~/validation/bin/check.py smoke    --endpoints http://localhost:8000 --out $R/smoke.json
python ~/validation/bin/check.py echo --preflight --smoke-report "$R/smoke.json" --seed 0 --endpoints http://localhost:8000 --out "$R/preflight.json"
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 64 --seed 0 --concurrency 1 --endpoints http://localhost:8000 --out $R/echo-c1.json
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 64 --seed 0 --concurrency 64 --reference $R/echo-c1.json --endpoints http://localhost:8000 --out $R/echo.json
python ~/validation/bin/check.py distinct --endpoints http://localhost:8000 --out $R/distinct.json
~/validation/bin/teardown.sh
```

| Criterion | Pass |
|---|---|
| `/health` | Up within 10 minutes |
| Log scan | No fatal pattern (section 5) |
| Smoke pair | Paris; 391 |
| Echo concurrency gate | 64/64 (launch max-num-seqs 64) |
| NVFP4 kernel | Log names `FlashInferCutlassNvFp4LinearKernel` |

## T2. Two-node tensor parallel

```bash
# spark1, then spark2, each with its section 2.2 exports
ray start --head --node-ip-address $SPARK1_CX7_A --port 6379    # spark1
ray start --address $SPARK1_CX7_A:6379                          # spark2
```

```bash
# spark1
R=~/validation/runs/$(date +%Y%m%d-%H%M)-t2-tp2; mkdir -p $R
ray status | grep -q '2.0 GPU' || { echo "FAIL ray"; exit 1; }
~/validation/bin/launch.sh "$R" ~/models/Qwen3.8-27B --served-model-name qwen \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --max-model-len 8192 --gpu-memory-utilization 0.80 \
  --generation-config vllm --load-format fastsafetensors \
  --max-num-seqs 64 --port 8000
~/validation/bin/wait_healthy.sh http://localhost:8000 600 $R/server.log
~/validation/bin/scan_log.sh $R/server.log
grep -c 'NET/IB : Using \[0\]rocep1s0f1:1/RoCE \[1\]roceP2p1s0f1:1/RoCE' $R/server.log
curl -s http://localhost:8000/metrics | grep cache_config_info > $R/manifest.txt
python ~/validation/bin/check.py smoke    --endpoints http://localhost:8000 --out $R/smoke.json
python ~/validation/bin/check.py echo --preflight --smoke-report "$R/smoke.json" --seed 0 --endpoints http://localhost:8000 --out "$R/preflight.json"
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 64 --seed 0 --concurrency 1 --endpoints http://localhost:8000 --out $R/echo-c1.json
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 64 --seed 0 --concurrency 64 --reference $R/echo-c1.json --endpoints http://localhost:8000 --out $R/echo.json
python ~/validation/bin/check.py distinct --endpoints http://localhost:8000 --out $R/distinct.json
```

Bytes per token, sampled on both nodes around one fixed generation:

```bash
# spark1 and spark2
~/validation/bin/rdma_snap.sh > $R/rdma-before.txt
```

```bash
# spark1
python ~/validation/bin/check.py fixed --endpoints http://localhost:8000 --tokens 256 --out $R/fixed.json
```

```bash
# spark1 and spark2
~/validation/bin/rdma_snap.sh > $R/rdma-after.txt
paste $R/rdma-before.txt $R/rdma-after.txt | awk '{d=$6-$3; printf "%s %s delta=%d per_token=%.0f\n",$1,$2,d,d/256}'
```

```bash
# spark1, then spark2
~/validation/bin/teardown.sh
```

| Criterion | Pass |
|---|---|
| `ray status` | 2 nodes, 2.0 GPU |
| NCCL rails | Both RoCE devices listed on both ranks |
| Smoke pair | Paris; 391 |
| Echo concurrency gate | 64/64 (launch max-num-seqs 64) |
| RDMA symmetry | spark1 tx delta equals spark2 rx delta per rail |
| Both rails active | Nonzero delta on both rails |
| Bytes per token | Recorded; reference 3.68 MB/token one-way per node for Qwen BF16 |

Use mlx5 RDMA counters only. `/sys/class/net/*/statistics` counts kernel
TCP/IP traffic and does not see RoCE.

## T3. Two-node data parallel, internal load balancing

```bash
# spark1
R=~/validation/runs/$(date +%Y%m%d-%H%M)-t3-dp2int; mkdir -p $R
~/validation/bin/launch.sh "$R" ~/models/Qwen3.8-27B --served-model-name qwen \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-address $SPARK1_CX7_A --data-parallel-rpc-port 13345 \
  --max-model-len 8192 --gpu-memory-utilization 0.80 \
  --generation-config vllm --load-format fastsafetensors \
  --max-num-seqs 64 --port 8000
```

```bash
# spark2
R=~/validation/runs/$(date +%Y%m%d-%H%M)-t3-dp2int; mkdir -p $R
~/validation/bin/launch.sh "$R" ~/models/Qwen3.8-27B --served-model-name qwen --headless \
  --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-start-rank 1 \
  --data-parallel-address $SPARK1_CX7_A --data-parallel-rpc-port 13345 \
  --max-model-len 8192 --gpu-memory-utilization 0.80 \
  --generation-config vllm --load-format fastsafetensors \
  --max-num-seqs 64
```

```bash
# spark1
~/validation/bin/wait_healthy.sh http://localhost:8000 600 $R/server.log
python ~/validation/bin/check.py smoke    --endpoints http://localhost:8000 --out $R/smoke.json
python ~/validation/bin/check.py echo --preflight --smoke-report "$R/smoke.json" --seed 0 --endpoints http://localhost:8000 --out "$R/preflight.json"
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 64 --seed 0 --concurrency 1 --endpoints http://localhost:8000 --out $R/echo-c1.json
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 64 --seed 0 --concurrency 64 --reference $R/echo-c1.json --endpoints http://localhost:8000 --out $R/echo.json
python ~/validation/bin/check.py distinct --endpoints http://localhost:8000 --out $R/distinct.json
curl -s http://localhost:8000/metrics | grep 'vllm:request_prompt_tokens_count'
```

```bash
# spark1, then spark2
~/validation/bin/teardown.sh
```

| Criterion | Pass |
|---|---|
| Smoke pair | Paris; 391 |
| Echo concurrency gate | 64/64 (launch max-num-seqs 64) |
| Routing | `request_prompt_tokens_count` nonzero for both engines |

## T4. Two-node independent endpoints, client-side balancing

For dense, non-MoE models, vLLM rejects `--data-parallel-rank`. External
balancing means two independent servers with no `--data-parallel-*` flags.

```bash
# spark1 and spark2
R=~/validation/runs/$(date +%Y%m%d-%H%M)-t4-dp2ext; mkdir -p $R
source ~/venvs/vllm029/bin/activate
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches
~/validation/bin/launch.sh "$R" ~/models/Qwen3.8-27B --served-model-name qwen \
  --max-model-len 8192 --gpu-memory-utilization 0.80 \
  --generation-config vllm --load-format fastsafetensors \
  --max-num-seqs 64 --port 8000
~/validation/bin/wait_healthy.sh http://localhost:8000 600 $R/server.log
```

```bash
# spark1
python ~/validation/bin/check.py smoke    --endpoints http://localhost:8000 http://$SPARK2_CX7_A:8000 --out $R/smoke.json
python ~/validation/bin/check.py echo --preflight --smoke-report "$R/smoke.json" --seed 0 --endpoints http://localhost:8000 http://$SPARK2_CX7_A:8000 --out "$R/preflight.json"
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 64 --seed 0 --concurrency 1 --endpoints http://localhost:8000 http://$SPARK2_CX7_A:8000 --out $R/echo-c1.json
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 64 --seed 0 --concurrency 64 --reference $R/echo-c1.json --endpoints http://localhost:8000 http://$SPARK2_CX7_A:8000 --out $R/echo.json
python ~/validation/bin/check.py distinct --endpoints http://localhost:8000 http://$SPARK2_CX7_A:8000 --out $R/distinct.json
```

```bash
# spark1 and spark2
~/validation/bin/teardown.sh
```

| Criterion | Pass |
|---|---|
| Smoke pair | Paris; 391 on both endpoints |
| Echo concurrency gate | 64/64, 32 per endpoint (launch max-num-seqs 64) |

## T5. Client headroom

Run once per new harness or when the planned concurrency rises above the last
validated level. Same server and workload from two client hosts.

```bash
# mac, then spark2, against a running T2 server
aiperf profile \
  --url http://$SPARK1_MGMT:8000 \
  --endpoint-type completions --model qwen --tokenizer builtin \
  --use-server-token-count --use-legacy-max-tokens \
  --extra-inputs '{"ignore_eos":true,"temperature":0}' \
  --isl 128 --osl 128 --concurrency 64 \
  --request-count 320 --warmup-request-count 32 \
  --export-level summary
```

| Criterion | Pass |
|---|---|
| Errors | 0 on both clients |
| Output tok/s | Clients agree within 2% |

Reference: MacBook 288.67 and spark2 288.57 output tok/s, TP2 Qwen BF16, C=64.

---

# 4. Suite order and duration

Run T0, then T1 on both nodes in parallel, then T2, T3, T4. T5 runs only when
its trigger applies. Each launch costs roughly 3 minutes of engine startup; the
full suite without T5 takes roughly 30 minutes.

The suite passes only when every criterion in every tier run passes. A failed
tier invalidates throughput numbers for that configuration until it is rerun
and passes.

---

# 5. Log scan patterns

| Pattern | Meaning | Action |
|---|---|---|
| `Killed`, exit code 137, watchdog `OOM_STOP` | OOM / low-memory stop | Stop and ask; no automatic retry |
| `NV_ERR_NO_MEMORY` | Unified memory exhausted | Fail; drop caches, lower utilization |
| `Gloo connectFullMesh` | Gloo bound to loopback | Fail; set `GLOO_SOCKET_IFNAME` |
| `Traceback`, `RuntimeError`, `CUDA error` | Engine error | Fail; report |
| `No tuned config covers` | FP4 autotuner fallback on lm_head | Record count; not a failure |
| `cudagraph_capture_sizes` | Captured graph sizes | Record in summary |

---

# 6. Scripts

Every script lives in [`validation/bin/`](../validation/bin/); install them at
`~/validation/bin/` on both Sparks. Unit tests: `python -m unittest test_check
test_memory_watchdog test_analysis` from `validation/`.

| Script | Purpose |
|---|---|
| `check.py` | Smoke pair, echo ownership gate, arithmetic diagnostic, fixed-length generation |
| `launch.sh` | Cache drop, bounded JIT build jobs, server under `setsid`, memory watchdog |
| `memory_watchdog.py` | Stops the server process group when free memory stays below 6 GiB |
| `wait_healthy.sh` | Polls `/health` with a timeout; prints the last 50 log lines on failure |
| `scan_log.sh` | Fatal-pattern scan and autotuner fallback count |
| `rdma_snap.sh` | mlx5 RDMA byte counters on both rails |
| `teardown.sh` | Stops vLLM and Ray and confirms nothing remains |
| `burn.py` | 15 s fp16 matmul burn for node health |
| `sample_metrics.py` | Samples `/metrics` every 2 s into a CSV |
| `make_workload_a.py`, `aiperf_workload_a.sh`, `d3_clients.sh` | Server-verified 1024/256 inputs and the AIPerf / `vllm bench serve` cross-check |
| `analyze_window.py`, `compare_d3.py` | Steady-state window and harness comparison |
| `sweep.sh`, `sweep_csv.py` | Concurrency sweep against one or two endpoints, with endpoint health checks; merges results into one CSV and flags invalid points |
| `allreduce_lat.py` | Cross-node all-reduce latency at decode-sized messages, eager and replayed in a CUDA graph (run with `torchrun`, no server up) |
| `paired_gsm8k.py` | Item-by-item comparison of two lm-eval GSM8K runs: paired interval and exact McNemar test |

## 6.1 `check.py`

Smoke pair, version 3 pre-flight/validity/eligible echo gate, arithmetic accuracy,
and fixed generation. Echo --preflight runs8 items at C1; requires 8/8.
N-sized echo requires --preflight-report. C1 ownership below 95% returns 2
(prompt problem); no concurrent requests are sent with an invalid reference.
Concurrent echo requires --reference; every C1-passed item must retain
ownership, and other-request number substrings fail anywhere. C1-failed
items are excluded and reported verbatim. C1 reference mode/seed/N/model and
prompt version are checked. Parsed-reference differences and first-integer
accuracy remain non-gating. Requests round-robin across endpoints. See 2.1.

```bash
# Same model+precision, same seed and N; reuse C1 across its concurrencies.
python ~/validation/bin/check.py echo --preflight --smoke-report "$R/smoke.json" --seed 0 --endpoints http://localhost:8000 --out "$R/preflight.json"
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 256 --seed 0 --concurrency 1 \
  --endpoints http://localhost:8000 --out "$R/echo-c1.json"
python ~/validation/bin/check.py echo --preflight-report "$R/preflight.json" --n 256 --seed 0 --concurrency 256 \
  --reference "$R/echo-c1.json" --endpoints http://localhost:8000 --out "$R/echo.json"
```

## 6.7 `launch.sh` and `memory_watchdog.py`

Create the run directory and verify no previous server/compiler processes
remain before calling the wrapper. TP2 requires Ray already running with the
correct environment. Each wrapper writes server.log/server.pid and
watchdog.log/watchdog.pid/watchdog-stderr.log locally; collect spark2.log
separately for distributed workers.

Installed FlashInfer 0.6.18 source was checked on both nodes, not inferred
from PyTorch conventions. In
`~/venvs/vllm029/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py`,
`MAX_JOBS` is read at lines 347–349 and passed as Ninja `-j` at 363–365;
`FLASHINFER_NVCC_THREADS` is read at 96–101 and becomes nvcc `--threads` at
112.
The wrapper exports **MAX_JOBS=2 and FLASHINFER_NVCC_THREADS=2**.

The server runs under setsid, owning its process group. A separate watchdog
validates that group and PID start time, then samples Linux MemAvailable every
1 s. **Below 6 GiB for 3 consecutive samples** logs OOM_STOP, sends TERM to
that group, waits 10 s, and sends KILL to the group, even if its leader exited
in the meantime. Ninja creates separate compiler job groups: the watchdog
captures descendant PID/start-time/group identities and signals those verified
groups too, preventing orphan nvcc/cicc work. It never targets an external
inherited group. This is an OOM stop condition; do not retry automatically.
The watchdog exits when the server exits and refuses to follow a reused PID.
Memory-read failure stops the group with WATCHDOG_ERROR. The wrapper checks
that monitoring initialized; failure stops the launched group.

Canonical watchdog source: `validation/bin/memory_watchdog.py`; install it
beside launch.sh on both nodes. Unit tests:
`python -m unittest validation.test_memory_watchdog`. Tests cover consecutive
samples/reset, exact threshold, server exit/PID reuse, TERM/KILL ordering,
independent-group validation, and immediate offline-build abort. Process-group
behavior, including a descendant in a new session, was checked with harmless
Linux subprocesses: all nine tests passed on both nodes before serving.

A warm JIT cache does not establish that post-load autotuning fits memory: a
launch can complete graph capture and still hit the watchdog during MoE kernel
autotuning. That is an OOM stop like any other.

Prewarm each required JIT module **offline, GPU idle, no server**, using the
same compiler flags as the subsequent launch. Dense GEMM warmth does not imply
MoE readiness. The offline build samples MemAvailable each second and aborts
on the **first** sample below 6 GiB (not the server's three-sample rule).
Offline prewarm runners also start a detached immediate-abort guard so memory
protection survives an SSH/client disconnect. Inspect active process identities
and logs before resuming; never overlap a second Ninja job on a live build.
Record elapsed time, minimum available and peak system used+unavailable memory
(MemTotal−MemAvailable). Synchronize completed module/generated dependencies
and verify `ninja -n` reports no work on both nodes before model launch.

---

## 6.8 `sample_metrics.py`

Source: `validation/bin/sample_metrics.py`; install
at `~/validation/bin/sample_metrics.py`. Run with the vLLM venv Python:

```bash
# spark1, alongside a running server
nohup ~/venvs/vllm029/bin/python ~/validation/bin/sample_metrics.py \
  --url http://localhost:8000 --interval 2 --out "$R/metrics.csv" \
  > "$R/sampler.log" 2>&1 < /dev/null &
echo $! > "$R/sampler.pid"
# after the workload drains
kill -TERM "$(cat "$R/sampler.pid")"
```

CSV columns include Unix time, elapsed seconds, running, waiting, capacity,
deferred, KV usage (fraction), preemptions, cumulative generation tokens,
instantaneous tokens/s, and scrape error. The initial rate is blank; subsequent
rates use actual monotonic elapsed time. Missing metrics remain blank. SIGTERM
flushes and closes the CSV.

---

---

# 7. Agent handoff

Paste to the agent to run the suite:

```text
Run the Cluster Validation Suite in VALIDATION.md. Follow section 2 exactly for
every tier: prepare, launch under nohup, wait with the 10-minute health
timeout, verify, tear down. Use `sudo -n` for every sudo call. Run tiers in the
section 4 order. Write summary.txt in each run directory with PASS or FAIL per
criterion from that tier's table. Stop and report on any section 2.3 stop
condition; only the named environment/operator error exception permits one
retry. When done, report one line per tier: tier, PASS or
FAIL, run directory, and any recorded values (KV tokens, bytes per token,
autotuner fallback count, capture sizes).
```
