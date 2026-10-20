# Multi-Node LLM Inference on Two DGX Sparks

Jacob Willson · October 2026

## 1. Summary

Two NVIDIA DGX Spark systems were joined over a direct 200 GbE ConnectX-7 cable and run as one vLLM inference cluster. The work had three stages: a pilot that compared parallelism strategies with predictions recorded in advance, a calibration phase that moved to a native software stack and built a correctness-gated measurement method, and a closing rerun of the pilot on that stack.

Main results:

- **Link bandwidth:** 196.03 Gb/s, 98% of line rate, with both PCIe paths driven at once. One path reaches about 112 Gb/s at any frame size.
- **Tensor versus pipeline parallelism:** TP2 was 2.01x faster than PP2 on single-stream decode and 1.65x faster at concurrency 32 (Llama 3.3 70B). The pilot predicted PP2 would win.
- **One node versus two:** splitting Qwen3.8 27B across both nodes gave 1.57x single-stream throughput and 1.77x lower per-token latency, although the model fits on one node. The pilot predicted no single-stream gain.
- **Decode efficiency:** single-stream decode reached 87.7% of the 273 GB/s memory-bandwidth ceiling. In eager mode, every configuration also paid about 30 ms per forward pass beyond weight streaming.
- **Compiled stack:** CUDA graphs removed most of the 30 ms constant on Qwen3.8 27B (29 to 9 ms on one node, about 30 to 15 ms with TP2) and none of it on Llama 3.3 70B TP2. Two of five closing predictions held.
- **Correctness gate:** on the native stack, each configuration had to return correct answers, with no cross-request contamination, under concurrent load before its throughput was counted.

![Line rate needs both PCIe paths](figures/interconnect.png)

## 2. System

Each DGX Spark is a GB10 Grace Blackwell system-on-chip: a 20-core Arm CPU and a Blackwell GPU (compute capability sm_121) sharing 128 GB of LPDDR5x at 273 GB/s. The operating system reports about 121 GiB to an inference server, and that figure, not the nominal 128 GB, decides whether a model fits on one node.

| Component | Specification |
|---|---|
| Nodes | 2 x NVIDIA DGX Spark (`spark1`, `spark2`) |
| SoC | GB10 Grace Blackwell, 20-core Arm CPU + Blackwell GPU, sm_121 |
| Memory | 128 GB LPDDR5x unified per node, 273 GB/s |
| Interconnect | ConnectX-7, one 200 GbE QSFP direct-attach cable |
| Management | 10 GbE RJ-45 per node into a switch; a laptop on the same private subnet |

Two networks with separate jobs:

- **Management network:** SSH, administration and file movement, on a small private subnet with static addresses and no DHCP.
- **ConnectX-7 cable:** every byte the inference engine moves between nodes.

Keeping them apart means an administrative session cannot disturb a benchmark, and a benchmark cannot lock an operator out.

![Cluster topology: a laptop and switch on the management network, two DGX Sparks joined by one ConnectX-7 cable](figures/topology.png)

## 3. Interconnect

The ConnectX-7 adapter presents each physical port as two logical interfaces, one per PCIe path into the SoC (`enp1s0f1np1` and `enP2p1s0f1np1`). With one cable installed, both are views of the same wire. A benchmark that drives one interface measures one PCIe path, and one path cannot saturate a 200 GbE port.

| Configuration | RoCE path MTU | Per path (Gb/s) | Aggregate (Gb/s) | Line rate |
|---|---:|---:|---:|---:|
| One path | 1024 | 111.72 | 111.72 | 55.9% |
| Both paths | 1024 | 92.57 | 185.14 | 92.6% |
| One path | 4096 | 111.73 | 111.73 | 55.9% |
| **Both paths** | **4096** | **98.01 / 98.02** | **196.03** | **98.0%** |

Two hardware properties show up in that table:

- **One path caps near 112 Gb/s.** Jumbo frames change single-path throughput by 0.01 Gb/s, which is noise.
- **Jumbo frames lift the aggregate by 10.9 Gb/s.** Larger frames only matter once two paths contend for the same wire.

RoCE derives its path MTU from the interface MTU: 1500 gives 1024, and 9000 gives 4096.

NVIDIA's Cluster Assistant reported 185.2 Gb/s in its own link test. That matches the dual-path measurement at path MTU 1024 to within noise; the test ran before the MTU change.

![Cluster Assistant link test: 185.2 Gbps, 1.7 µs](figures/screenshots/cluster-assistant-link-test.png)

Latency carries one warning. `ping` reports 1.1 to 2.0 ms across this link because ICMP goes through the kernel network stack. RDMA bypasses it and measures 1.7 µs. Anyone validating this link with `ping` will conclude it is broken.

## 4. Method

### 4.1 Parallelism

Each Spark has one GPU, so every multi-node configuration is a two-way split across the wire. The two strategies differ only in how they divide the model:

- **Tensor parallelism (TP2):** splits every layer, so both nodes work on each layer at once and exchange partial results.
- **Pipeline parallelism (PP2):** splits by depth, so a token passes through node one, then node two.

Ray coordinates multi-node serving; vLLM's multiprocess backend cannot span hosts.

![Two ways to split a model across two nodes](figures/parallelism.png)

### 4.2 What the benchmark sends

Every pilot run used vLLM's built-in random dataset. It downloads nothing: it builds prompts by walking consecutive token IDs from a random start and decoding them to text, then corrects the length until the server sees exactly what was asked for. The prompts are real vocabulary in a meaningless order. Hardware cost per token is the same whether text is coherent or not.

Every request in every run had exactly **1024 input tokens and 256 output tokens**, with a fixed seed. That makes comparisons across configurations clean. It also means the absolute numbers describe fixed-length synthetic traffic, not a production mix.

### 4.3 Pilot configuration

| Setting | Value | Why |
|---|---|---|
| Memory fraction | 0.85 | The 0.92 default fails the startup check on unified memory |
| Max model length | 8192 | Fixed context budget |
| Prompt shape | 1024 in / 256 out | No length variation |
| Concurrency | 1 and 32 | 5 prompts at 1, 64 at 32 |
| Execution | Eager | One compiled configuration took over 37 minutes to compile |
| Load generator | `vllm bench serve` | No custom harness to defend |

Pilot stack: vLLM 0.19.2rc1 (CUDA 13 nightly container), Ray 2.58.0, PyTorch 2.11.0, NCCL 2.28.9, September 2026.

Eager mode penalizes every run equally, so comparisons between configurations hold. Absolute numbers are floors. Section 8 tests that claim.

## 5. Predictions and outcomes

Four predictions were recorded before measurement. Two were falsified.

| Prediction | Outcome |
|---|---|
| Single-stream decode lands at 80 to 90% of the memory-bandwidth ceiling | **Confirmed:** 87.7% (TP2) and 87.3% (PP2) for Llama 3.3 70B |
| DeepSeek V4 Flash decodes much faster than Llama 3.3 70B (about 11x fewer active parameters) | **Not tested:** the model was unsupported on this stack |
| Pipeline parallelism beats tensor parallelism across the link | **Falsified:** TP2 won every metric |
| A second node helps throughput under load but not single-stream latency | **Falsified:** single-stream improved 1.57x |

**Why pipeline parallelism was expected to win.** Memory runs at 273 GB/s and the link at 24.5 GB/s, about 11x slower. Tensor parallelism exchanges data at every layer; pipeline parallelism only at the stage boundary. On that framing the link looks like the bottleneck.

**Why that was wrong.** The deciding quantity is transfer *volume*, not the bandwidth ratio.

- **Tensor parallelism** moves a few megabytes per token, a fraction of a millisecond over this link against a 259 ms memory budget. Section 7 later measured it directly on Qwen3.8 27B: 3.68 MB per output token.
- **Each node also reads half the weights, and both nodes read at the same time.** That cuts memory time in half.
- **Pipeline parallelism also halves the weights per node, but reads them one stage after the other.** A single stream gains nothing.

The network cost could never overcome a 2x advantage in memory time.

**The revised prediction also failed.** After the first runs, the prediction narrowed: TP wins decode, but PP should still win time to first token under load, because prefill exchanges are large. Measured, PP's median time to first token was 11% *worse*. The cause is a scheduler default: chunked prefill splits long prefills into small pieces and interleaves them with decode, so the large exchanges never happen.

## 6. Pilot results

| Run | Model | Config | Concurrency | Output tok/s | Median TTFT | Median TPOT | P99 ITL |
|---|---|---|---:|---:|---:|---:|---:|
| L-TP-1 | Llama 3.3 70B | TP2 | 1 | 3.39 | 1,337 ms | 289.2 ms | 358 ms |
| L-TP-32 | Llama 3.3 70B | TP2 | 32 | 65.31 | 9,709 ms | 457.0 ms | 2,431 ms |
| L-PP-1 | Llama 3.3 70B | PP2 | 1 | 1.69 | 2,212 ms | 581.9 ms | 645 ms |
| L-PP-32 | Llama 3.3 70B | PP2 | 32 | 39.48 | 10,796 ms | 783.9 ms | 4,064 ms |
| Q-1N-1 | Qwen3.8 27B | 1 node | 1 | 3.64 | 1,101 ms | 230.1 ms | 245 ms |
| Q-1N-32 | Qwen3.8 27B | 1 node | 32 | 72.69 | 5,917 ms | 400.9 ms | 1,975 ms |
| Q-2N-1 | Qwen3.8 27B | TP2 | 1 | 5.71 | 817 ms | 130.4 ms | 148 ms |
| Q-2N-32 | Qwen3.8 27B | TP2 | 32 | 120.34 | 3,856 ms | 245.2 ms | 1,286 ms |

TTFT is time to first token, TPOT time per output token, ITL inter-token latency. Medians are reported because two concurrency-1 runs had a single 21 to 24 s first-request warmup outlier. Data: [`results/pilot/runs.csv`](results/pilot/runs.csv).

### 6.1 Tensor versus pipeline parallelism

![Tensor parallelism beats pipeline parallelism](figures/tp_vs_pp.png)

Pipeline parallelism's 1.69 tok/s is what one node would achieve if it could hold the whole model: both stages read all the weights in sequence for every token. Splitting by depth buys capacity on this hardware and nothing else. PP scales slightly better with concurrency (23.4x against 19.3x), but nowhere near enough to recover a 2x single-stream deficit.

### 6.2 One node versus two

![A second node helps even when the model fits on one](figures/one_vs_two.png)

Qwen3.8 27B fits on one node. Splitting it anyway improved single-stream throughput 1.57x, per-token latency 1.77x and time to first token 1.35x. At concurrency 1 there is no batching to exploit, so the gain is memory bandwidth: each node reads half the weights, at the same time.

On cost: the second node buys about 1.6x for 2x the hardware. That is a latency and capacity win, not an efficiency win. The efficiency case is Llama 3.3 70B: 132 GB does not fit in one node's usable memory, so the second node is the difference between 3.39 tok/s and no service.

### 6.3 The bandwidth ceiling and the 30 ms constant

Decode streams the active weights out of memory for every token, so the ceiling is 273 GB/s divided by the bytes each node reads per token. Llama 3.3 70B reached 87.7% of that ceiling under TP2 and 87.3% under PP2: two strategies, two ceilings, near-identical efficiency.

![Eager mode adds about 30 ms per forward pass](figures/weight_streaming.png)

Subtracting predicted weight-streaming time from measured time per token leaves about 30 ms per sequential forward pass, nearly constant across model, architecture and node count. It covers everything that is not weight streaming: eager-mode kernel launches, collectives and scheduling. Pipeline parallelism pays it twice, since its two stages run in sequence.

That constant explains why smaller models fall further below the ceiling: 30 ms is 29% of Qwen's two-node token time and 12% of Llama's. It also predicts where compiled execution with CUDA graphs should pay off most: the small-model, low-latency corner. Section 8 tests this.

## 7. Moving to a native stack

The calibration phase rebuilt the software stack and the measurement method before any further throughput claims. Stack: DGX OS 7.6.0, kernel 7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0, vLLM 0.29.0, PyTorch 2.13.0, FlashInfer 0.6.18, NCCL 2.29.7, Ray 2.58.0.

| Finding | Measured |
|---|---|
| **vLLM runs natively on sm_121.** No container, no patching. Do not install `flashinfer-cubin`: its version check hard-fails | `uv pip install vllm==0.29.0 --torch-backend=auto` on both nodes |
| **Weight loading is 44x faster with `fastsafetensors`** | 6.90 s versus 302.9 to 389.5 s for Qwen3.8 27B BF16 |
| **CUDA graphs are safe here and cheap to compile** | 15 to 17 s compile, correct output at concurrency 64; the pilot avoided them after a 37-minute compile |
| **Lower precision buys capacity** | NVFP4 weights hold 1.64x the KV-cache tokens of BF16; an FP8 KV cache adds 1.35x with BF16 weights |
| **One node saturates early** | 120.7 tok/s at 64 concurrent requests; about 142 tok/s at the 191-sequence ceiling. Three times the concurrency bought 18% |
| **Tensor-parallel traffic is small** | 3.68 MB per output token, one way per node, split across both RDMA rails |
| **Two harnesses agree** | `vllm bench serve` 118.98 and AIPerf 118.91 output tok/s on identical inputs, a 0.06% gap |

Data: [`results/phase0/`](results/phase0/).

### 7.1 Why a correctness gate

A published two-Spark DeepSeek V4 Flash recipe reported 300 tok/s at 16 streams, then found the model was answering every prompt with `1. 2. 3. 4. ...` once CUDA graphs and more than two sequences were combined. With graphs off, the coherent figure was 269 tok/s ([source](https://github.com/Mirrdhyn/dsv4-flash-dgx-spark)). Nothing in the logs flagged it.

So every configuration here passes a gate before its throughput counts:

1. **Smoke:** "The capital of France is" must contain Paris; 17 x 23 must return 391.
2. **Echo ownership:** 64 or more concurrent requests each ask the model to repeat a unique random six-digit number. Every response must contain its own number and no other request's number, at concurrency 1 and again at full concurrency.

The gate also showed why simpler checks fail. At temperature 0, vLLM output is **not** batch-invariant. When 43 responses from a 256-request run were replayed one at a time, 4 came back worded differently: a different batch size changes floating-point reduction order and can flip near-tie tokens. None of the four contained another request's number. Exact-text comparison across concurrency levels is therefore not a valid correctness test; ownership is.

### 7.2 Platform traps

| Trap | What happens | Fix |
|---|---|---|
| Gloo binds to loopback | Two-node TP fails with `Gloo connectFullMesh failed`, even though NCCL is configured | Set `GLOO_SOCKET_IFNAME` as well as `NCCL_SOCKET_IFNAME` on both nodes, before `ray start` |
| Ray workers inherit the wrong environment | Workers take the environment of the shell that started Ray, not a later one; a missing `nvcc` silently changes code paths | Export everything before `ray start` |
| Just-in-time kernel builds run out of memory | FlashInfer compiles kernels at first use with the model already resident; parallel `nvcc` jobs exhaust unified memory and the engine is killed | Bound with `MAX_JOBS=2` and `FLASHINFER_NVCC_THREADS=2`; build kernels offline before the first launch that needs them |
| Page cache competes with the GPU | Unified memory means file cache shrinks what the model can use | Drop caches before every launch; swap off; memory fraction 0.85 on one node, 0.80 for two-node TP |
| GPU firmware timeout after idle | After about two weeks idle, one node showed `ERR!` in `nvidia-smi` and an Xid 119 GSP timeout | Check `nvidia-smi` at the start of every session; a reboot cleared it, and a 15 s matmul burn confirmed recovery |
| `nvidia-smi` memory fields | Read `N/A` on GB10 | Use vLLM's `cache_config_info` metric and OS memory |
| NVFP4 output-layer fallback | The FP4 autotuner has no tuned config for the output projection at some token counts and falls back silently | Grep logs for `No tuned config covers` on every NVFP4 launch |

## 8. Revisiting the pilot on the compiled stack

Section 6.3 claimed the eager-mode numbers were floors and that CUDA graphs would remove most of the 30 ms constant. The closing session reran the pilot's Qwen and Llama TP2 cells on the native compiled stack to test that, with predictions committed before the first run.

Same workload as the pilot: random dataset, 1024 input / 256 output tokens, seed 0, concurrency 1 and 32. Differences, all intentional: vLLM 0.29.0 native, CUDA graphs on (full and piecewise capture, batch sizes 1 to 128, confirmed in both two-node server logs), `--load-format fastsafetensors`, `--generation-config vllm`, two warmup requests, and 10 / 160 prompts instead of 5 / 64. Every configuration passed the section 7.1 gate before its numbers counted.

| Config | Weight streaming | Pilot median TPOT (eager) | Predicted (compiled) | Measured | Outcome |
|---|---:|---:|---:|---:|---|
| Qwen3.8 27B, one node | 201 ms | 230.1 ms | 203 to 212 ms | 210.1 ms | **Confirmed** |
| Qwen3.8 27B, TP2 | 100.5 ms | 130.4 ms | 103 to 115 ms | 115.3 ms | **Missed** by 0.3 ms |
| Llama 3.3 70B, TP2 | 259 ms | 289.2 ms | 261 to 272 ms | 290.0 ms | **Falsified** |

![CUDA graphs helped Qwen, not Llama](figures/eager_vs_compiled.png)

| Prediction | Pilot | Predicted | Measured | Outcome |
|---|---|---|---|---|
| Residual per-token overhead at concurrency 1 | ~30 ms | 12 ms or less in all three | 9.1 / 14.8 / 31.0 ms | **Falsified:** met on one node only |
| Qwen two-node per-token speedup | 1.77x | 1.80x to 1.95x | 1.82x | **Confirmed** |
| Output tok/s at concurrency 32 | 72.69 / 120.34 / 65.31 | each 5% to 35% higher | 86.70 (+19.3%) / 150.24 (+24.8%) / 64.07 (-1.9%) | **Falsified** on Llama |
| DeepSeek V4 Flash 0731 NVFP4, stock stack, TP2 | not run | more likely fails to load or fails the gate than serves | failed before loading weights | **Confirmed** |
| GSM8K, 200 items, relative to BF16 | not run | FP8 within 2 points, NVFP4 within 3 | FP8 +3.0, NVFP4 -5.5 | **Falsified** as stated; see 8.3 |

Two of five held. Data: [`results/closing/`](results/closing/).

### 8.1 The constant is gone on Qwen and untouched on Llama

CUDA graphs cut the residual from 29 ms to 9 ms on one node and from about 30 ms to 15 ms for Qwen under TP2. On Llama TP2 it did not move: 30.2 ms eager, 31.0 ms compiled. Throughput at concurrency 32 tells the same story: Qwen gained 19% and 25%, Llama lost 2%.

Graph capture was confirmed in the Llama log, including batch size 1, so the remaining constant is not kernel-launch overhead. It is also not link bandwidth: Llama moved 6.50 MB per output token over RDMA, under 0.3 ms at 24.5 GB/s. What it is was not isolated. The next measurement is a profile of one Llama decode step, or an all-reduce latency test at Llama's message size.

The practical reading: for a model that fits on one node, compiled execution brings decode within 5% of the memory-bandwidth ceiling. For the 70B model that needs two nodes, about 31 ms per token is still unaccounted for.

### 8.2 DeepSeek V4 Flash on the stock stack

The launch failed 38 seconds in, before loading weights:

```text
AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto
```

This is further than the pilot got. vLLM 0.29.0 recognizes the architecture; the model requires an FP8 KV cache that the standard two-node recipe does not set. The plan allowed one stock attempt with no changes, so `--kv-cache-dtype fp8` was not tried. Whether DeepSeek V4 Flash serves on this stack, and passes the correctness gate, is still open.

### 8.3 Precision and answer quality

Qwen3.8 27B on one node at each precision, full gate first, then GSM8K: 200 items, 5-shot, greedy, through the completions endpoint.

| Precision | Flexible extract | Strict match | Flexible vs BF16 |
|---|---:|---:|---:|
| BF16 | 67.5% | 65.0% | |
| FP8 | 70.5% | 67.5% | +3.0 |
| NVFP4 | 62.0% | 57.0% | -5.5 |

At 200 items, each score carries a 95% interval of about plus or minus 6.5 points. Bands of 2 and 3 points were narrower than this sample can resolve, which is a flaw in the prediction, not a finding. FP8 scoring above BF16 is within noise. NVFP4 is lowest on both metrics (8 points lower on strict match), which is suggestive but not conclusive at this size. Per-item outputs were logged, so a paired comparison on the same 200 items is possible without more cluster time.

The absolute scores reflect this harness (raw completions prompts, no chat template, thinking off) and should not be compared with published GSM8K results. Compare across precisions only.

## 9. Known limits

- **Nemotron 3 Super 120B (NVFP4) did not serve.** With every kernel prebuilt offline, the launch completed CUDA graph capture, then free memory fell below 6 GiB during MoE kernel autotuning (peak 115.9 GiB in use). The memory watchdog stopped it, as designed. The allocation behind the growth was not isolated.
- **Llama 3.3 70B TP2 keeps a 31 ms constant with CUDA graphs on.** Graph capture was confirmed; the cause was not profiled (section 8.1).
- **DeepSeek V4 Flash did not serve.** One stock attempt failed on a KV-cache dtype requirement; a launch with an FP8 KV cache was not tried (section 8.2).
- **Quality checks are small.** GSM8K at 200 items resolves differences of about 6.5 points, and absolute scores are specific to this harness (section 8.3).
- **Gemma 4 was not served.** The test plan required someone physically present in case a launch hung the machine, and that session did not happen.
- **No full concurrency sweep.** The single-node ceiling was located (about 191 sequences), but the accepted throughput curve at each configuration's ceiling was not run.
- **Power is board-reported.** GPU-reported SoC power, not wall power.
- **One hardware pair, synthetic traffic.** Results describe two DGX Sparks on a direct link with fixed-length prompts, not larger servers or production request mixes.
- **No fine-tuning.** Every model runs as published.

## 10. How it was run

The cluster was operated from a laptop. Planning, review and documentation were done with an AI assistant (Claude). Execution was handed to a coding agent (OpenAI Codex) that ran commands on both nodes over SSH, following written procedures ([`docs/VALIDATION.md`](docs/VALIDATION.md)).

Guardrails made it safe to leave the agent running:

- **Scoped privilege.** The agent's account could run exactly one command as root without a password: dropping the page cache. Every `sudo` call used `sudo -n`, so anything else failed immediately instead of waiting on a prompt.
- **One launch wrapper.** Every server started through [`launch.sh`](validation/bin/launch.sh) in its own process group, with an independent memory watchdog that kills the whole group if free memory stays below 6 GiB for three seconds.
- **Hard stop rules.** No retries after an out-of-memory stop, a hang, a GPU error or a failed correctness gate. No unattended reboots. The agent stops and reports.
- **Correctness before throughput.** No number counted until its configuration passed the gate in section 7.1.

The Nemotron stop in section 9 is the guardrails working: the watchdog ended the launch before the node ran out of memory, and the agent stopped instead of retrying.

## Appendix: reproducing

- Build the cluster: [`docs/SETUP.md`](docs/SETUP.md)
- Certify a configuration before benchmarking it: [`docs/VALIDATION.md`](docs/VALIDATION.md)
- Scripts and unit tests: [`validation/`](validation/)
- Figures: `python3 scripts/make_figures.py` regenerates every chart from [`results/`](results/)
