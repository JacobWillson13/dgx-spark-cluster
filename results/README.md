# Results

Every number in the report and every figure comes from a CSV here.
Regenerate the figures with `python3 scripts/make_figures.py`.

| Folder | Stage | Report section |
|---|---|---|
| [`pilot/`](pilot/) | Eager-mode pilot: interconnect, TP2 vs PP2, one node vs two, weight-streaming model | 3, 5, 6 |
| [`phase0/`](phase0/) | Native-stack calibration: loading, capacity, precision, RDMA traffic | 7 |
| [`closing/`](closing/) | Compiled rerun, DeepSeek attempts, GSM8K by precision, all-reduce latency | 8 |
| [`sweep/`](sweep/) | Concurrency sweep (one node, TP2, DP2), echo checks, GovReport | 9 |
| [`predictions.csv`](predictions.csv) | All 20 predictions, each recorded before its run, with outcome | 5, 8, 9 |

| File | One row per |
|---|---|
| `pilot/interconnect.csv` | Link configuration (paths, MTU) |
| `pilot/runs.csv` | Pilot benchmark run |
| `pilot/weight_streaming.csv` | Configuration: predicted memory time against measured TPOT |
| `pilot/predictions.csv` | Pilot prediction |
| `phase0/measurements.csv` | Calibration measurement |
| `phase0/precision_kv.csv` | Weight precision and KV-cache dtype |
| `phase0/rdma_bytes_per_token.csv` | Node and RDMA rail |
| `closing/pilot_rerun.csv` | Rerun benchmark (model, topology, concurrency) |
| `closing/deepseek.csv` | DeepSeek V4 Flash attempt |
| `closing/quality.csv` | GSM8K evaluation |
| `closing/quality_paired.csv` | Precision compared with BF16, per filter |
| `closing/allreduce.csv` | All-reduce measurement run |
| `sweep/sweep.csv` | Setup and concurrency |
| `sweep/echo.csv` | Echo correctness check |
| `sweep/govreport.csv` | GovReport run |

Conventions: times in milliseconds unless the column says `_s`; throughput is
output tokens per second; TTFT is time to first token, TPOT time per output
token, ITL inter-token latency. Raw results were never edited by hand.
