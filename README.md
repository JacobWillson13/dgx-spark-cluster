# Two DGX Sparks, One Cable

What a second NVIDIA DGX Spark actually buys for LLM inference, measured. Two GB10 systems joined by a direct 200 GbE ConnectX-7 link, serving models across both nodes with vLLM, with predictions recorded before each experiment and scored after.

**Read the write-up: [REPORT.md](REPORT.md)**

![Two networks, two jobs](figures/topology.png)

## Findings

- **The link reaches 196 Gb/s, but only with both PCIe paths driven at once.** One path caps near 112 Gb/s at any frame size.
- **Tensor parallelism beats pipeline parallelism across the wire,** 2.01x single-stream and 1.65x at concurrency 32. The prediction said the opposite; transfer volume, not bandwidth ratio, decides it.
- **A second node helps even when the model fits on one:** 1.57x single-stream throughput and 1.77x faster per-token latency for a 27B model.
- **Decode is memory-bandwidth bound plus a constant.** 87.7% of the 273 GB/s ceiling, and about 30 ms per forward pass of eager-mode overhead regardless of model or node count.
- **Throughput claims need a correctness gate.** Every configuration must return correct, uncontaminated answers under load before its numbers count.

![Tensor parallelism beats pipeline parallelism across the link](figures/tp_vs_pp.png)

## Repository

| Path | Contents |
|---|---|
| [`REPORT.md`](REPORT.md) | Full write-up: method, predictions, results, platform findings, known limits |
| [`docs/SETUP.md`](docs/SETUP.md) | Build the cluster: networks, ConnectX-7 link, native vLLM, single- and two-node serving |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | Certify a configuration before benchmarking it: run protocol, correctness gate, stop rules |
| [`validation/`](validation/) | The validation scripts and their unit tests |
| [`results/`](results/) | Measured data as CSV |
| [`figures/`](figures/) | Charts, regenerated from `results/` by [`scripts/make_figures.py`](scripts/make_figures.py) |

## Hardware and software

2 x NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory, sm_121), one 200 GbE QSFP direct-attach cable, a laptop for control. Native vLLM 0.29.0, PyTorch 2.13, FlashInfer 0.6.18, NCCL 2.29.7, Ray 2.58.0 on DGX OS 7.6.0. Pilot runs used an earlier containerized vLLM; versions are listed in the report.

## License

Code is under the [MIT License](LICENSE). The report, documentation and figures are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Jacob Willson
