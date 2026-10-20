# Closing runs

Final sessions on the compiled native stack. Predictions were committed before
each set of runs; results and outcomes are in REPORT.md section 8.

- `pilot_rerun.csv`: the pilot's Qwen and Llama cells rerun with CUDA graphs on.
- `deepseek.csv`: stock DeepSeek V4 Flash attempts and their stop lines. The
  `C-DEEPSEEK-2` row is a setup stop (stale Ray status) before any launch.
- `quality.csv`: GSM8K by precision, first on 200 items, then on all 1,319. The
  first row is an attempt blocked by the client's network before any
  evaluation ran; it is kept unedited.
- `quality_paired.csv`: item-by-item comparison of FP8 and NVFP4 against BF16
  on all 1,319 items, with paired 95% intervals and exact McNemar p.
- `allreduce.csv`: cross-node all-reduce latency at each model's per-token
  message size, eager and replayed in a CUDA graph, two runs each.

Home-directory paths in `deepseek.csv` were replaced with `~` for publication
(`sed 's#/home/[a-z_][a-z0-9_-]*#~#g'`). Nothing else differs from the run output.
