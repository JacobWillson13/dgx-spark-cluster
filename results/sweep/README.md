# Concurrency sweep and GovReport

Results for REPORT.md section 9. Predictions were committed before the runs.

- `sweep.csv`: one row per setup and concurrency, from `vllm bench serve`.
  For DP2, throughput is summed over both servers' clients and latency is the
  mean of the two. Only rows with `valid` = 1 are reported; all are valid here.
- `echo.csv`: every echo correctness check in the session. The `S-DP2` rows
  come from the stopped first DP2 attempt; its gate passed before the stop,
  and its sweep data is not used.
- `govreport.csv`: 100 GovReport summaries per setup, with throughput,
  latency and ROUGE against the expert summaries.

GovReport text and model summaries are not committed. The sample is defined
by a seed and the row positions of the `ccdv/govreport-summarization` test
split; credit GAO and CRS as the original publishers.
