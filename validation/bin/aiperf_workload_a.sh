#!/usr/bin/env bash
# Run on spark2: aiperf_workload_a.sh <artifact_dir> <concurrency>
set -euo pipefail
: "${SERVER_URL:?set SERVER_URL to the spark1 server, e.g. http://$SPARK1_CX7_A:8000}"
R=$1
C=$2
mkdir -p "$R"
~/venvs/vllm029/bin/python ~/validation/bin/make_workload_a.py \
  --out "$R/inputs.jsonl" --count "$((5*C))" --seed "$((20261015+C))" \
  | tee "$R/input-verification.json"
~/venvs/aiperf-uv/bin/aiperf profile --url "$SERVER_URL" \
  --endpoint-type completions --streaming --model qwen \
  --tokenizer $HOME/models/Qwen3.8-27B \
  --custom-dataset-type single-turn --input-file "$R/inputs.jsonl" \
  --random-seed 42 --use-server-token-count --use-legacy-max-tokens \
  --extra-inputs '{"ignore_eos":true,"temperature":0}' \
  --osl 256 --concurrency "$C" --request-count "$((5*C))" \
  --warmup-request-count 32 --request-timeout-seconds 600 \
  --output-artifact-dir "$R/aiperf" --export-level records \
  --ui none 2>&1 | tee "$R/aiperf-raw.txt"
