#!/usr/bin/env bash
# Run on spark2 after the spark1 server passes G0.
set -euo pipefail
: "${SERVER_URL:?set SERVER_URL to the spark1 server, e.g. http://$SPARK1_CX7_A:8000}"
R=$1
mkdir -p "$R"
source ~/venvs/vllm029/bin/activate
vllm bench serve --base-url "$SERVER_URL" \
  --backend openai --endpoint /v1/completions \
  --model qwen --tokenizer $HOME/models/Qwen3.8-27B --served-model-name qwen \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --random-range-ratio 0 --seed 42 --max-concurrency 64 --num-prompts 320 \
  --num-warmups 32 --ignore-eos --temperature 0 \
  --percentile-metrics ttft,itl,tpot,e2el --metric-percentiles 50,99 \
  --save-result --save-detailed --result-dir "$R" \
  --result-filename vllm.json --disable-tqdm 2>&1 | tee "$R/vllm-raw.txt"
~/venvs/aiperf-uv/bin/aiperf profile --url "$SERVER_URL" \
  --endpoint-type completions --streaming --model qwen \
  --tokenizer $HOME/models/Qwen3.8-27B \
  --prompt-corpus random --random-corpus-style vllm --random-range-ratio 0 \
  --random-seed 42 --use-server-token-count --use-legacy-max-tokens \
  --extra-inputs '{"ignore_eos":true,"temperature":0}' \
  --isl 1024 --osl 256 --concurrency 64 --request-count 320 \
  --warmup-request-count 32 --request-timeout-seconds 600 \
  --output-artifact-dir "$R/aiperf" --export-level records \
  --ui none 2>&1 | tee "$R/aiperf-raw.txt"
