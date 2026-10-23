#!/usr/bin/env bash
# Concurrency sweep with vllm bench, against one server or two independent ones.
# Run on spark2 after the server(s) pass G0.
#
#   sweep.sh <run dir> <tokenizer dir> "<concurrency list>" <endpoint> [endpoint2]
#
# One endpoint: each point runs at concurrency C.
# Two endpoints (DP2): two clients run at the same time, C/2 each, seeds 0 and 1,
# so total concurrency is C. Points with C < 2 are skipped for DP2.
# Prompts per point: max(5, 4C) up to C=32, then 2C, split evenly for DP2.
set -euo pipefail
R=$1; TOK=$2; CS=$3; E1=$4; E2=${5:-}
mkdir -p "$R"
source ~/venvs/vllm029/bin/activate

bench() {  # endpoint concurrency prompts seed tag
  vllm bench serve --backend vllm --base-url "$1" \
    --model qwen --tokenizer "$TOK" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --ignore-eos --seed "$4" --num-prompts "$3" --max-concurrency "$2" --num-warmups 2 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$R" --result-filename "bench-$5.json" \
    > "$R/bench-$5.txt" 2>&1
}

up() {  # fail fast if an endpoint is not serving
  for e in "$@"; do
    curl -sf -m 5 "$e/health" >/dev/null || { echo "$(date +%T) ENDPOINT_DOWN $e; stopping sweep"; exit 3; }
  done
}

for C in $CS; do
  up "$E1" ${E2:+"$E2"}
  if [ "$C" -le 32 ]; then N=$(( C * 4 > 5 ? C * 4 : 5 )); else N=$(( C * 2 )); fi
  if [ -z "$E2" ]; then
    echo "$(date +%T) c$C n$N"
    bench "$E1" "$C" "$N" 0 "c$C"
  else
    [ "$C" -lt 2 ] && { echo "skip c$C for two endpoints"; continue; }
    H=$(( C / 2 )); HN=$(( (N + 1) / 2 ))
    echo "$(date +%T) c$C n$N as 2 x c$H n$HN"
    bench "$E1" "$H" "$HN" 0 "c$C-e1" & P1=$!
    bench "$E2" "$H" "$HN" 1 "c$C-e2" & P2=$!
    wait $P1; wait $P2
    up "$E1" "$E2"
  fi
done
echo "$(date +%T) done"
