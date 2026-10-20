#!/usr/bin/env bash
# usage: launch.sh <run_dir> <vllm serve args...>
set -euo pipefail
R=$1; shift
BIN=$(cd -- "$(dirname -- "$0")" && pwd)
source ~/venvs/vllm029/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
# FlashInfer 0.6.18 jit/cpp_ext.py:94-112,346-365 (verified installed source).
export MAX_JOBS=2
export FLASHINFER_NVCC_THREADS=2
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
command -v ninja >/dev/null || { echo "FAIL: ninja not on PATH"; exit 1; }
command -v setsid >/dev/null
test -r "$BIN/memory_watchdog.py"
: > "$R/watchdog.log"
nohup setsid vllm serve "$@" > "$R/server.log" 2>&1 < /dev/null &
server_pid=$!
echo "$server_pid" > "$R/server.pid"
nohup python "$BIN/memory_watchdog.py" --pid "$server_pid" \
  --log "$R/watchdog.log" > "$R/watchdog-stderr.log" 2>&1 < /dev/null &
watchdog_pid=$!
echo "$watchdog_pid" > "$R/watchdog.pid"
for ((attempt=0; attempt<60; attempt++)); do
  grep -q '"event": "watching"' "$R/watchdog.log" && exit 0
  kill -0 "$watchdog_pid" 2>/dev/null || break
  sleep 0.1
done
echo "FAIL: memory watchdog did not initialize; stopping server group"
kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
sleep 10
kill -KILL -- "-$server_pid" 2>/dev/null || true
exit 1
