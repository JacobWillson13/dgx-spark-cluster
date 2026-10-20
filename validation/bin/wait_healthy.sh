#!/usr/bin/env bash
# usage: wait_healthy.sh <url> <timeout_s> <logfile>
url=$1; t=${2:-600}; log=$3
for ((s=0; s<t; s+=10)); do
  curl -sf "$url/health" >/dev/null && { echo "healthy after ${s}s"; exit 0; }
  sleep 10
done
echo "FAIL: not healthy after ${t}s"; tail -n 50 "$log"; exit 1
