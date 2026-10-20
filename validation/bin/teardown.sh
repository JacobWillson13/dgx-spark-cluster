#!/usr/bin/env bash
pkill -f 'vllm serve' 2>/dev/null
source ~/venvs/vllm029/bin/activate
ray stop --force >/dev/null 2>&1
sleep 5
left=$(pgrep -af 'vllm serve|raylet|gcs_server' | grep -v pgrep)
[ -z "$left" ] && echo "clean" || { echo "FAIL processes remain:"; echo "$left"; exit 1; }
