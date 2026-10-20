#!/usr/bin/env bash
# usage: scan_log.sh <logfile>
log=$1; rc=0
for pat in 'Killed' 'NV_ERR_NO_MEMORY' 'Gloo connectFullMesh' 'Traceback' 'CUDA error'; do
  n=$(grep -c "$pat" "$log"); [ "$n" -gt 0 ] && { echo "FATAL $pat x$n"; rc=1; }
done
echo "INFO fp4_autotuner_fallbacks=$(grep -c 'No tuned config covers' "$log")"
grep -oE "cudagraph_capture_sizes': \[[^]]*\]" "$log" | head -1
exit $rc
