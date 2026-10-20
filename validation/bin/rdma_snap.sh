#!/usr/bin/env bash
for i in enp1s0f1np1 enP2p1s0f1np1; do
  ethtool -S "$i" | awk -v i="$i" '/[rt]x_vport_rdma_unicast_bytes/ {sub(":","",$1); print i, $1, $2}'
done
