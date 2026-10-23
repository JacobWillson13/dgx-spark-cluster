#!/usr/bin/env python3
"""Collect sweep.sh results into one CSV row per configuration and concurrency.

  python3 sweep_csv.py --out sweep.csv <config>=<run dir> [<config>=<run dir> ...]

For a two-endpoint (DP2) point, output tok/s is the sum of both clients and
latency figures are the mean over clients that completed requests; `clients`
says 2. `valid` is 0 when any request failed or an expected client file is
missing; such rows must not be reported as results.
"""

import argparse
import csv
import glob
import json
import os
import re

FIELDS = ["config", "concurrency", "clients", "num_prompts", "completed", "failed",
          "output_tok_s", "median_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "p99_itl_ms", "valid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+", help="config=run_dir")
    a = ap.parse_args()
    rows = []
    for spec in a.runs:
        config, d = spec.split("=", 1)
        groups, two = {}, set()
        for f in glob.glob(os.path.join(d, "bench-c*.json")):
            m = re.match(r"bench-c(\d+)(-e\d)?\.json$", os.path.basename(f))
            if m:
                groups.setdefault(int(m.group(1)), []).append(json.load(open(f)))
                if m.group(2):
                    two.add(int(m.group(1)))
        for c in sorted(groups):
            js = groups[c]
            ok = [j for j in js if j["completed"] > 0] or js
            mean = lambda k: sum(j[k] for j in ok) / len(ok)
            valid = int(all(j["failed"] == 0 for j in js) and (c not in two or len(js) == 2))
            rows.append({
                "config": config, "concurrency": c, "clients": len(js),
                "num_prompts": sum(j["num_prompts"] for j in js),
                "completed": sum(j["completed"] for j in js),
                "failed": sum(j["failed"] for j in js),
                "output_tok_s": round(sum(j["output_throughput"] for j in js), 3),
                "median_ttft_ms": round(mean("median_ttft_ms"), 2),
                "median_tpot_ms": round(mean("median_tpot_ms"), 2),
                "p99_tpot_ms": round(mean("p99_tpot_ms"), 2),
                "p99_itl_ms": round(mean("p99_itl_ms"), 2),
                "valid": valid,
            })
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
