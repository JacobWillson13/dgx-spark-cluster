#!/usr/bin/env python3
"""Paired comparison of two lm-eval GSM8K runs on the same items.

  python3 paired_gsm8k.py <baseline lmeval dir> <candidate lmeval dir> [--filter flexible-extract]

Each dir is an lm_eval --output_path with --log_samples; the newest
samples_gsm8k_*.jsonl under it is used. Prints accuracy for both, the
discordant counts (b: baseline right, candidate wrong; c: the reverse), the
difference with a 95% paired interval, and an exact two-sided McNemar p.
"""

import argparse
import glob
import json
import math
import os
import sys


def load(d, flt):
    files = sorted(glob.glob(os.path.join(d, "**", "samples_gsm8k*.jsonl"), recursive=True), key=os.path.getmtime)
    if not files:
        sys.exit(f"no samples_gsm8k*.jsonl under {d}")
    out = {}
    with open(files[-1]) as f:
        for line in f:
            r = json.loads(line)
            if r.get("filter", flt) != flt:
                continue
            out[r["doc_id"]] = int(float(r["exact_match"]) > 0)
    if not out:
        sys.exit(f"no rows with filter {flt} in {files[-1]}")
    return out, files[-1]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--filter", default="flexible-extract")
    a = ap.parse_args()
    base, fb = load(a.baseline, a.filter)
    cand, fc = load(a.candidate, a.filter)
    ids = sorted(set(base) & set(cand))
    n = len(ids)
    b = sum(1 for i in ids if base[i] and not cand[i])
    c = sum(1 for i in ids if cand[i] and not base[i])
    pa, pb = sum(base[i] for i in ids) / n, sum(cand[i] for i in ids) / n
    d = (c - b) / n
    se = math.sqrt(max((b + c) / n - d * d, 0) / n)
    print(json.dumps({
        "filter": a.filter, "n_paired": n, "baseline_file": fb, "candidate_file": fc,
        "baseline_acc": round(pa, 4), "candidate_acc": round(pb, 4),
        "diff_points": round(100 * d, 2), "ci95_points": [round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)],
        "baseline_only_right": b, "candidate_only_right": c, "mcnemar_p": round(mcnemar_exact(b, c), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
