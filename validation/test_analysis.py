"""Tests for the sweep and GSM8K analysis scripts. Run from validation/:
    python -m unittest test_analysis
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BIN = Path(__file__).parent / "bin"


def run(script, *args):
    return subprocess.run([sys.executable, "-I", str(BIN / script), *args],
                          capture_output=True, text=True, check=True).stdout


def bench(d, name, c, tok_s, tpot, completed, failed=0):
    (d / f"bench-{name}.json").write_text(json.dumps({
        "num_prompts": completed + failed, "completed": completed, "failed": failed,
        "output_throughput": tok_s, "median_ttft_ms": 900.0, "median_tpot_ms": tpot,
        "p99_tpot_ms": tpot + 10, "p99_itl_ms": tpot + 20}))


class SweepCsv(unittest.TestCase):
    def rows(self, *specs):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "s.csv"
            run("sweep_csv.py", "--out", str(out), *specs)
            lines = out.read_text().splitlines()
            head = lines[0].split(",")
            return [dict(zip(head, l.split(","))) for l in lines[1:]]

    def test_single_endpoint(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            bench(d, "c1", 1, 4.7, 212.0, 5)
            bench(d, "c64", 64, 163.8, 370.0, 128)
            r = self.rows(f"1N={d}")
        self.assertEqual([x["concurrency"] for x in r], ["1", "64"])
        self.assertTrue(all(x["valid"] == "1" and x["clients"] == "1" for x in r))

    def test_two_endpoints_sum_throughput_and_average_latency(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            bench(d, "c32-e1", 16, 60.0, 240.0, 64)
            bench(d, "c32-e2", 16, 58.0, 246.0, 64)
            r = self.rows(f"DP2={d}")[0]
        self.assertEqual(r["clients"], "2")
        self.assertAlmostEqual(float(r["output_tok_s"]), 118.0)
        self.assertAlmostEqual(float(r["median_tpot_ms"]), 243.0)
        self.assertEqual(r["valid"], "1")

    def test_dead_endpoint_is_invalid_and_not_averaged(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            bench(d, "c4-e1", 2, 9.0, 218.0, 8)
            bench(d, "c4-e2", 2, 0.0, 0.0, 0, failed=8)
            bench(d, "c16-e1", 8, 32.0, 232.0, 32)  # partner file missing
            r = {x["concurrency"]: x for x in self.rows(f"DP2={d}")}
        self.assertEqual(r["4"]["valid"], "0")
        self.assertAlmostEqual(float(r["4"]["median_tpot_ms"]), 218.0)
        self.assertEqual(r["16"]["valid"], "0")


class PairedGsm8k(unittest.TestCase):
    def write(self, d, correct):
        m = d / "model"
        m.mkdir(parents=True)
        with open(m / "samples_gsm8k_2026.jsonl", "w") as f:
            for i, ok in enumerate(correct):
                for flt in ("strict-match", "flexible-extract"):
                    f.write(json.dumps({"doc_id": i, "filter": flt, "exact_match": float(ok)}) + "\n")

    def test_discordant_counts_and_exact_p(self):
        base = [1] * 60 + [0] * 40
        cand = [1] * 50 + [0] * 10 + [1] * 2 + [0] * 38  # 10 lost, 2 gained
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            self.write(a, base)
            self.write(b, cand)
            r = json.loads(run("paired_gsm8k.py", str(a), str(b)))
        self.assertEqual(r["n_paired"], 100)
        self.assertEqual((r["baseline_only_right"], r["candidate_only_right"]), (10, 2))
        self.assertAlmostEqual(r["diff_points"], -8.0)
        self.assertAlmostEqual(r["mcnemar_p"], 0.0386, places=4)  # exact binomial, n=12, k=2
        self.assertLess(r["ci95_points"][1], 0)

    def test_identical_runs(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            self.write(a, [1, 0, 1, 1])
            self.write(b, [1, 0, 1, 1])
            r = json.loads(run("paired_gsm8k.py", str(a), str(b)))
        self.assertEqual(r["diff_points"], 0.0)
        self.assertEqual(r["mcnemar_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
