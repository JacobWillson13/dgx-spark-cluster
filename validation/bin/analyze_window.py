#!/usr/bin/env python3
"""Analyze a complete AIPerf run against a two-second vLLM metrics CSV."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--metrics', type=Path, required=True)
parser.add_argument('--records', type=Path, required=True)
parser.add_argument('--concurrency', type=int, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
records = [json.loads(line) for line in args.records.read_text().splitlines()]
warmups = [r for r in records if r['metadata'].get('benchmark_phase') == 'warmup']
records = [r for r in records if r['metadata'].get('benchmark_phase') == 'profiling']
errors = [r for r in records if r.get('error')]
def exact_lengths(r):
    return (r.get('metrics', {}).get('input_sequence_length', {}).get('value') == 1024
            and r.get('metrics', {}).get('output_sequence_length', {}).get('value') == 256)
bad_lengths = [r for r in records if not exact_lengths(r)]
accepted = len(records) == 5*args.concurrency and len(warmups) == 32 and not errors and not bad_lengths
starts = [r['metadata']['request_start_ns']/1e9 for r in records]
ends = sorted(r['metadata']['request_end_ns']/1e9 for r in records)
# Exclude the first completed wave and everything after the final dispatch.
start = ends[args.concurrency-1]
end = max(starts)
metrics = []
with args.metrics.open() as source:
    for row in csv.DictReader(source):
        if not row['error']:
            metrics.append({k: float(v) if v else None for k, v in row.items() if k != 'error'})
window = [m for m in metrics if start <= m['unix_time'] <= end]
if len(window) < 2:
    raise ValueError('Not enough samples in the sustained-load window')
first, last = window[0], window[-1]
duration = last['unix_time']-first['unix_time']
aggregate = (last['generation_tokens']-first['generation_tokens'])/duration
running = statistics.mean(m['running'] for m in window)
eligible = [r for r in records if not r.get('error') and exact_lengths(r) and r['metadata']['request_start_ns']/1e9 >= first['unix_time']
            and r['metadata']['request_end_ns']/1e9 <= last['unix_time']]
rates = [r['metrics']['output_sequence_length']['value'] /
         (r['metrics']['request_latency']['value']/1000) for r in eligible]
median = statistics.median(rates) if rates else None
per_running = aggregate/running
gap = 100*abs(per_running/median-1) if median is not None else None
capacity = [m for m in metrics if min(starts) <= m['unix_time'] <= max(ends)
            and (m['capacity'] or 0) > 0]
sustained_capacity = [m for m in window if (m['capacity'] or 0) > 0]
running_counts = Counter(m['running'] for m in window)
plateau_running = running_counts.most_common(1)[0][0]
plateau = [m for m in window if m['running'] == plateau_running]
first_capacity = capacity[0] if capacity else None
def first_sustained(rows, predicate):
    # Three consecutive two-second scrapes: at least four seconds observed.
    for i in range(len(rows)-2):
        group = rows[i:i+3]
        if all(predicate(m) for m in group) and group[-1]['unix_time']-group[0]['unix_time'] <= 6:
            return group[0]
    return None

first_window_wait = first_sustained(window, lambda m: (m['capacity'] or 0) > 0)
profile_metrics = [m for m in metrics if min(starts) <= m['unix_time'] <= max(ends)]
first_kv_wait = first_sustained(profile_metrics, lambda m: (m['capacity'] or 0) > 0 and m['kv_usage'] >= .98)
def onset(row):
    return ({'s_after_profile_start': row['unix_time']-min(starts),
             'running': row['running'], 'kv_fraction': row['kv_usage'],
             'capacity_waits': row['capacity']} if row else None)
result = {
    'accepted_zero_error_workload': accepted,
    'acceptance_note': 'diagnostic only: errors/lengths/counts invalid' if not accepted else 'exact Workload A, zero errors',
    'profile_errors': len(errors), 'error_types': dict(Counter(r['error']['type'] for r in errors)),
    'invalid_length_records': len(bad_lengths), 'warmup_requests': len(warmups),
    'first_sustained_window_capacity_wait': onset(first_window_wait),
    'first_sustained_kv_near_full_wait': onset(first_kv_wait),
    'sustained_onset_definition': '3 consecutive scrapes (~4 seconds); KV-near-full threshold .98',
    'profile_requests': len(records), 'profile_duration_s': max(ends)-min(starts),
    'window_rule': 'after first C completions, before final dispatch; align inward to sampler timestamps',
    'ramp_s': first['unix_time']-min(starts),
    'drain_s': max(ends)-last['unix_time'], 'window_duration_s': duration,
    'steady_aggregate_tok_s': aggregate, 'mean_running': running,
    'running_plateau_definition': 'modal running count in the fixed sustained-load window',
    'running_plateau': plateau_running, 'running_plateau_samples': len(plateau),
    'window_samples': len(window),
    'running_histogram': dict(sorted(running_counts.items())),
    'plateau_kv_mean': statistics.mean(m['kv_usage'] for m in plateau),
    'plateau_kv_min': min(m['kv_usage'] for m in plateau),
    'plateau_kv_max': max(m['kv_usage'] for m in plateau),
    'first_capacity_s_after_profile_start': first_capacity['unix_time']-min(starts) if first_capacity else None,
    'first_capacity_running': first_capacity['running'] if first_capacity else None,
    'first_capacity_kv': first_capacity['kv_usage'] if first_capacity else None,
    'first_capacity_waiting': first_capacity['capacity'] if first_capacity else None,
    'steady_tok_s_per_running': per_running,
    'contained_request_count': len(eligible), 'median_request_e2e_tok_s': median,
    'normalized_mean_median_gap_percent': gap, 'agrees_within_2_percent': gap is not None and gap <= 2,
    'max_kv_fraction': max(m['kv_usage'] for m in metrics if m['kv_usage'] is not None),
    'max_running': max(m['running'] for m in metrics),
    'max_capacity_wait': max((m['capacity'] or 0) for m in metrics),
    'capacity_samples': len(capacity),
    'sustained_capacity_samples': len(sustained_capacity),
    'sustained_capacity_max': max((m['capacity'] for m in sustained_capacity), default=0),
    'kv_at_capacity_min': min((m['kv_usage'] for m in capacity), default=None),
    'kv_at_capacity_max': max((m['kv_usage'] for m in capacity), default=None),
    'preemptions_delta': metrics[-1]['preemptions']-metrics[0]['preemptions'],
}
args.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
