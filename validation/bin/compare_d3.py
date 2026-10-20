#!/usr/bin/env python3
"""Compare saved vLLM and AIPerf summaries against the D3 gate."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('vllm', type=Path)
parser.add_argument('aiperf', type=Path)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
v = json.loads(args.vllm.read_text())
a = json.loads(args.aiperf.read_text())

rows = {
    'vllm': {
        'output_tokens': v['total_output_tokens'],
        'output_tok_s': v['output_throughput'],
        'request_s': v['request_throughput'],
        'ttft_p50_ms': v['p50_ttft_ms'],
        'ttft_p99_ms': v['p99_ttft_ms'],
        'itl_p50_ms': v['p50_itl_ms'],
        'duration_s': v['duration'],
        'input_tokens': v['total_input_tokens'],
    },
    'aiperf': {
        'output_tokens': a['total_output_tokens']['avg'],
        'output_tok_s': a['output_token_throughput']['avg'],
        'request_s': a['request_throughput']['avg'],
        'ttft_p50_ms': a['time_to_first_token']['p50'],
        'ttft_p99_ms': a['time_to_first_token']['p99'],
        'itl_p50_ms': a['inter_token_latency']['p50'],
        'duration_s': a['benchmark_duration']['avg'],
        'input_tokens': a['input_sequence_length']['sum'],
    },
}
gap = 100 * abs(rows['aiperf']['output_tok_s']/rows['vllm']['output_tok_s']-1)
same = rows['vllm']['output_tokens'] == rows['aiperf']['output_tokens'] == 320*256
result = {'metrics': rows, 'exact_output_total': same, 'gap_percent': gap,
          'pass': same and gap <= 3}
args.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['pass'] else 1)
