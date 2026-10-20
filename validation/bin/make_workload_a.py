#!/usr/bin/env python3
"""Generate vLLM random prompts and verify server-side input lengths."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from vllm.benchmarks.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--count', type=int, default=320)
parser.add_argument('--seed', type=int, default=20261015)
args = parser.parse_args()
tok = get_tokenizer(os.path.expanduser('~/models/Qwen3.8-27B'))
samples = RandomDataset(random_seed=args.seed).sample(
    tok, num_requests=args.count, input_len=1024, output_len=256,
    prefix_len=0, range_ratio=0.0)


def verify(sample):
    payload = {'model': 'qwen', 'prompt': sample.prompt, 'add_special_tokens': False}
    req = urllib.request.Request(os.environ.get('SERVER_URL', 'http://localhost:8000') + '/tokenize',
                                 json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        count = json.load(response)['count']
    if count != 1024:
        raise ValueError(f'server input count {count}, expected 1024')
    return {'text': sample.prompt, 'output_length': 256}


with ThreadPoolExecutor(max_workers=16) as pool:
    rows = list(pool.map(verify, samples))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(''.join(json.dumps(row, ensure_ascii=True)+'\n' for row in rows))
print(json.dumps({'count': len(rows), 'server_isl': 1024, 'osl': 256,
                  'seed': args.seed, 'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}))
