#!/usr/bin/env python3
"""Scrape vLLM scheduler and token counters at fixed intervals."""
import argparse
import csv
import signal
import threading
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--out', required=True)
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    columns = ['unix_time', 'elapsed_s', 'running', 'waiting', 'capacity',
               'deferred', 'kv_usage', 'preemptions', 'generation_tokens',
               'instant_tok_s', 'error']
    names = {'vllm:num_requests_running': 'running',
             'vllm:num_requests_waiting': 'waiting',
             'vllm:kv_cache_usage_perc': 'kv_usage',
             'vllm:num_preemptions_total': 'preemptions',
             'vllm:generation_tokens_total': 'generation_tokens'}
    start = time.monotonic()
    previous = None
    with open(args.out, 'w') as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        while not stop.is_set():
            now = time.monotonic()
            row = dict.fromkeys(columns, '')
            row.update(unix_time=time.time(), elapsed_s=now-start)
            try:
                with urllib.request.urlopen(args.url.rstrip('/')+'/metrics', timeout=5) as response:
                    body = response.read().decode()
                values = {}
                for family in text_string_to_metric_families(body):
                    for sample in family.samples:
                        key = names.get(sample.name)
                        if sample.name == 'vllm:num_requests_waiting_by_reason':
                            key = sample.labels.get('reason')
                        if key in columns:
                            values[key] = values.get(key, 0) + sample.value
                row.update(values)
                total = values.get('generation_tokens')
                if total is not None and previous is not None:
                    dt = now-previous[0]
                    if total >= previous[1]:
                        row['instant_tok_s'] = (total-previous[1])/dt
                if total is not None:
                    previous = (now, total)
            except Exception as exc:
                row['error'] = repr(exc)
            writer.writerow(row)
            out.flush()
            stop.wait(max(0, args.interval-(time.monotonic()-now)))


if __name__ == '__main__':
    main()
