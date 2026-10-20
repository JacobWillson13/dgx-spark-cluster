#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import sys
import urllib.request


def complete(url, model, prompt, max_tokens, ignore_eos=False, stop=None):
    body = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0}
    if stop is not None:
        body['stop'] = stop
    if ignore_eos:
        body['ignore_eos'] = True
    req = urllib.request.Request(url.rstrip('/')+'/v1/completions',
                                 json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def first_int(text):
    match = re.search(r'-?\d[\d,]*', text)
    return int(match.group().replace(',', '')) if match else None


def smoke(eps, model):
    out = []
    for ep in eps:
        france = complete(ep, model, 'The capital of France is', 16)['choices'][0]['text']
        mult = complete(ep, model, 'What is 17 times 23? Answer with the number only.', 16)['choices'][0]['text']
        out.append({'endpoint': ep, 'france': france, 'ok_france': 'Paris' in france,
                    'mult': mult, 'ok_mult': first_int(mult) == 391})
    return out, all(r['ok_france'] and r['ok_mult'] for r in out)


def arithmetic_pairs(n, seed):
    if not 1 <= n <= len({a*b for a in range(11, 100) for b in range(11, 100)}):
        raise ValueError('n exceeds the number of distinct products available')
    rng, pairs, seen = random.Random(seed), [], set()
    while len(pairs) < n:
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        if a*b not in seen:
            seen.add(a*b)
            pairs.append((a, b))
    return pairs


def request_text(ep, model, prompt, max_tokens=16, stop=None):
    try:
        kwargs = {'stop': stop} if stop is not None else {}
        text = complete(ep, model, prompt, max_tokens, **kwargs)['choices'][0]['text']
        return {'text': text, 'got': first_int(text)}
    except Exception as exc:
        return {'text': None, 'got': None, 'error': repr(exc)}


def distinct(eps, model, n, seed, concurrency):
    pairs = arithmetic_pairs(n, seed)
    def one(i):
        a, b = pairs[i]
        ep = eps[i % len(eps)]
        result = request_text(ep, model, f'What is {a} times {b}? Answer with the number only.')
        return {'endpoint': ep, 'a': a, 'b': b, 'expected': a*b, **result,
                'ok': result['got'] == a*b,
                'numeric_only': result['text'] is not None and bool(
                    re.fullmatch(r'\s*-?\d[\d,]*[.!]?\s*', result['text']))}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        out = list(pool.map(one, range(n)))
    return out, all(r['ok'] for r in out)


def echo_verdict(text, own_number, numbers):
    """Judge text ownership independently of first-integer accuracy/batch equality."""
    text = text or ''
    others = [i for i, value in enumerate(numbers)
              if value != own_number and str(value) in text]
    own = str(own_number) in text
    has_digits = bool(text.strip()) and bool(re.search(r'\d', text))
    return {'contains_own_number': own, 'other_request_indices': others,
            'nonempty_with_digits': has_digits,
            'ok': own and not others and has_digits,
            'misrouting': bool(others),
            'other_request_index': others[0] if others else None}


def chat_echo_text(ep, model, number):
    body = {'model': model,
            'messages': [{'role': 'user', 'content': f'Repeat this number exactly, and nothing else: {number}'}],
            'chat_template_kwargs': {'enable_thinking': False},
            'temperature': 0, 'max_tokens': 16}
    request = urllib.request.Request(ep.rstrip('/')+'/v1/chat/completions',
        json.dumps(body).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.load(response)
        message = data['choices'][0]['message']
        text = message.get('content') or ''
        return {'text': text, 'got': first_int(text), 'message': message,
                'finish_reason': data['choices'][0].get('finish_reason'), 'usage': data.get('usage')}
    except Exception as exc:
        return {'text': None, 'got': None, 'error': repr(exc)}


def echo(eps, model, n, seed, concurrency, echo_mode="completions"):
    if not 1 <= n <= 900000:
        raise ValueError('echo n must be in 1..900000')
    numbers = random.Random(seed).sample(range(100000, 1000000), n)
    def one(i):
        ep = eps[i % len(eps)]
        if echo_mode == 'chat-nothink':
            result = chat_echo_text(ep, model, numbers[i])
        elif echo_mode in ('completions', 'completions-long'):
            result = request_text(ep, model,
                f'Repeat this number exactly, and nothing else: {numbers[i]}\nAnswer:',
                max_tokens=256 if echo_mode == 'completions-long' else 16)
        else:
            raise ValueError(f'unknown echo mode: {echo_mode}')
        return {'endpoint': ep, 'expected': numbers[i], **result,
                **echo_verdict(result['text'], numbers[i], numbers),
                'accuracy_ok': result['got'] == numbers[i],
                'numeric_only': result['text'] is not None and bool(
                    re.fullmatch(r'\s*-?\d[\d,]*[.!]?\s*', result['text']))}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        out = list(pool.map(one, range(n)))
    return out, all(r['ok'] for r in out)


def echo_ownership(results):
    numbers = [r['expected'] for r in results]
    return [{**r, **echo_verdict(r.get('text'), r['expected'], numbers)} for r in results]


def echo_validity(results, preflight=False):
    judged = echo_ownership(results)
    passed = sum(r['ok'] for r in judged)
    valid = passed == 8 and len(judged) == 8 if preflight else passed*100 >= 95*len(judged)
    return {'valid': valid and not any('error' in r for r in judged),
            'ownership_matches': passed, 'total': len(judged),
            'required_percent': 100 if preflight else 95,
            'failed_items': [r for r in judged if not r['ok']]}


def echo_preflight(eps, model, seed, echo_mode='auto', smoke=None):
    attempts = []
    def attempt(mode):
        results, _ = echo(eps, model, 8, seed, 1, mode)
        validity = echo_validity(results, preflight=True)
        attempts.append({'echo_mode': mode, 'results': results, 'test_validity': validity})
        return results, validity['valid']
    if echo_mode != 'auto':
        results, valid = attempt(echo_mode)
        return results, echo_mode, attempts
    thinking = '<think>' in json.dumps(smoke or {}).lower()
    if not thinking:
        results, valid = attempt('completions')
        thinking = any('<think>' in (r.get('text') or '').lower() for r in results)
        if not thinking:
            return results, 'completions', attempts
    for mode in ('chat-nothink', 'completions-long'):
        results, valid = attempt(mode)
        if valid:
            return results, mode, attempts
    return results, 'completions-long', attempts


def echo_gate(results, reference):
    refs = {r['expected']: r for r in echo_ownership(reference['results'])}
    if len(refs) != len(results) or set(refs) != {r['expected'] for r in results}:
        raise ValueError('echo reference has different or duplicate inputs')
    eligible, excluded, failures = [], [], []
    judged = echo_ownership(results)
    for result in judged:
        ref = refs[result['expected']]
        result['eligible'] = ref['ok']
        if ref['ok']:
            eligible.append(result)
            if not result['ok']:
                failures.append(result)
        else:
            excluded.append({**result, 'reference_text': ref.get('text')})
    cross = [r for r in judged if r['misrouting']]
    return {'pass': not failures and not cross and not any('error' in r for r in judged),
            'matches': sum(r['ok'] for r in eligible), 'total': len(eligible),
            'failures': failures, 'excluded_items': excluded,
            'cross_request_matches': cross}


def compare_reference(results, reference):
    is_echo = 'a' not in results[0]
    def key(r):
        return r['expected'] if is_echo else (r['a'], r['b'])
    def identity(r):
        return {'expected': r['expected']} if is_echo else {'a': r['a'], 'b': r['b']}
    refs = {key(r): r for r in reference['results']}
    if (len(refs) != len(results) or len(refs) != len(reference['results'])
            or set(refs) != {key(r) for r in results}):
        raise ValueError('reference and concurrent run contain different or duplicate inputs')
    if is_echo:
        differences, text_differences = [], []
        for result in results:
            ref = refs[key(result)]
            result['reference_got'] = ref['got']
            result['matches_reference'] = result['got'] == ref['got']
            result['matches_reference_text'] = result['text'] == ref['text']
            if not result['matches_reference']:
                differences.append({**result, 'reference_text': ref['text']})
            if not result['matches_reference_text']:
                text_differences.append({**result, 'reference_text': ref['text']})
        return {'gating': False, 'matches': len(results)-len(differences),
                'total': len(results), 'differences': differences,
                'text_matches': len(results)-len(text_differences),
                'text_differences': text_differences,
                'cross_request_matches': [r for r in results if r.get('misrouting')],
                'non_numeric': [r for r in results if not r.get('numeric_only')],
                'reference_non_numeric': [r for r in reference['results'] if not r.get('numeric_only')]}
    differences, cross_matches, non_numeric = [], [], []
    for result in results:
        pair = key(result)
        ref = refs[pair]
        result['reference_got'] = ref['got']
        result['matches_reference'] = result['got'] == ref['got']
        if not result['matches_reference']:
            result['reference_text'] = ref['text']
            differences.append(result)
        if not result['numeric_only']:
            non_numeric.append(result)
        if result['got'] is None or result['matches_reference']:
            continue
        matches = []
        for other in results:
            other_pair = key(other)
            if other_pair == pair:
                continue
            kinds = [kind for kind, value in (
                ('input_number' if is_echo else 'true_product', other['expected']),
                ('c1_answer', refs[other_pair]['got']),
                ('concurrent_answer', other['got'])) if value == result['got']]
            if kinds:
                matches.append({**identity(other), 'kinds': kinds})
        if matches:
            cross_matches.append({**identity(result),
                                  'got': result['got'], 'text': result['text'], 'matches': matches})
    return {'matches': len(results)-len(differences), 'total': len(results),
            'differences': differences, 'cross_request_matches': cross_matches,
            'non_numeric': non_numeric,
            'reference_non_numeric': [r for r in reference['results'] if not r.get('numeric_only', False)]}


def fixed(eps, model, tokens):
    result = complete(eps[0], model, 'Write a long story about a mountain.', tokens, ignore_eos=True)
    usage = result['usage']
    ok = usage['completion_tokens'] == tokens
    return [{'usage': usage, 'finish': result['choices'][0]['finish_reason'], 'ok': ok}], ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'echo', 'distinct', 'fixed'])
    parser.add_argument('--endpoints', nargs='+', required=True)
    parser.add_argument('--model', default='qwen')
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--concurrency', type=int)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--preflight', action='store_true', help='echo: 8 items at C1, require 8/8')
    parser.add_argument('--echo-mode', choices=['auto', 'completions', 'chat-nothink', 'completions-long'])
    parser.add_argument('--smoke-report', type=Path, help='preflight uses <think> in smoke to select chat-first')
    parser.add_argument('--preflight-report', type=Path, help='passed version3 preflight required for N-sized echo')
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.preflight:
        if args.mode != 'echo' or args.reference:
            parser.error('--preflight is only for echo without --reference')
        args.n, args.concurrency = 8, 1
    if args.echo_mode and args.mode != 'echo':
        parser.error('--echo-mode is only for echo')
    if args.smoke_report and not args.preflight:
        parser.error('--smoke-report is only for echo --preflight')
    if args.echo_mode == 'auto' and not args.preflight:
        parser.error('--echo-mode auto is only for preflight; N-sized runs reuse the selected mode')
    if args.preflight_report and args.mode != 'echo':
        parser.error('--preflight-report is only for echo')
    concurrency = args.concurrency if args.concurrency is not None else args.n
    if concurrency < 1 or args.n < 1:
        parser.error('n and concurrency must be positive')
    if args.reference and args.mode not in ('echo', 'distinct'):
        parser.error('--reference is only for echo or distinct')
    if args.mode == 'echo' and concurrency != 1 and not args.reference:
        parser.error('concurrent echo requires --reference from C=1')
    if args.mode == 'echo' and not args.preflight:
        if not args.preflight_report:
            parser.error('N-sized echo requires --preflight-report from an 8/8 C1 preflight')
        preflight = json.loads(args.preflight_report.read_text())
        if any(preflight.get(k) != v for k, v in (('mode', 'echo'), ('model', args.model),
                ('seed', args.seed), ('n', 8), ('concurrency', 1), ('preflight', True), ('echo_prompt_version', 3))):
            parser.error('preflight metadata does not match this version3 run')
        selected_mode = preflight.get('echo_mode', 'completions')
        if args.echo_mode is not None and args.echo_mode != selected_mode:
            parser.error('echo mode mismatch with preflight')
        args.echo_mode = selected_mode
        if not echo_validity(preflight['results'], preflight=True)['valid']:
            parser.error('PROMPT PROBLEM: preflight did not pass 8/8 ownership')
    reference = None
    if args.reference:
        reference = json.loads(args.reference.read_text())
        for field, expected in (('mode', args.mode), ('model', args.model),
                                ('n', args.n), ('seed', args.seed), ('concurrency', 1)):
            if field == 'model' and args.mode == 'distinct' and field not in reference:
                continue  # Legacy arithmetic reference artifacts predate model metadata.
            if reference.get(field) != expected:
                parser.error(f'reference {field} must be {expected!r}')
        if args.mode == 'echo' and reference.get('echo_prompt_version') != 3:
            parser.error('echo reference must use prompt version 3; old references are invalid')
        if args.mode == 'echo' and reference.get('echo_mode', 'completions') != args.echo_mode:
            parser.error('echo mode mismatch with C1 reference')
        if any('error' in r for r in reference['results']):
            print('REQUEST ERROR: reference contains transport errors; no requests sent', flush=True)
            return 1
        if args.mode == 'echo' and not echo_validity(reference['results'])['valid']:
            print('PROMPT PROBLEM: C1 reference ownership below 95%; no concurrent requests sent', flush=True)
            print(json.dumps(reference['results']), flush=True)
            return 2
    preflight_attempts = None
    if args.mode == 'smoke':
        results, ok = smoke(args.endpoints, args.model)
    elif args.mode == 'echo':
        if args.preflight:
            smoke_report = json.loads(args.smoke_report.read_text()) if args.smoke_report else None
            results, args.echo_mode, preflight_attempts = echo_preflight(
                args.endpoints, args.model, args.seed, args.echo_mode or 'auto', smoke_report)
            ok = echo_validity(results, preflight=True)['valid']
        else:
            results, ok = echo(args.endpoints, args.model, args.n, args.seed, concurrency, args.echo_mode)
    elif args.mode == 'distinct':
        results, ok = distinct(args.endpoints, args.model, args.n, args.seed, concurrency)
    else:
        results, ok = fixed(args.endpoints, args.model, args.tokens)
    accuracy_matches = sum(r.get('accuracy_ok', r.get('ok', False)) for r in results)
    accuracy_ok = accuracy_matches == len(results)
    comparison = compare_reference(results, reference) if reference is not None else None
    validity = echo_validity(results, args.preflight) if args.mode == 'echo' and reference is None else None
    gate = echo_gate(results, reference) if args.mode == 'echo' and reference is not None else None
    if gate is not None:
        ok = gate['pass']
    elif validity is not None:
        ok = validity['valid']
    gating = args.mode not in ('echo', 'distinct') or (args.mode == 'echo' and reference is not None)
    transport_error = any('error' in r for r in results)
    ok = ok and not transport_error
    report = {'mode': args.mode, 'model': args.model, 'pass': ok, 'gating': gating,
              'n': args.n, 'seed': args.seed, 'concurrency': concurrency, 'results': results}
    if args.mode == 'echo':
        report.update(echo_prompt_version=3, echo_gate_version=4, echo_mode=args.echo_mode, accuracy_pass=accuracy_ok,
                      accuracy_matches=accuracy_matches, ownership_matches=sum(r['ok'] for r in results),
                      gate_matches=gate['matches'] if gate else None, gate_total=gate['total'] if gate else None,
                      preflight=args.preflight, preflight_report=str(args.preflight_report) if args.preflight_report else None,
                      reference=str(args.reference) if args.reference else None)
        if preflight_attempts is not None:
            report['preflight_attempts'] = preflight_attempts
            report['selected_echo_mode'] = args.echo_mode if validity['valid'] else None
        if validity is not None:
            report['test_validity'] = validity
            report['prompt_problem'] = not validity['valid']
        if gate is not None:
            report['ownership_gate'] = gate
    if comparison is not None:
        report['reference_comparison'] = comparison
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    failures = [r for r in results if not r.get('ok', r.get('ok_france', True) and r.get('ok_mult', True))]
    status = 'PASS' if ok else 'FAIL'
    if not gating:
        status = 'ACCURACY (non-gating)'
    if validity is not None:
        status = 'VALID TEST' if validity['valid'] else 'PROMPT PROBLEM (not a server gate failure)'
    if args.mode == 'echo' and comparison is not None:
        print(f"echo reference parsed matches (non-gating): {comparison['matches']}/{args.n}", flush=True)
    print(f'{args.mode}: {status} ({len(results)-len(failures)}/{len(results)})', flush=True)
    if args.mode == 'echo':
        print(f'echo mode: {args.echo_mode}', flush=True)
        if preflight_attempts is not None:
            print('PREFLIGHT_ATTEMPTS', json.dumps(preflight_attempts), flush=True)
        if gate is not None:
            print(f"echo eligible gate: {'PASS' if ok else 'FAIL'} ({gate['matches']}/{gate['total']}); excluded {len(gate['excluded_items'])}; cross-request matches {len(gate['cross_request_matches'])}", flush=True)
            print('EXCLUDED_C1_ITEMS', json.dumps(gate['excluded_items']), flush=True)
        if args.preflight and not ok:
            print('PREFLIGHT_ALL_8_STRINGS', json.dumps(results), flush=True)
        print(f'echo first-integer accuracy (non-gating): {accuracy_matches}/{len(results)}', flush=True)
    for failure in failures:
        print('  MISMATCH', json.dumps(failure), flush=True)
    if args.reference:
        print('REFERENCE_COMPARISON', json.dumps(report['reference_comparison']), flush=True)
    if validity is not None and not validity['valid'] and not transport_error:
        return 2
    return 1 if transport_error or (gating and not ok) else 0


if __name__ == '__main__':
    sys.exit(main())
