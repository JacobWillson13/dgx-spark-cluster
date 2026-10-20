import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('checker', Path(__file__).parent/'bin/check.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def write_preflight(directory):
    path = directory/'preflight.json'
    numbers = random.Random(0).sample(range(100000, 1000000), 8)
    path.write_text(json.dumps({'mode': 'echo', 'model': 'qwen', 'seed': 0, 'n': 8,
        'concurrency': 1, 'preflight': True, 'echo_prompt_version': 3,
        'results': [{'expected': n, 'got': n, 'text': str(n)} for n in numbers]}))
    return path


class CheckTests(unittest.TestCase):
    def test_seed_zero_pairs_match_recorded_run(self):
        recorded = json.loads((Path(__file__).parent/'fixtures/arithmetic_seed0_n256.json').read_text())
        self.assertEqual(checker.arithmetic_pairs(256, 0), [tuple(p) for p in recorded['pairs']])

    def test_echo_flags_cross_request_values(self):
        numbers = random.Random(0).sample(range(100000, 1000000), 64)
        def swapped(url, model, prompt, max_tokens, **kwargs):
            value = int(prompt.splitlines()[0].rsplit(' ', 1)[1])
            answer = numbers[1] if value == numbers[0] else value
            return {'choices': [{'text': str(answer)}]}
        with patch.object(checker, 'complete', side_effect=swapped):
            results, ok = checker.echo(['http://test'], 'qwen', 64, 0, 64)
        self.assertFalse(ok)
        self.assertTrue(results[0]['misrouting'])
        self.assertEqual(results[0]['other_request_index'], 1)
        self.assertEqual(sum(r['ok'] for r in results), 63)

    def test_echo_delimits_trailing_digits_without_stop(self):
        def answer(url, model, prompt, max_tokens, **kwargs):
            self.assertTrue(prompt.endswith('\nAnswer:'))
            self.assertEqual(max_tokens, 16)
            self.assertNotIn('stop', kwargs)
            value = prompt.splitlines()[0].rsplit(' ', 1)[1]
            return {'choices': [{'text': value}]}
        with patch.object(checker, 'complete', side_effect=answer):
            results, ok = checker.echo(['http://test'], 'qwen', 2, 0, 1)
        self.assertTrue(ok)
        # The old completion continues trailing prompt digits; first_int must
        # preserve this failure, rather than extracting the echoed input later.
        self.assertEqual(checker.first_int('2\n\n5997482'), 2)
        self.assertNotEqual(checker.first_int('2\n\n5997482'), 599748)

    def test_echo_own_number_with_trailing_continuation_passes(self):
        result = checker.echo_verdict('2\n\n5997482', 599748, [599748, 314410])
        self.assertTrue(result['ok'])
        self.assertNotEqual(checker.first_int('2\n\n5997482'), 599748)

    def test_echo_other_number_even_with_own_fails(self):
        result = checker.echo_verdict('599748 314410', 599748, [599748, 314410])
        self.assertFalse(result['ok'])
        self.assertEqual(result['other_request_indices'], [1])

    def test_echo_zeros_empty_and_missing_number_fail(self):
        for text in ('0000000000000000', '', None, 'words only', '42'):
            with self.subTest(text=text):
                self.assertFalse(checker.echo_verdict(text, 599748, [599748, 314410])['ok'])

    def test_echo_changed_but_own_number_passes_reference_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, out = Path(tmp)/'ref.json', Path(tmp)/'out.json'
            preflight = write_preflight(Path(tmp))
            def run(concurrency, reference=None, changed=False, other=False):
                argv = ['check.py', 'echo', '--endpoints', 'http://test',
                        '--n', '2', '--concurrency', str(concurrency), '--preflight-report', str(preflight), '--out', str(out)]
                if reference:
                    argv += ['--reference', str(reference)]
                def answer(url, model, prompt, max_tokens, **kwargs):
                    own = prompt.splitlines()[0].rsplit(' ', 1)[1]
                    text = '2 ' + own + '2' if changed else own
                    if other:
                        text += ' ' + str(random.Random(0).sample(range(100000, 1000000), 2)[1])
                    return {'choices': [{'text': text}]}
                with patch('sys.argv', argv), patch.object(checker, 'complete', side_effect=answer), contextlib.redirect_stdout(io.StringIO()):
                    return checker.main(), json.loads(out.read_text())
            code, report = run(1)
            self.assertEqual(code, 0)
            self.assertFalse(report['gating'])
            ref.write_text(json.dumps(report))
            code, report = run(2, ref, changed=True)
            self.assertEqual(code, 0)
            self.assertTrue(report['pass'])
            self.assertEqual(report['gate_matches'], 2)
            self.assertEqual(report['accuracy_matches'], 0)
            self.assertEqual(report['reference_comparison']['matches'], 0)
            self.assertEqual(len(report['reference_comparison']['differences']), 2)
            self.assertIn('reference_text', report['reference_comparison']['differences'][0])
            code, report = run(2, ref, other=True)
            self.assertEqual(code, 1)
            self.assertFalse(report['pass'])

    def test_echo_leading_newlines_and_own_number_pass(self):
        self.assertTrue(checker.echo_verdict('\n\n599748', 599748, [599748, 314410])['ok'])
        self.assertFalse(checker.echo_verdict('', 599748, [599748, 314410])['ok'])

    def test_c1_failed_item_excluded_but_cross_number_anywhere_fails(self):
        numbers = list(range(100000, 100020))
        refs = [{'expected': value, 'text': str(value), 'got': value} for value in numbers]
        refs[-1].update(text='', got=None)
        self.assertTrue(checker.echo_validity(refs)['valid'])
        concurrent = [dict(r) for r in refs]
        result = checker.echo_gate(concurrent, {'results': refs})
        self.assertTrue(result['pass'])
        self.assertEqual((result['matches'], result['total']), (19, 19))
        self.assertEqual(result['excluded_items'][0]['reference_text'], '')
        concurrent[-1]['text'] = str(numbers[0])
        result = checker.echo_gate(concurrent, {'results': refs})
        self.assertFalse(result['pass'])
        self.assertEqual(len(result['cross_request_matches']), 1)
        concurrent[-1]['text'] = ''
        concurrent[0]['text'] = ''
        self.assertFalse(checker.echo_gate(concurrent, {'results': refs})['pass'])

    def test_validity_below_95_stops_as_prompt_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out.json'
            preflight = write_preflight(Path(tmp))
            argv = ['check.py', 'echo', '--endpoints', 'http://test', '--n', '20',
                    '--concurrency', '1', '--preflight-report', str(preflight), '--out', str(out)]
            with patch('sys.argv', argv), patch.object(checker, 'complete', return_value={'choices': [{'text': ''}]}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 2)
            report = json.loads(out.read_text())
            self.assertTrue(report['prompt_problem'])
            self.assertFalse(report['gating'])
            argv += ['--reference', str(out)]
            with patch('sys.argv', argv), patch.object(checker, 'complete') as request, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(), 2)
                request.assert_not_called()

    def test_preflight_requires_8_and_reports_all_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'preflight.json'
            argv = ['check.py', 'echo', '--endpoints', 'http://test', '--preflight', '--out', str(out)]
            output = io.StringIO()
            def answer(url, model, prompt, max_tokens, **kwargs):
                value = prompt.splitlines()[0].rsplit(' ', 1)[1]
                return {'choices': [{'text': '' if value == str(random.Random(0).sample(range(100000, 1000000), 8)[0]) else value}]}
            with patch('sys.argv', argv), patch.object(checker, 'complete', side_effect=answer), contextlib.redirect_stdout(output):
                self.assertEqual(checker.main(), 2)
            report = json.loads(out.read_text())
            self.assertEqual(report['n'], 8)
            self.assertEqual(report['concurrency'], 1)
            self.assertEqual(report['ownership_matches'], 7)
            self.assertEqual(report['echo_prompt_version'], 3)
            self.assertIn('PREFLIGHT_ALL_8_STRINGS', output.getvalue())

    def test_old_echo_reference_rejected_without_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, out = Path(tmp)/'old.json', Path(tmp)/'out.json'
            preflight = write_preflight(Path(tmp))
            ref.write_text(json.dumps({'mode': 'echo', 'model': 'qwen', 'n': 8, 'seed': 0,
                                      'concurrency': 1, 'echo_prompt_version': 2}))
            argv = ['check.py', 'echo', '--endpoints', 'http://test', '--n', '8',
                    '--reference', str(ref), '--preflight-report', str(preflight), '--out', str(out)]
            with patch('sys.argv', argv), patch.object(checker, 'complete') as request, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    checker.main()
                request.assert_not_called()

    def test_chat_nothink_request_shape(self):
        number = random.Random(0).sample(range(100000, 1000000), 1)[0]
        response = io.BytesIO(json.dumps({'choices': [{'message': {'role': 'assistant', 'content': str(number)}, 'finish_reason': 'stop'}]}).encode())
        with patch.object(checker.urllib.request, 'urlopen', return_value=response) as opened:
            rows, ok = checker.echo(['http://test'], 'model', 1, 0, 1, 'chat-nothink')
        self.assertTrue(ok)
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, 'http://test/v1/chat/completions')
        self.assertEqual(json.loads(request.data), {'model': 'model',
            'messages': [{'role': 'user', 'content': f'Repeat this number exactly, and nothing else: {number}'}],
            'chat_template_kwargs': {'enable_thinking': False}, 'temperature': 0, 'max_tokens': 16})
        self.assertEqual(rows[0]['text'], str(number))

    def test_completions_long_request_shape(self):
        number = random.Random(0).sample(range(100000, 1000000), 1)[0]
        response = io.BytesIO(json.dumps({'choices': [{'text': str(number)}]}).encode())
        with patch.object(checker.urllib.request, 'urlopen', return_value=response) as opened:
            _, ok = checker.echo(['http://test'], 'model', 1, 0, 1, 'completions-long')
        self.assertTrue(ok)
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, 'http://test/v1/completions')
        self.assertEqual(json.loads(request.data), {'model': 'model',
            'prompt': f'Repeat this number exactly, and nothing else: {number}\nAnswer:',
            'temperature': 0, 'max_tokens': 256})

    def test_thinking_preflight_chat_first_then_long_fallback(self):
        numbers = random.Random(0).sample(range(100000, 1000000), 8)
        bad = [{'expected': n, 'text': '<think>thinking', 'got': None} for n in numbers]
        good = [{'expected': n, 'text': str(n), 'got': n} for n in numbers]
        with patch.object(checker, 'echo', side_effect=[(bad, False), (good, True)]) as requests:
            results, mode, attempts = checker.echo_preflight(['http://test'], 'model', 0,
                smoke={'results': [{'mult': '<think>391'}]})
        self.assertEqual(mode, 'completions-long')
        self.assertEqual([c.args[-1] for c in requests.call_args_list], ['chat-nothink', 'completions-long'])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(results, good)
        with patch.object(checker, 'echo', side_effect=[(bad, False), (bad, False)]):
            _, _, attempts = checker.echo_preflight(['http://test'], 'model', 0, smoke={'text': '<think>'})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sum(len(a['results']) for a in attempts), 16)

    def test_think_in_completions_preflight_selects_chat(self):
        numbers = random.Random(0).sample(range(100000, 1000000), 8)
        bad = [{'expected': n, 'text': '<think>', 'got': None} for n in numbers]
        good = [{'expected': n, 'text': str(n), 'got': n} for n in numbers]
        with patch.object(checker, 'echo', side_effect=[(bad, False), (good, True)]) as requests:
            _, mode, _ = checker.echo_preflight(['http://test'], 'model', 0)
        self.assertEqual(mode, 'chat-nothink')
        self.assertEqual([c.args[-1] for c in requests.call_args_list], ['completions', 'chat-nothink'])

    def test_echo_mode_mismatch_rejected_before_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            preflight = write_preflight(Path(tmp))
            data = json.loads(preflight.read_text())
            data['echo_mode'] = 'chat-nothink'
            preflight.write_text(json.dumps(data))
            ref = Path(tmp)/'ref.json'
            data = dict(data, echo_mode='completions', preflight=False)
            ref.write_text(json.dumps(data))
            argv = ['check.py', 'echo', '--n', '8', '--endpoints', 'http://test',
                    '--preflight-report', str(preflight), '--reference', str(ref), '--out', str(Path(tmp)/'out.json')]
            with patch('sys.argv', argv), patch.object(checker, 'echo') as requests, contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit):
                    checker.main()
                requests.assert_not_called()
                self.assertIn('mode mismatch with C1 reference', output.getvalue())
            argv += ['--echo-mode', 'completions']
            with patch('sys.argv', argv), patch.object(checker, 'echo') as requests, contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit):
                    checker.main()
                requests.assert_not_called()
                self.assertIn('mode mismatch with preflight', output.getvalue())

    def test_sequential_reference_has_one_inflight_request(self):
        active, peak = 0, 0
        lock = threading.Lock()
        def complete(*args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.001)
            with lock:
                active -= 1
            return {'choices': [{'text': '0'}]}
        with patch.object(checker, 'complete', side_effect=complete):
            results, _ = checker.distinct(['http://test'], 'qwen', 16, 0, 1)
        self.assertEqual(peak, 1)
        self.assertEqual(len(results), 16)

    def test_reference_comparison_detects_swapped_answers(self):
        original = [{'a': 11, 'b': 12, 'expected': 132, 'got': 132, 'text': '132', 'numeric_only': True},
                    {'a': 13, 'b': 14, 'expected': 182, 'got': 182, 'text': '182', 'numeric_only': True}]
        concurrent = [dict(r, got=other['got'], text=other['text']) for r, other in zip(original, reversed(original))]
        result = checker.compare_reference(concurrent, {'results': original})
        self.assertEqual(result['matches'], 0)
        self.assertEqual(len(result['cross_request_matches']), 2)

    def test_arithmetic_accuracy_is_non_gating(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out.json'
            argv = ['check.py', 'distinct', '--endpoints', 'http://test', '--n', '2', '--out', str(out)]
            with patch('sys.argv', argv), patch.object(checker, 'complete', return_value={'choices': [{'text': '0'}]}), contextlib.redirect_stdout(io.StringIO()):
                code = checker.main()
            report = json.loads(out.read_text())
            self.assertFalse(report['pass'])
            self.assertFalse(report['gating'])
            self.assertEqual(code, 0)

    def test_unchanged_reference_collision_is_not_misrouting(self):
        original = [{'a': 29, 'b': 24, 'expected': 696, 'got': 704,
                     'text': '704', 'numeric_only': True},
                    {'a': 16, 'b': 44, 'expected': 704, 'got': 704,
                     'text': '704', 'numeric_only': True}]
        result = checker.compare_reference([dict(r) for r in original], {'results': original})
        self.assertEqual(result['matches'], 2)
        self.assertEqual(result['differences'], [])
        self.assertEqual(result['cross_request_matches'], [])


if __name__ == '__main__':
    unittest.main()
