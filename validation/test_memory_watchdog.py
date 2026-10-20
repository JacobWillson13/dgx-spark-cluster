import importlib.util
import signal
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('watchdog', Path(__file__).parent/'bin/memory_watchdog.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class WatchdogTests(unittest.TestCase):
    def run_monitor(self, available, identities=None, consecutive=3):
        events = []
        identity = (12345, 999)
        identities = identities or [identity]*(len(available)+1)+[None]
        with patch.object(watchdog, 'process_identity', side_effect=identities), \
                patch.object(watchdog, 'memory_kib', side_effect=[(128*1024**2, a) for a in available]), \
                patch.object(watchdog.time, 'sleep') as sleep, \
                patch.object(watchdog, 'descendant_groups', return_value={12345: []}), \
                patch.object(watchdog.os, 'killpg') as kill:
            rc = watchdog.monitor(12345, lambda **e: events.append(e), consecutive=consecutive)
        return rc, events, kill.call_args_list, sleep.call_args_list

    def test_three_consecutive_lows_kill_group_term_then_kill(self):
        rc, events, signals, sleeps = self.run_monitor([5*1024**2]*3)
        self.assertEqual(rc, 42)
        self.assertEqual([c.args for c in signals], [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])
        self.assertEqual([c.args for c in sleeps], [(1,), (1,), (10,)])
        self.assertTrue(any(e['event'] == 'OOM_STOP' for e in events))

    def test_exact_threshold_resets_low_counter(self):
        rc, events, signals, _ = self.run_monitor([5*1024**2, 5*1024**2, 6*1024**2, 5*1024**2])
        self.assertEqual(rc, 0)
        self.assertFalse(signals)
        self.assertEqual([e['consecutive_low'] for e in events if e['event'] == 'sample'], [1, 2, 0, 1])

    def test_server_exit_or_pid_reuse_stops_watchdog(self):
        for final in (None, (12345, 1000)):
            rc, _, signals, _ = self.run_monitor([9*1024**2], [(12345, 999), (12345, 999), final])
            self.assertEqual(rc, 0)
            self.assertFalse(signals)

    def test_offline_build_aborts_on_first_low_sample(self):
        rc, _, signals, _ = self.run_monitor([5*1024**2], consecutive=1)
        self.assertEqual(rc, 42)
        self.assertEqual(len(signals), 2)

    def test_leader_exit_during_grace_does_not_skip_group_kill(self):
        events = []
        with patch.object(watchdog.os, 'killpg') as kill, patch.object(watchdog.time, 'sleep'), \
                patch.object(watchdog, 'descendant_groups', return_value={12345: []}):
            watchdog.kill_group(12345, 10, lambda **e: events.append(e))
        self.assertEqual([c.args for c in kill.call_args_list], [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_does_not_signal_reused_descendant_pid_or_group(self):
        with patch.object(watchdog, 'descendant_groups', return_value={12345: [(12345, 999)]}), \
                patch.object(watchdog, 'process_identity', return_value=(12345, 1000)), \
                patch.object(watchdog.time, 'sleep'), patch.object(watchdog.os, 'killpg') as kill:
            watchdog.kill_group(12345, 10, lambda **e: None)
        kill.assert_not_called()

    def test_refuses_unrelated_process_group(self):
        with patch.object(watchdog, 'process_identity', return_value=(54321, 999)), \
                patch.object(watchdog.time, 'monotonic', side_effect=[0, 6]), \
                self.assertRaisesRegex(RuntimeError, 'independent process group'):
            watchdog.monitor(12345, lambda **e: None)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux process-group integration')
    def test_real_setsid_group_kills_descendant_after_leader_exits(self):
        child_code = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'
        parent_code = ('import subprocess,sys,time; '
                       f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}],stdout=subprocess.PIPE,text=True); '
                       'p.stdout.readline(); print(p.pid,flush=True); time.sleep(60)')
        parent = subprocess.Popen(['setsid', sys.executable, '-c', parent_code], stdout=subprocess.PIPE, text=True)
        try:
            child_pid = int(parent.stdout.readline())
            self.assertEqual(os.getpgid(parent.pid), parent.pid)
            self.assertEqual(os.getpgid(child_pid), parent.pid)
            with patch.object(watchdog, 'memory_kib', return_value=(128*1024**2, 5*1024**2)):
                rc = watchdog.monitor(parent.pid, lambda **e: None, grace=0.1, interval=0.01)
            self.assertEqual(rc, 42)
            parent.wait(timeout=2)
            deadline = time.monotonic()+2
            while watchdog.process_identity(child_pid) is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNone(watchdog.process_identity(child_pid))
        finally:
            try:
                os.killpg(parent.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            parent.wait(timeout=2)
            parent.stdout.close()

    @unittest.skipUnless(sys.platform == 'linux', 'Linux escaped compiler-group integration')
    def test_kills_ninja_style_descendant_in_new_session(self):
        child_code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'
        parent_code = ('import subprocess,sys,time; '
                       f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}],start_new_session=True,stdout=subprocess.PIPE,text=True); '
                       'p.stdout.readline(); print(p.pid,flush=True); time.sleep(60)')
        parent = subprocess.Popen(['setsid', sys.executable, '-c', parent_code], stdout=subprocess.PIPE, text=True)
        child_pid = None
        try:
            child_pid = int(parent.stdout.readline())
            self.assertEqual(os.getpgid(child_pid), child_pid)
            with patch.object(watchdog, 'memory_kib', return_value=(128*1024**2, 5*1024**2)):
                self.assertEqual(watchdog.monitor(parent.pid, lambda **e: None, grace=0.1, interval=0.01), 42)
            parent.wait(timeout=2)
            deadline = time.monotonic()+2
            while watchdog.process_identity(child_pid) is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNone(watchdog.process_identity(child_pid))
        finally:
            for pid in [parent.pid, child_pid]:
                if pid:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            parent.wait(timeout=2)
            parent.stdout.close()


if __name__ == '__main__':
    unittest.main()
