#!/usr/bin/env python3
"""Watch one Linux server process group; never target unrelated processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return (int(fields[2]), int(fields[19]))  # process group, start ticks
    except FileNotFoundError:
        return None


def memory_kib():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemTotal'].split()[0]), int(fields['MemAvailable'].split()[0])


def descendant_groups(pid):
    """Ninja gives jobs new groups; capture only groups in this server tree."""
    records = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z':
                records[int(entry.name)] = (int(fields[1]), int(fields[2]), int(fields[19]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    tree = {pid}
    while True:
        more = {p for p, (parent, _, _) in records.items() if parent in tree}
        if more <= tree:
            break
        tree |= more
    groups = {}
    for p in tree:
        if p in records:
            _, group, start = records[p]
            if group in tree:  # Never signal an inherited external group.
                groups.setdefault(group, []).append((p, start))
    return groups


def kill_group(pid, grace, log):
    # The caller verified PID == PGID. Retain the group through the grace
    # period even if the leader exits, so compiler descendants also receive KILL.
    groups = descendant_groups(pid)
    log(event='captured_groups', groups=groups)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group, members in groups.items():
            # At KILL, retain orphan descendants but reject reused PID/groups.
            if members and not any(process_identity(p) == (group, start) for p, start in members):
                log(event='group_gone', pgid=group)
                continue
            try:
                os.killpg(group, sig)
                log(event='signal', signal=sig.name, pgid=group)
            except ProcessLookupError:
                log(event='group_gone', pgid=group)
        if sig == signal.SIGTERM:
            time.sleep(grace)


def monitor(pid, log, consecutive=3, grace=10, interval=1):
    identity = process_identity(pid)
    # setsid may still be executing when the independent watchdog starts.
    deadline = time.monotonic() + 5
    while identity is not None and identity[0] != pid and time.monotonic() < deadline:
        time.sleep(0.05)
        identity = process_identity(pid)
    if identity is None:
        log(event='server_exited', pid=pid)
        return 0
    if identity[0] != pid or pid == os.getpgrp():
        raise RuntimeError('server does not own an independent process group')
    log(event='watching', pid=pid, pgid=pid, threshold_kib=6*1024**2,
        consecutive=consecutive, interval_s=interval, grace_s=grace)
    low = 0
    while process_identity(pid) == identity:
        try:
            total, available = memory_kib()
        except Exception as exc:
            log(event='WATCHDOG_ERROR', error=repr(exc))
            kill_group(pid, grace, log)
            return 43
        low = low + 1 if available < 6*1024**2 else 0
        log(event='sample', mem_total_kib=total, mem_available_kib=available,
            consecutive_low=low)
        if low >= consecutive:
            log(event='OOM_STOP', pid=pid, mem_available_kib=available)
            kill_group(pid, grace, log)
            return 42
        time.sleep(interval)
    log(event='server_exited', pid=pid)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--consecutive', type=int, choices=[1, 3], default=3,
                        help='1 only for offline build; server always uses 3')
    args = parser.parse_args()
    if args.pid <= 1:
        parser.error('invalid server PID')
    with args.log.open('a', buffering=1) as output:
        def log(**entry):
            output.write(json.dumps({'time': time.time(), **entry})+'\n')
        return monitor(args.pid, log, consecutive=args.consecutive)


if __name__ == '__main__':
    raise SystemExit(main())
