r"""GPU readback syncs inside a time window, from the [synctime] lines of a run's stderr.

Usage: python -I tools\synctime_window.py <err.txt> <from HH:MM:SS> <to HH:MM:SS>
[synctime] <unix ms> reason=syncs/ms ... lines are cumulative (every ~2 s, written by
bufferCache.cpp); the window delta is last-line-before-`to` minus last-line-before-`from`.
Reason names: gpu:fault-read@opXX = the GPU thread faulted on GPU-written memory while running
PM4 opcode XX (op10YY = NOP custom command YY, e.g. op1017 R_FLIP, op1018 R_RELEASE_MEM).
"""
import datetime
import sys

path, start_text, stop_text = sys.argv[1:4]
rows = []
for line in open(path, encoding='utf-8', errors='replace'):
    if not line.startswith('[synctime] '):
        continue
    parts = line.split()
    stamp = int(parts[1])
    totals = {}
    for field in parts[2:]:
        name, _, value = field.rpartition('=')
        syncs, _, ms = value.partition('/')
        totals[name] = (int(syncs), int(ms))
    rows.append((stamp, totals))
if not rows:
    raise SystemExit('no [synctime] lines')

day = datetime.datetime.fromtimestamp(rows[0][0] / 1000).date()


def stamp_of(text):
    clock = datetime.datetime.strptime(text, '%H:%M:%S').time()
    return int(datetime.datetime.combine(day, clock).timestamp() * 1000)


def at(stamp):
    """Totals of the last line at or before stamp (empty before the first line)."""
    best = {}
    for row_stamp, totals in rows:
        if row_stamp > stamp:
            break
        best = totals
    return best


low, high = at(stamp_of(start_text)), at(stamp_of(stop_text))
seconds = (stamp_of(stop_text) - stamp_of(start_text)) / 1000
delta = []
for name, (syncs, ms) in high.items():
    before = low.get(name, (0, 0))
    if syncs - before[0] or ms - before[1]:
        delta.append((ms - before[1], syncs - before[0], name))
delta.sort(reverse=True)
gpu_ms = sum(ms for ms, _, name in delta if name.startswith('gpu:'))
print(f'{start_text}-{stop_text} ({seconds:.0f} s): GPU-thread sync wait {gpu_ms / 1000:.1f} s')
for ms, syncs, name in delta:
    print(f'  {name:28s} {syncs:7d} syncs {ms / 1000:7.2f} s')
