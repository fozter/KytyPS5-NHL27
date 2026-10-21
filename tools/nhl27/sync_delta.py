"""Diff two [syncreason] reports of one run: per-reason syncs/ms between them.
Usage: python -I sync_delta.py <err.txt> [blocks_back=6]
Uses the last report and the one blocks_back reports earlier."""
import re
import sys

path = sys.argv[1]
back = int(sys.argv[2]) if len(sys.argv) > 2 else 6
blocks = []
current = {}
timeline = []
for raw in open(path, 'rb'):
    line = raw.decode('utf-8', 'replace').rstrip()
    m = re.match(r'\[syncreason\] (\S+)\s+(\d+) syncs\s+(\d+) ms', line)
    if m:
        current[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        continue
    if current and not line.startswith('[syncreason]'):
        blocks.append((dict(current), len(timeline)))
        current = {}
    t = re.match(r'\[gputimeline\] (\d+) ms wall: (\d+) submits.*GPU busy (\d+) ms', line)
    if t:
        timeline.append((int(t.group(1)), int(t.group(3))))
if current:
    blocks.append((dict(current), len(timeline)))
if len(blocks) <= back:
    back = len(blocks) - 1
(a, ta), (b, tb) = blocks[-1 - back], blocks[-1]
wall = sum(w for w, _ in timeline[ta:tb])
busy = sum(g for _, g in timeline[ta:tb])
print(f'window: {wall / 1000:.1f} s wall (from gputimeline), GPU busy {busy / 1000:.1f} s')
rows = []
for name, (n, ms) in b.items():
    n0, ms0 = a.get(name, (0, 0))
    if ms - ms0 > 0:
        rows.append((ms - ms0, n - n0, name))
for ms, n, name in sorted(rows, reverse=True)[:12]:
    share = f'{100 * ms / wall:5.1f}% of wall' if wall else ''
    print(f'{name:36s} {n:7d} syncs {ms:8d} ms  {share}')
