r"""Frame pacing from a KYTY_FRAME_TRACE capture (graphicsRun.cpp FrameTrace).

Usage: python -I tools\frame_trace.py <trace.txt> [--from HH:MM:SS] [--to HH:MM:SS] [--frames N]
A "frame" ends when the game enqueues its suspend point (S); NHL 27 places one every second
displayed frame (run 106: 4.6 suspend points/s at ~9 fps on the title bar). Per frame:
  game wait   - the game blocked at the suspend point (P -> A): waiting for the GPU to complete the
                previous frame's suspend point
  gpu busy    - the GPU thread processing submissions/commands (b..e, x..y) inside the frame
  gpu drain   - inside gpu busy: the GPU thread waiting for the GPU (W..w, readbacks/finishes)
  gpu tail    - from the GPU thread finishing the previous suspend point to the GPU completing it
                (R): real GPU work after the last flush, plus the completion callback latency
--frames N prints the first N frames one per line.
"""
import argparse
import datetime
import statistics

SUSPEND = 3  # SubmissionType::SuspendPoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace')
    parser.add_argument('--from', dest='start')
    parser.add_argument('--to', dest='stop')
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--alternate', type=float, default=0.0,
                        help='period of a KYTY_*_ALTERNATE switch: report even (switch ON) and odd '
                             '(OFF) windows separately; intervals crossing a switch are dropped')
    args = parser.parse_args()

    offset_ms = None
    events = []
    with open(args.trace, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('#'):
                parts = line.split()
                steady, unix_ms = int(parts[5]), int(parts[9])
                offset_ms = unix_ms - steady / 1000.0
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            events.append((int(parts[0]), parts[1], int(parts[2])))

    def to_us(text):
        day = datetime.datetime.fromtimestamp((events[0][0] / 1000.0 + offset_ms) / 1000).date()
        clock = datetime.datetime.strptime(text, '%H:%M:%S').time()
        return int((datetime.datetime.combine(day, clock).timestamp() * 1000 - offset_ms) * 1000)

    low = to_us(args.start) if args.start else 0
    high = to_us(args.stop) if args.stop else 1 << 62
    events = [e for e in events if low <= e[0] < high]
    if not events:
        raise SystemExit('no events in the window')

    # GPU thread busy intervals; suspend point processing ends; suspend completions; game waits.
    busy = []
    drains = []
    drain_at = None
    open_at = None
    suspend_done = []
    pending_type = None
    releases = []
    waits = []
    wait_start = None
    frame_ends = []
    counts = []
    current = {'G': 0, 'C': 0, 'F': 0, 'I': 0}
    for us, kind, arg in events:
        if kind in 'bx':
            open_at = us
            pending_type = arg if kind == 'b' else None
        elif kind in 'ey' and open_at is not None:
            busy.append((open_at, us))
            if kind == 'e' and pending_type == SUSPEND:
                suspend_done.append(us)
            open_at = None
        elif kind == 'W':
            drain_at = us
        elif kind == 'w' and drain_at is not None:
            drains.append((drain_at, us))
            drain_at = None
        elif kind == 'R':
            releases.append(us)
        elif kind == 'P':
            wait_start = us
        elif kind == 'A' and wait_start is not None:
            waits.append((wait_start, us))
            wait_start = None
        elif kind == 'S':
            frame_ends.append(us)
            counts.append(dict(current))
            current = {'G': 0, 'C': 0, 'F': 0, 'I': 0}
        elif kind in current:
            current[kind] += 1

    def overlap(intervals, a, b):
        return sum(max(0, min(e, b) - max(s, a)) for s, e in intervals)

    rows = []
    for index in range(1, len(frame_ends)):
        a, b = frame_ends[index - 1], frame_ends[index]
        frame = b - a
        wait = sum(e - s for s, e in waits if a <= e <= b)  # the wait that ended this frame's S
        gpu = overlap(busy, a, b)
        drain = overlap(drains, a, b)
        done = [t for t in suspend_done if a <= t < b]
        rel = [t for t in releases if a <= t < b + 200000]
        tail = (rel[0] - done[0]) if done and rel and rel[0] >= done[0] else None
        rows.append((a, frame, wait, gpu, tail, counts[index], drain))
    if not rows:
        raise SystemExit('fewer than two suspend points in the window')

    def ms(us):
        return us / 1000.0

    if args.alternate:
        period = int(args.alternate * 1e6)
        groups = {'ON (even windows)': [], 'OFF (odd windows)': []}
        dropped = 0
        for row in rows:
            first, last = row[0] // period, (row[0] + row[1]) // period
            if first != last:
                dropped += 1
                continue
            groups['ON (even windows)' if first % 2 == 0 else 'OFF (odd windows)'].append(row)
        print(f'ALTERNATE {args.alternate} s: {dropped} intervals crossing a switch dropped')
        for name, group in groups.items():
            if not group:
                continue
            frames = [r[1] for r in group]
            print(f'  {name}: {len(group)} intervals, frame median {ms(statistics.median(frames)):.1f} ms, '
                  f'mean {ms(statistics.mean(frames)):.1f} ms | gpu busy mean '
                  f'{ms(statistics.mean(r[3] for r in group)):.1f} | drain '
                  f'{ms(statistics.mean(r[6] for r in group)):.1f} | record '
                  f'{ms(statistics.mean(r[3] - r[6] for r in group)):.1f} | G '
                  f'{statistics.mean(r[5]["G"] for r in group):.1f}')
        # Adjacent windows see nearly the same scene: compare their median intervals pairwise.
        windows = {}
        for row in rows:
            if row[0] // period == (row[0] + row[1]) // period:
                windows.setdefault(row[0] // period, []).append(row[1])
        diffs = []
        for key in sorted(windows):
            if key + 1 in windows:
                on, off = (key, key + 1) if key % 2 == 0 else (key + 1, key)
                diffs.append(ms(statistics.median(windows[off]) - statistics.median(windows[on])))
        if diffs:
            slower = sum(d > 0 for d in diffs)
            print(f'  adjacent window pairs: OFF slower in {slower} of {len(diffs)}, median OFF - ON '
                  f'{statistics.median(diffs):+.1f} ms')
        print()

    print(f'{len(rows)} frames, {ms(rows[-1][0] + rows[-1][1] - rows[0][0]) / 1000:.1f} s')
    frames = [r[1] for r in rows]
    print(f'  frame time   median {ms(statistics.median(frames)):6.1f} ms, mean {ms(statistics.mean(frames)):6.1f} ms '
          f'({1e6 / statistics.mean(frames):.2f} fps)')
    for label, values in (('game wait', [r[2] for r in rows]), ('gpu busy', [r[3] for r in rows]),
                          ('gpu drain', [r[6] for r in rows]),
                          ('gpu record', [r[3] - r[6] for r in rows]),
                          ('gpu tail', [r[4] for r in rows if r[4] is not None])):
        if values:
            print(f'  {label:11s}  median {ms(statistics.median(values)):6.1f} ms, mean {ms(statistics.mean(values)):6.1f} ms '
                  f'({100 * sum(values) / sum(frames):.0f}% of frame time)')
    print('  submissions per frame: ' + ', '.join(
        f'{k} {statistics.mean(r[5][k] for r in rows):.1f}' for k in ('G', 'C', 'F', 'I')))
    if args.frames:
        print('\n  start          frame   wait   gpu busy  drain   tail   G  C  I')
        for a, frame, wait, gpu, tail, count, drain in rows[:args.frames]:
            clock = datetime.datetime.fromtimestamp((a / 1000.0 + offset_ms) / 1000)
            tail_text = f'{ms(tail):6.1f}' if tail is not None else '     ?'
            print(f'  {clock.strftime("%H:%M:%S")}.{clock.microsecond // 1000:03d}  {ms(frame):6.1f} {ms(wait):6.1f}'
                  f'   {ms(gpu):6.1f} {ms(drain):6.1f}  {tail_text}  {count["G"]:2d} {count["C"]:2d} {count["I"]:2d}')


if __name__ == '__main__':
    main()
