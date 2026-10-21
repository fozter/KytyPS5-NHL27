"""Profile from a KYTY_DIAG_SAMPLER=1/2/3 capture (_Build\\windows\\_diag_sampler.txt).

Usage: python -I tools\\symbolize_samples.py <samples.txt> [--under TEXT] [--callers TEXT] [--top N]
  --under TEXT    only samples whose stack has a function containing TEXT
  --callers TEXT  who calls functions containing TEXT (first exe frame matching, its caller)
  --from/--to T   v3+ captures only: keep samples in [from, to); T = HH:MM:SS local time on the
                  capture's day, or unix milliseconds
  --buckets SEC   v3+ only: per time bucket, share of samples whose stack has each --track text
  --track A,B,..  substrings for --buckets (default: compile, SRT walk, waits)
  --lines TEXT    samples under a function containing TEXT, charged to its source line that leads
                  there (its own code, or the call/inlined call it is in); needs line tables
  --threads       v4 (all threads, KYTY_DIAG_SAMPLER=3) only: per-thread summary - CPU time, where
                  the samples are (guest code / emulator / waits), what it waits in, hot guest PCs
  --thread T      v4 only: keep one thread (thread id, or a substring of its name)
  --timeline SEC  v4 only: from --from, SEC seconds as one character per sample per thread (rows:
                  --rows names, or a root function such as GuestGpu::ThreadRun). Legend printed.
Captures must be symbolized with the binary that ran (v2+ captures record its PE timestamp; archive
kyty_emulator.exe + .pdb next to a capture you want to keep). Self = innermost function (inlined frames resolved); "exe self" charges DLL time (malloc, memmove)
to the first kyty_emulator.exe frame. Inclusive = share of samples with the function on the stack.
"""
import argparse
import bisect
import collections
import datetime
import json
import os
import struct
import subprocess

SYMBOLIZER = (r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin'
              r'\llvm-symbolizer.exe')
DEFAULT_EXE = r'D:\PS5_GAMES\NHL27\KytyPS5\_Build\windows\kyty_emulator.exe'
STRIP = ('Libs::Graphics::ShaderRecompiler::', 'Libs::Graphics::', 'Libs::',
         '`anonymous namespace\'::', '(anonymous namespace)::')
WAIT_DLLS = ('ntdll.dll', 'kernelbase.dll', 'kernel32.dll', 'win32u.dll')


def pe_header(exe):
    """(ImageBase, TimeDateStamp) of a PE32+ file."""
    with open(exe, 'rb') as f:
        data = f.read(4096)
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    return struct.unpack_from('<Q', data, pe + 24 + 24)[0], struct.unpack_from('<I', data, pe + 8)[0]


def short(name):
    for prefix in STRIP:
        name = name.replace(prefix, '')
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples')
    parser.add_argument('--exe', default=DEFAULT_EXE)
    parser.add_argument('--under')
    parser.add_argument('--callers')
    parser.add_argument('--lines')
    parser.add_argument('--from', dest='start')
    parser.add_argument('--to', dest='stop')
    parser.add_argument('--buckets', type=float)
    parser.add_argument('--track', default='ProgramCache::Get,MaterializeResources,Wait')
    parser.add_argument('--top', type=int, default=40)
    parser.add_argument('--threads', action='store_true')
    parser.add_argument('--thread')
    parser.add_argument('--timeline', type=float)
    parser.add_argument('--ignore-stamp', action='store_true')
    parser.add_argument('--rows', default='GuestGpu::ThreadRun,RenderLoop,GameSimulationLoop,'
                                          'GfxEopEventLoop,Job0,Job1,Job2,Job3,Job4,Job5')
    args = parser.parse_args()

    modules = {}
    stamps = {}
    samples = []  # tuples of addresses; repeated stacks share one tuple
    times = []
    tids = []
    thread_names = {}
    cpu = collections.defaultdict(list)  # tid -> [(unix ms, kernel+user 100 ns)]
    last_stack = {}
    version = 1
    period_ms = 0.5
    with open(args.samples, encoding='utf-8', errors='replace') as f:
        for line in f:
            kind = line[:2]
            if kind == 'r ':
                _, stamp, tid = line.split()
                tid = int(tid)
                times.append(int(stamp))
                tids.append(tid)
                samples.append(last_stack[tid])
            elif kind == 'u ':
                parts = line.split()
                tid = int(parts[2])
                stack = tuple(int(x, 16) for x in parts[4:])
                last_stack[tid] = stack
                times.append(int(parts[1]))
                tids.append(tid)
                samples.append(stack)
            elif kind == 'c ':
                _, stamp, tid, kernel, user = line.split()
                cpu[int(tid)].append((int(stamp), int(kernel) + int(user)))
            elif kind == 't ':
                _, tid, name = (line.rstrip('\n').split(' ', 2) + [''])[:3]
                thread_names[int(tid)] = name
            elif kind == 'm ':
                _, base, size, rest = line.rstrip('\n').split(' ', 3)
                if rest.startswith('t') and ' ' in rest:  # v2: PE timestamp before the path
                    stamp, rest = rest.split(' ', 1)
                    stamps[os.path.basename(rest).lower()] = int(stamp[1:], 16)
                modules[(int(base, 16), int(size, 16))] = rest
            elif line.startswith('# kyty diag sampler v3'):
                version = 3
                period_ms = 2.0 if 'mode 2' in line else 0.5
            elif line.startswith('# kyty diag sampler v4'):
                version = 4
                period_ms = 10.0
            elif kind == 's ':
                parts = line.split()
                tids.append(0)
                if version >= 3:  # s <unix ms> <frames> <addresses>
                    times.append(int(parts[1]))
                    samples.append(tuple(int(x, 16) for x in parts[3:]))
                else:
                    samples.append(tuple(int(x, 16) for x in parts[2:]))

    def to_ms(text):
        if text is None:
            return None
        if text.isdigit():
            return int(text)
        day = datetime.datetime.fromtimestamp(times[0] / 1000).date()
        clock = datetime.datetime.strptime(text, '%H:%M:%S').time()
        return int(datetime.datetime.combine(day, clock).timestamp() * 1000)

    low, high = 0, 1 << 62
    if args.start or args.stop:
        if not times:
            raise SystemExit('--from/--to need a v3 capture (timestamps)')
        low, high = to_ms(args.start) or 0, to_ms(args.stop) or 1 << 62
    keep_tid = None
    if args.thread:
        if version < 4:
            raise SystemExit('--thread needs a v4 capture (KYTY_DIAG_SAMPLER=3)')
        if args.thread.isdigit():
            keep_tid = {int(args.thread)}
        else:
            keep_tid = {tid for tid, name in thread_names.items() if args.thread in name}
    if low or high != 1 << 62 or keep_tid is not None:
        kept = [(t, s, tid) for t, s, tid in zip(times, samples, tids)
                if low <= t < high and (keep_tid is None or tid in keep_tid)]
        times = [t for t, _, _ in kept]
        samples = [s for _, s, _ in kept]
        tids = [tid for _, _, tid in kept]
    if times:
        first = datetime.datetime.fromtimestamp(times[0] / 1000).strftime('%H:%M:%S')
        last = datetime.datetime.fromtimestamp(times[-1] / 1000).strftime('%H:%M:%S')
        print(f'samples from {first} to {last}')
    ranges = sorted(modules)
    bases = [base for base, _ in ranges]

    def module_of(address):
        index = bisect.bisect_right(bases, address) - 1
        if index >= 0 and address < ranges[index][0] + ranges[index][1]:
            return ranges[index][0], modules[ranges[index]]
        return None, None

    preferred, stamp = pe_header(args.exe)
    exe_name = os.path.basename(args.exe).lower()
    if exe_name not in stamps:
        print('WARNING: v1 capture (no PE timestamps): make sure --exe is the binary that ran')
    elif stamps[exe_name] != stamp and args.ignore_stamp:
        print(f'WARNING: stamp {stamp:08x} != sampled {stamps[exe_name]:08x}, ignored (--ignore-stamp: '
              'only for a relink of the same sources)')
    elif stamps[exe_name] != stamp:
        raise SystemExit(f'{args.exe} (stamp {stamp:08x}) is not the binary that was sampled '
                         f'(stamp {stamps[exe_name]:08x}); pass --exe <archived copy>')
    # Per unique stack: list of (is_exe, key); key = preferred VA for exe frames, else DLL name
    # ('<guest>' for addresses outside every module: guest code).
    unique = collections.Counter(samples)
    frames_of = {}
    wanted = set()
    for sample in unique:
        frames = []
        for depth, address in enumerate(sample):
            base, path = module_of(address)
            if path is None:
                frames.append((False, '<guest>'))
            elif os.path.basename(path).lower() == exe_name:
                va = preferred + address - base - (1 if depth else 0)
                wanted.add(va)
                frames.append((True, va))
            else:
                frames.append((False, os.path.basename(path)))
        frames_of[sample] = frames

    names = {}
    lines = {}
    ordered = sorted(wanted)
    if ordered:
        result = subprocess.run([SYMBOLIZER, '--obj=' + args.exe, '--output-style=JSON'],
                                input='\n'.join('0x%x' % va for va in ordered), capture_output=True,
                                text=True, check=True)
        for va, line in zip(ordered, result.stdout.splitlines()):
            symbols = json.loads(line).get('Symbol', [])
            chain = [short(s.get('FunctionName') or '??') for s in symbols] or ['0x%x' % va]
            names[va] = chain  # innermost first
            # Inline chain innermost first: (function, file:line) of each level.
            lines[va] = [(short(sym.get('FunctionName') or '??'),
                          '%s:%s' % (os.path.basename(sym.get('FileName', '?')), sym.get('Line', '?')))
                         for sym in symbols]

    def expand(frames):
        """All function names on the stack, innermost first (inlined frames included)."""
        out = []
        for is_exe, key in frames:
            out.extend(names.get(key, ['0x%x' % key]) if is_exe else [key])
        return out

    stack_of = {sample: expand(frames) for sample, frames in frames_of.items()}
    under_ok = {sample: not args.under or any(args.under in name for name in stack)
                for sample, stack in stack_of.items()}

    if args.threads:
        thread_summary(args, times, samples, tids, thread_names, cpu, frames_of, stack_of, names,
                       period_ms, low, high)
        return
    if args.timeline:
        timeline(args, times, samples, tids, thread_names, frames_of, stack_of, period_ms)
        return

    if args.buckets and times:
        tracked = [t for t in args.track.split(',') if t]
        has = {sample: [any(text in name for name in stack) for text in tracked]
               for sample, stack in stack_of.items()}
        width = args.buckets * 1000
        buckets = collections.OrderedDict()
        for stamp, sample in zip(times, samples):
            if not under_ok[sample]:
                continue
            row = buckets.setdefault(int(stamp // width), [0] + [0] * len(tracked))
            row[0] += 1
            for column, hit in enumerate(has[sample]):
                row[column + 1] += hit
        print('\nTIMELINE (share of samples per bucket)')
        print('  time      samples  ' + '  '.join(f'{t[:18]:>18s}' for t in tracked))
        for key, row in buckets.items():
            clock = datetime.datetime.fromtimestamp(key * width / 1000).strftime('%H:%M:%S')
            print(f'  {clock}  {row[0]:7d}  ' + '  '.join(f'{100.0 * c / row[0]:17.0f}%' for c in row[1:]))
    weighted = [(sample, count) for sample, count in unique.items() if under_ok[sample]]
    total = sum(count for _, count in weighted)
    print(f'{total} samples' + (f' under "{args.under}"' if args.under else '') +
          f' (of {len(samples)}; ~{total * period_ms / 1000:.1f} s at {period_ms} ms)')
    if not total:
        return

    def report(title, counter):
        print(f'\n{title}')
        for name, count in counter.most_common(args.top):
            print(f'  {100.0 * count / total:5.1f}%  {count:7d}  {name[:150]}')

    self_count = collections.Counter()
    exe_self = collections.Counter()
    inclusive = collections.Counter()
    callers = collections.Counter()
    by_line = collections.Counter()
    for sample, weight in weighted:
        frames, stack = frames_of[sample], stack_of[sample]
        self_count[stack[0]] += weight
        first_exe = next((names.get(key, ['?'])[0] for is_exe, key in frames if is_exe), '<no exe frame>')
        dll = '' if frames[0][0] else f' [in {frames[0][1]}]'
        exe_self[first_exe + dll] += weight
        for name in set(stack):
            inclusive[name] += weight
        if args.lines:
            # Innermost level (leaf side first) that belongs to the function.
            hit = next((where for is_exe, key in frames if is_exe
                        for function, where in lines.get(key, []) if args.lines in function), None)
            if hit is not None:
                by_line[hit] += weight
        if args.callers:
            for index, name in enumerate(stack):
                if args.callers in name:
                    callers[f'{name}  <-  {stack[index + 1] if index + 1 < len(stack) else "?"}'] += weight
                    break
    report('SELF (innermost function)', self_count)
    report('EXE SELF (DLL time charged to the first exe frame)', exe_self)
    report('INCLUSIVE', inclusive)
    if args.lines:
        report(f'LINES of "{args.lines}" (inclusive, by its line)', by_line)
    if args.callers:
        report(f'CALLERS of "{args.callers}"', callers)


def thread_summary(args, times, samples, tids, thread_names, cpu, frames_of, stack_of, names,
                   period_ms, low, high):
    """Per thread: CPU share of the window, where its samples are, what it waits in."""
    if not times:
        print('no samples')
        return
    start, stop = max(low, times[0]), min(high, times[-1])
    seconds = max((stop - start) / 1000.0, 1e-3)

    def root_of(stack):
        """Outermost emulator function that is not thread-start boilerplate (for a guest thread:
        the HLE function the guest called)."""
        exe = [name for name in stack if not name.lower().endswith('.dll') and name != '<guest>'
               and not name.startswith('std::') and name != '<unknown>']
        return exe[-1] if exe else ('<guest>' if '<guest>' in stack else stack[-1])

    def classify(sample):
        frames, stack = frames_of[sample], stack_of[sample]
        first_exe = next((names.get(key, ['?'])[0] for is_exe, key in frames if is_exe), None)
        leaf_exe, leaf = frames[0]
        if leaf == '<guest>':
            return 'guest', None
        if leaf_exe:
            return 'emulator', first_exe
        if leaf.lower() in WAIT_DLLS:
            # The emulator function nearest the OS frames, and the outermost one (HLE entry).
            root = root_of(stack)
            where = first_exe or '<no exe frame>'
            return 'os', where if root == where else f'{where} <- {root}'
        return 'dll ' + leaf, first_exe

    kinds = {sample: classify(sample) for sample in frames_of}
    per = collections.defaultdict(lambda: {'n': 0, 'kinds': collections.Counter(),
                                           'os': collections.Counter(), 'emu': collections.Counter(),
                                           'guest': collections.Counter(), 'root': collections.Counter()})
    for tid, sample in zip(tids, samples):
        row = per[tid]
        row['n'] += 1
        kind, where = kinds[sample]
        row['kinds'][kind if not kind.startswith('dll ') else 'dll'] += 1
        if kind == 'os':
            row['os'][where] += 1
        elif kind == 'emulator':
            row['emu'][where] += 1
        elif kind == 'guest':
            row['guest'][sample[0] & ~0xf] += 1
        row['root'][root_of(stack_of[sample])] += 1

    def cpu_seconds(tid):
        rows = [r for r in cpu.get(tid, []) if start <= r[0] <= stop]
        return (rows[-1][1] - rows[0][1]) / 1e7 if len(rows) >= 2 else 0.0

    order = sorted(per, key=lambda tid: (-cpu_seconds(tid), -per[tid]['n']))
    print(f'\nTHREADS ({seconds:.0f} s window; CPU = share of one core; samples every {period_ms} ms)')
    for tid in order[:args.top]:
        row = per[tid]
        n = row['n']
        share = lambda key: 100.0 * row['kinds'][key] / n
        print(f'\n  tid {tid} "{thread_names.get(tid, "?")}" root {row["root"].most_common(1)[0][0][:70]}')
        print(f'    CPU {100.0 * cpu_seconds(tid) / seconds:5.1f}% | samples {n}: guest code {share("guest"):.0f}%, '
              f'emulator {share("emulator"):.0f}%, OS/wait {share("os"):.0f}%, other DLL {share("dll"):.0f}%')
        for label, key in (('in OS (by emulator caller)', 'os'), ('in emulator', 'emu')):
            top = row[key].most_common(4)
            if top:
                print(f'    {label}: ' + '; '.join(f'{100.0 * c / n:.0f}% {name[:90]}' for name, c in top))
        top = row['guest'].most_common(4)
        if top:
            print('    hot guest PCs: ' + ', '.join(f'0x{pc:x} {100.0 * c / n:.0f}%' for pc, c in top))


TIMELINE_LEGEND = ('# guest code   e emulator   v other DLL (driver)   . waiting (semaphore/cond/'
                   'equeue/sleep)   a audio lock (AudioOut2)   o other OS call\n'
                   'GPU thread: _ idle (no submission)   W waiting for the GPU (MasterSemaphore)   '
                   'S SRT walk / MaterializeResources   C shader compile   F fault handling')


def timeline(args, times, samples, tids, thread_names, frames_of, stack_of, period_ms):
    start = times[0]
    stop = start + int(args.timeline * 1000)
    wanted = [r for r in args.rows.split(',') if r]
    # Resolve each row to a tid: by thread name first, else by a function in its stacks.
    roots = collections.defaultdict(collections.Counter)
    for tid, sample in zip(tids, samples):
        for name in stack_of[sample]:
            roots[tid][name] += 1
    rows = []
    for row in wanted:
        tid = next((t for t, name in thread_names.items() if name == row and t in roots), None)
        if tid is None:
            hits = {t: sum(c for name, c in counter.items() if row in name) for t, counter in roots.items()}
            tid = max((t for t in hits if hits[t]), key=lambda t: hits[t], default=None)
        if tid is not None:
            rows.append((row, tid))

    def char(sample):
        frames, stack = frames_of[sample], stack_of[sample]
        joined = ' | '.join(stack)
        if 'GuestGpu::ThreadRun' in joined:
            if 'MasterSemaphore::Wait' in joined:
                return 'W'
            if 'MaterializeResources' in joined:
                return 'S'
            if 'ProgramCache::Get' in joined or 'CompileProgram' in joined:
                return 'C'
            if 'HandleFault' in joined:
                return 'F'
            if stack[0].lower() in WAIT_DLLS and 'CondVar::Wait' in joined and 'GuestGpu::Process' not in joined:
                return '_'
        leaf_exe, leaf = frames[0]
        if leaf == '<guest>':
            return '#'
        if leaf_exe:
            return 'e'
        if leaf.lower() in WAIT_DLLS:
            if 'AudioOut2' in joined:
                return 'a'
            if 'wait' in joined.lower() or 'sleep' in joined.lower():
                return '.'
            return 'o'
        return 'v'

    ticks = int((stop - start) / period_ms)
    grid = {tid: [' '] * ticks for _, tid in rows}
    for stamp, tid, sample in zip(times, tids, samples):
        if tid in grid and start <= stamp < stop:
            index = int((stamp - start) / period_ms)
            if 0 <= index < ticks:
                grid[tid][index] = char(sample)
    print(TIMELINE_LEGEND)
    per_line = int(1000 / period_ms)
    for offset in range(0, ticks, per_line):
        clock = datetime.datetime.fromtimestamp((start + offset * period_ms) / 1000)
        print(f'\n{clock.strftime("%H:%M:%S")}.{clock.microsecond // 1000:03d}  '
              + ''.join('|' if i % 10 == 0 else ' ' for i in range(min(per_line, ticks - offset))))
        for row, tid in rows:
            print(f'  {row[:18]:>18s}  ' + ''.join(grid[tid][offset:offset + per_line]))


if __name__ == '__main__':
    main()
