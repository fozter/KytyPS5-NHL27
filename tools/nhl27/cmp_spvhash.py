r"""Equivalence + timing check between two boots (stderr logs with [spvhash]/[shphase]/[shaderdiag]).

Usage: python -I tools\cmp_spvhash.py <A_err.txt> <B_err.txt>
Per (stage, shader) the set of SPIR-V hashes of B must be a subset of A's (a run that sits longer can
reach extra variants). Timings sum the FIRST compile of each shader present in both runs.
"""
import re
import statistics
import sys
from collections import defaultdict

PHASES = ('ir', 'ssa', 'simp1', 'readlane', 'simp2', 'track')


def load(path):
    spv = defaultdict(set)
    first = {}
    diag = {}
    for line in open(path, 'rb').read().decode('utf-8', 'replace').splitlines():
        m = re.match(r'\[spvhash\] (\S+) ([0-9a-f]{16}) ([0-9a-f]{16}) words=(\d+)', line)
        if m:
            spv[(m.group(1), m.group(2))].add(m.group(3))
        elif line.startswith('[shaderdiag]'):
            parts = line.split()
            key = parts[2]
            for field in parts[3:]:
                name, _, value = field.partition('=')
                if value.endswith('ms') and (key, name) not in diag:
                    diag[(key, name)] = float(value[:-2])
        elif line.startswith('[shphase]'):
            shader = line.split()[1]
            if shader not in first:
                first[shader] = {n: float(re.search(r'\b' + n + r'=([\d.]+)ms', line).group(1)) for n in PHASES}
    return spv, first, diag


a_spv, a_first, a_diag = load(sys.argv[1])
b_spv, b_first, b_diag = load(sys.argv[2])
common = set(a_spv) & set(b_spv)
not_subset = [k for k in common if not b_spv[k] <= a_spv[k]]
extra_variants = sum(len(a_spv[k] - b_spv[k]) for k in common)
print(f'common keys={len(common)}; B hashes missing from A: {len(not_subset)} (must be 0); '
      f'A-only extra variants: {extra_variants}')
shaders = set(a_first) & set(b_first)
# Per-shader median ratio over shaders above 2 ms: robust to a few big shaders and to noise.
big = [x for x in shaders if sum(a_first[x].values()) > 2.0]


def median_ratio(phase):
    ratios = []
    for x in big:
        va = sum(a_first[x].values()) if phase == 'total' else a_first[x][phase]
        vb = sum(b_first[x].values()) if phase == 'total' else b_first[x][phase]
        if va > 0.05:
            ratios.append(vb / va)
    return statistics.median(ratios) if ratios else 0.0


print(f'first compile of {len(shaders)} shaders in both (ms)     A        B     B/A  median({len(big)} >2ms)')
ta = tb = 0.0
for n in PHASES:
    sa = sum(a_first[s][n] for s in shaders)
    sb = sum(b_first[s][n] for s in shaders)
    ta, tb = ta + sa, tb + sb
    print(f'  {n:9s} {sa:8.0f} {sb:8.0f}  {sb / sa:5.2f}  {median_ratio(n):5.2f}')
print(f'  {"total":9s} {ta:8.0f} {tb:8.0f}  {tb / ta:5.2f}  {median_ratio("total"):5.2f}')
for name in ('prep', 'translate', 'plan', 'compile_program', 'spv_module'):
    sa = sum(a_diag.get((x, name), 0.0) for x in shaders)
    sb = sum(b_diag.get((x, name), 0.0) for x in shaders)
    print(f'  [shaderdiag] {name:16s} {sa:8.0f} {sb:8.0f}  {sb / sa if sa else 0:5.2f}')
