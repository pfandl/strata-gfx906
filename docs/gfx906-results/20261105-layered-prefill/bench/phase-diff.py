#!/usr/bin/env python3
"""Side-by-side median phase table for two arms (all 20 phases, sorted by the delta)."""
import re, statistics, sys
CH = re.compile(r'prefill timing: (\d+) tokens, GPU timeline (\d+) ms.*?: (.*)$')
PH = re.compile(r'([a-z+ /]+?) (\d+) \(([\d.]+)%\)')
ACC = re.compile(r'prefill timing: (\d+) tokens, GPU timeline (\d+) ms')

def table(arm):
    txt = open(f'/home/me/Documents/strata/data/rp-layer/{arm}/stderr.log', errors='replace').read()
    rows = []
    for line in txt.splitlines():
        m = CH.search(line)
        if not m:
            continue
        if int(m.group(1)) > 4096:      # the cumulative summary row, not a chunk
            continue
        rows.append((int(m.group(2)), {k.strip(): int(v) for k, v, _ in PH.findall(m.group(3))}))
    rows = rows[1:] if len(rows) > 1 else rows
    med = statistics.median(r[0] for r in rows)
    rows = [r for r in rows if r[0] <= 1.5 * med]
    phases = set()
    for _, p in rows:
        phases |= set(p)
    out = {k: statistics.median(r[1].get(k, 0) for r in rows) for k in phases}
    return statistics.median(r[0] for r in rows), out, len(rows)

a, b = sys.argv[1], sys.argv[2]
ta, pa, na = table(a)
tb, pb, nna = table(b)
print(f'{a} ({na} chunks, {ta:.0f} ms)   ->   {b} ({nna} chunks, {tb:.0f} ms)')
print(f'{"phase":<16} {a:>8} {b:>8} {"delta":>8} {"share":>8}')
for k in sorted(set(pa) | set(pb), key=lambda k: -(pb.get(k, 0) - pa.get(k, 0))):
    d = pb.get(k, 0) - pa.get(k, 0)
    print(f'{k:<16} {pa.get(k,0):>8} {pb.get(k,0):>8} {d:>+8} {100.0*d/ta:>+7.1f}%')
print(f'{"GPU timeline":<16} {ta:>8.0f} {tb:>8.0f} {tb-ta:>+8.0f} {100.0*(tb-ta)/ta:>+7.1f}%')
