#!/usr/bin/env python3
"""`qsa attn` per full-size chunk for the int8-scores A/B, and the pooled reading.

  qsa-phases.py ARM_DIR [ARM_DIR ...]        # prints one line per arm, then the pooled base/int8 medians

The phase numbers come from the engine's `prefill timing:` line (`STRATA_PREFILL_TIMING=1`).  The first chunk
of an arm is dropped (it carries the selection's cold read) and so is any chunk whose GPU timeline exceeds
1.5x the arm's median - the same rule as 20261105-layered-prefill/bench/parse-phases.py, which this box needs
because a single chunk can lose a third of a second to a host-copy outlier without the kernel changing.
Arms are pooled by the first letter of their name's flag column in the caller's ARMS string: `base*` and `i8*`
here, which is why the pooled line is hard-coded to those two prefixes.
"""
import os
import re
import statistics
import sys

CH = re.compile(r"prefill timing: (\d+) tokens, GPU timeline (\d+) ms.*?: (.*)$")
PH = re.compile(r"([a-z+ /]+?) (\d+) \(([\d.]+)%\)")


def arm(d):
    rows = []
    for line in open(os.path.join(d, "stderr.log"), errors="replace"):
        m = CH.search(line)
        if not m or m.group(1) != "2048":
            continue
        ph = {n.strip(): int(v) for n, v, _ in PH.findall(m.group(3))}
        rows.append((int(m.group(2)), ph["qsa attn"]))
    rows = rows[1:]
    med = statistics.median(g for g, _ in rows)
    return [(g, q) for g, q in rows if g <= 1.5 * med]


def main(dirs):
    kept = {}
    print(f"{'arm':8s} {'kept':>5s} {'gpu med':>8s} {'qsa med':>8s}  qsa, every kept chunk")
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        kept[name] = arm(d)
        g = statistics.median(x for x, _ in kept[name])
        q = statistics.median(y for _, y in kept[name])
        print(f"{name:8s} {len(kept[name]):5d} {g:8.0f} {q:8.1f}  {sorted(y for _, y in kept[name])}")
    base = [q for n, rows in kept.items() if n.startswith("base") for _, q in rows]
    i8 = [q for n, rows in kept.items() if n.startswith("i8") for _, q in rows]
    gb = [g for n, rows in kept.items() if n.startswith("base") for g, _ in rows]
    gi = [g for n, rows in kept.items() if n.startswith("i8") for g, _ in rows]
    if base and i8:
        mb, mi = statistics.median(base), statistics.median(i8)
        cb, ci = statistics.median(gb), statistics.median(gi)
        print(f"\nqsa attn   base {mb:.1f} (n={len(base)}) -> int8 {mi:.1f} (n={len(i8)}): "
              f"{mi - mb:+.1f} ms = {(mi / mb - 1) * 100:+.1f} %")
        print(f"chunk      base {cb:.0f} (n={len(gb)}) -> int8 {ci:.0f} (n={len(gi)}): "
              f"{ci - cb:+.0f} ms = {(ci / cb - 1) * 100:+.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
