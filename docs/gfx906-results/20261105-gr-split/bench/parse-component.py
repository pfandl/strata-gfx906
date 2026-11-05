#!/usr/bin/env python3
"""The component half of a STRATA_GR_SPLIT session: fused_gr_bench, flag off against flag on.

    python3 parse-component.py <dir with bench-{off,on}-<n>.log>

`fused_gr_bench` runs the multi-token hyper-connection read twice per shape - `fused_gr_set_fast(0)`
(the older kernels) and `fused_gr_set_fast(1)` (STRATA_GR_FAST) - and prints both times plus a hash
of the first column.  STRATA_GR_SPLIT replaces the down read *inside* both of them, so the pair to
compare across the two modes is `fast` against `fast` (and `old` against `old`); the hash is what
says the sums changed.
"""
from __future__ import annotations

import glob
import os
import re
import statistics
import sys

LINE = re.compile(r"T (\d+) apply (\d) inject (\d) \| old\s+([\d.]+) us\s+fast\s+([\d.]+) us \| (.*?) \| hash (\w+)")


def main(argv: list[str]) -> int:
    d = argv[1] if len(argv) > 1 else "."
    rows: dict[tuple[int, str, str], dict[str, list[tuple[float, float, str]]]] = {}
    hashes: dict[tuple[int, str, str], dict[str, set[str]]] = {}
    for f in sorted(glob.glob(os.path.join(d, "bench-*.log"))):
        mode = "on" if "bench-on" in os.path.basename(f) else "off"
        for line in open(f, errors="replace"):
            m = LINE.match(line)
            if not m:
                continue
            key = (int(m.group(1)), m.group(2), m.group(3))
            rows.setdefault(key, {}).setdefault(mode, []).append((float(m.group(4)), float(m.group(5)), m.group(6)))
            hashes.setdefault(key, {}).setdefault(mode, set()).add(m.group(7))
    if not rows:
        print("no fused_gr_bench lines", file=sys.stderr)
        return 2
    print(f"{'T':>2} {'ap':>2} {'inj':>3} {'off old':>8} {'off fast':>9} {'on fast':>8} {'split':>7} {'hash':>6}")
    for key in sorted(rows):
        off, on = rows[key].get("off"), rows[key].get("on")
        if not off or not on:
            continue
        off_old = statistics.median([x[0] for x in off])
        off_fast = statistics.median([x[1] for x in off])
        on_fast = statistics.median([x[1] for x in on])
        same = hashes[key]["off"] == hashes[key]["on"]
        print(f"{key[0]:2} {key[1]:>2} {key[2]:>3} {off_old:8.1f} {off_fast:9.1f} {on_fast:8.1f} "
              f"{100 * (on_fast - off_fast) / off_fast:+6.1f}% {'same' if same else 'differs':>6}")
    deltas = []
    for key in sorted(rows):
        off, on = rows[key].get("off"), rows[key].get("on")
        if off and on:
            deltas.append(100 * (statistics.median([x[1] for x in on]) - statistics.median([x[1] for x in off])) /
                          statistics.median([x[1] for x in off]))
    if deltas:
        print(f"\nfast read, flag on vs off over {len(deltas)} shapes: median {statistics.median(deltas):+.1f} %, "
              f"range {min(deltas):+.1f} % .. {max(deltas):+.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
