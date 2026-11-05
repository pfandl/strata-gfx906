#!/usr/bin/env python3
"""Summarise the arms of a bench/gfx906/overlap-ab.sh session.

    python3 parse-arms.py <session dir> [<session dir> ...]

Reads every <arm>/stdout.log and <arm>/stderr.log and prints one row per arm: the engine's own
end-to-end line (prompt tok/s, decode tok/s), the decode milliseconds per token, the hand-off line
(so an arm whose flag did not take cannot be read as a null result), and the md5 of the generated
token ids - the same md5 in every arm is the bit-exactness check.
"""
from __future__ import annotations

import hashlib
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(
    r"prompt (\d+) tokens = \d+ reused \+ \d+ read in ([\d.]+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def arm(d: Path) -> dict | None:
    try:
        err = (d / "stderr.log").read_text(errors="replace")
        out = (d / "stdout.log").read_text(errors="replace")
    except OSError:
        return None
    m = None
    for m in LINE.finditer(err):
        pass
    if m is None:
        return None
    ids = [l for l in out.splitlines() if l.startswith("T ")]
    overlap = "overlapped: STRATA_SPLIT_OVERLAP=1" in err
    return {"arm": d.name, "prompt": int(m.group(1)), "pp_ms": float(m.group(2)),
            "pp": float(m.group(3)), "gen": int(m.group(4)), "dec_ms": float(m.group(5)),
            "tg": float(m.group(6)), "drafts": m.group(7) + "/" + m.group(8),
            "ms_per_token": float(m.group(5)) / max(int(m.group(4)), 1),
            "handoff": "overlapped" if overlap else "serial",
            "ids_md5": hashlib.md5("\n".join(ids).encode()).hexdigest()[:12],
            "ids": len(ids)}


def main(argv: list[str]) -> int:
    rows: list[dict] = []
    for sd in argv[1:]:
        for d in sorted(Path(sd).iterdir()):
            if d.is_dir():
                r = arm(d)
                if r:
                    rows.append(r)
    if not rows:
        print("no arms with a result line", file=sys.stderr)
        return 2
    print(f"{'arm':8} {'prompt':>6} {'pp tok/s':>9} {'tg tok/s':>9} {'dec ms':>8} {'ms/tok':>7} "
          f"{'drafts':>7} {'hand-off':>11} {'ids':>4} {'md5':>13}")
    for r in rows:
        print(f"{r['arm']:8} {r['prompt']:6} {r['pp']:9.1f} {r['tg']:9.1f} {r['dec_ms']:8.1f} "
              f"{r['ms_per_token']:7.2f} {r['drafts']:>7} {r['handoff']:>11} {r['ids']:4} "
              f"{r['ids_md5']:>13}")
    for key, label in (("tg", "decode tok/s"), ("ms_per_token", "decode ms/token"), ("pp", "prompt tok/s")):
        groups = {}
        for r in rows:
            groups.setdefault(r["handoff"], []).append(r[key])
        if len(groups) == 2 and all(groups.values()):
            (a, va), (b, vb) = groups.items()
            ma, mb = statistics.median(va), statistics.median(vb)
            print(f"\n{label}: {a} median {ma:.2f} (n={len(va)}), {b} median {mb:.2f} (n={len(vb)}), "
                  f"delta {100.0 * (mb - ma) / ma:+.1f} %")
    if len({r["ids_md5"] for r in rows}) == 1:
        print(f"generated token ids: identical in all {len(rows)} arms ({rows[0]['ids']} tokens)")
    else:
        print("generated token ids: DIFFER between arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
