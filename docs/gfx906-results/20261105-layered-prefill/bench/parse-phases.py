#!/usr/bin/env python3
"""Per-arm summary of the layered prefill A/B, robust against the per-chunk outliers this box shows.

  parse-phases.py <root-or-arm-dir> ...          # table of arms
  parse-phases.py --groups <root> ...            # median/min/max of pp across repeats of one config

Per arm: end-to-end pp tok/s (the engine's own line), the median GPU ms per full-size chunk
(first chunk dropped, chunks slower than 1.5x the median dropped as outliers), and the median of
the phases that move between layers.
"""
import os
import re
import statistics
import sys

PP = re.compile(r"prompt (\d+) tokens = 0 reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)")
CH = re.compile(r"prefill timing: (\d+) tokens, GPU timeline (\d+) ms.*?: (.*)$")
PH = re.compile(r"([a-z+ /]+?) (\d+) \(([\d.]+)%\)")
PHASES = ("hc read", "qsa attn", "gemm gate/up", "gemm down", "gdn", "dequant")


def arm(d):
    try:
        txt = open(os.path.join(d, "stderr.log"), errors="replace").read()
    except OSError:
        return None
    pp = None
    chunks = []
    for line in txt.splitlines():
        m = PP.search(line)
        if m:
            pp = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
        m = CH.search(line)
        if m:
            phases = {k.strip(): int(v) for k, v, _p in PH.findall(m.group(3))}
            chunks.append((int(m.group(1)), int(m.group(2)), phases))
    if pp is None or not chunks:
        return None
    body = chunks[1:] if len(chunks) > 1 else chunks
    sizes = [c[0] for c in body]
    full = max(set(sizes), key=sizes.count)
    keep = [c for c in body if c[0] == full]
    med = statistics.median(c[1] for c in keep)
    keep = [c for c in keep if c[1] <= 1.5 * med]
    tl = statistics.median(c[1] for c in keep)
    ph = {k: statistics.median(c[2].get(k, 0) for c in keep) for k in PHASES}
    return dict(label=os.path.basename(d.rstrip("/")), pp=pp[2], chunk=full, n=len(keep),
                ms=tl, ph=ph, dropped=len([c for c in body if c[0] == full]) - len(keep))


def expand(argv):
    dirs = []
    for a in argv:
        if os.path.exists(os.path.join(a, "stderr.log")):
            dirs.append(a)
        elif os.path.isdir(a):
            dirs += sorted(os.path.join(a, n) for n in os.listdir(a)
                           if os.path.exists(os.path.join(a, n, "stderr.log")))
    return dirs


def main(argv):
    groups = False
    if argv and argv[0] == "--groups":
        groups = True
        argv = argv[1:]
    rows = [r for r in (arm(d) for d in expand(argv)) if r]
    if not rows:
        print("no arms")
        return 1
    print(f"{'arm':<12} {'pp':>7} {'chunk':>6} {'ms/chunk':>8} {'out':>3} {'hc':>5} {'qsa':>5} "
          f"{'g/up':>5} {'down':>5} {'gdn':>5} {'deq':>4} {'GPU tok/s':>9}")
    for r in rows:
        p = r["ph"]
        print(f"{r['label']:<12} {r['pp']:>7.1f} {r['chunk']:>6} {r['ms']:>8.0f} "
              f"{r['dropped']:>3} {p['hc read']:>5} {p['qsa attn']:>5} {p['gemm gate/up']:>5} "
              f"{p['gemm down']:>5} {p['gdn']:>5} {p['dequant']:>4} "
              f"{r['chunk'] * 1000.0 / r['ms']:>9.1f}")
    if groups:
        by = {}
        for r in rows:
            key = r["label"].rsplit("-", 1)[0]
            by.setdefault(key, []).append(r)
        print()
        print(f"{'config':<12} {'n':>2} {'pp med':>7} {'pp min':>7} {'pp max':>7} {'spread':>7} "
              f"{'ms med':>7} {'ms min':>7} {'ms max':>7}")
        for key, rs in sorted(by.items()):
            pp = [r["pp"] for r in rs]
            ms = [r["ms"] for r in rs]
            print(f"{key:<12} {len(rs):>2} {statistics.median(pp):>7.1f} {min(pp):>7.1f} "
                  f"{max(pp):>7.1f} {100.0 * (max(pp) - min(pp)) / statistics.median(pp):>6.1f}% "
                  f"{statistics.median(ms):>7.0f} {min(ms):>7.0f} {max(ms):>7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
