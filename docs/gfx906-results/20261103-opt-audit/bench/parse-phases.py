#!/usr/bin/env python3
"""Per-chunk phase table from a rp-ab.sh arm's stderr.log (STRATA_PREFILL_TIMING=1).

  parse-phases.py <arm dir> [...]      prints mean ms/phase over the chunks, and the per-chunk GPU timeline
Chunk 1 and any chunk whose GPU timeline is an outlier are kept in the per-chunk list but excluded from the
mean (the first chunk pays first-touch, one chunk in the 32K arm is 2x the others).
"""
import re
import sys

LINE = re.compile(r"prefill timing: (\d+) tokens, GPU timeline (\d+) ms, wall (\d+) ms.*?:(.*)$")
ITEM = re.compile(r"([a-z+/ ]+?) (\d+) \((\d+\.\d)%\)")


def parse(path):
    chunks = []
    with open(path, errors="replace") as f:
        for ln in f:
            m = LINE.search(ln)
            if not m:
                continue
            tok, gpu, wall, rest = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            items = {name.strip(): (int(ms), float(pc)) for name, ms, pc in ITEM.findall(rest)}
            chunks.append({"tokens": tok, "gpu": gpu, "wall": wall, "phases": items})
    return chunks


def main(dirs):
    for d in dirs:
        path = d.rstrip("/") + "/stderr.log"
        chunks = parse(path)
        if not chunks:
            print(f"== {d}: no timing lines")
            continue
        print(f"== {d}: {len(chunks)} chunks")
        # the request's own summary line ("N tokens, GPU timeline ...") is not a chunk: keep the modal size
        sizes = {}
        for c in chunks:
            sizes[c["tokens"]] = sizes.get(c["tokens"], 0) + 1
        modal = max(sizes, key=lambda k: sizes[k])
        chunks = [c for c in chunks if c["tokens"] == modal]
        print(f"   {len(chunks)} chunks of {modal} tokens")
        print("   per chunk: gpu ms | " + " ".join(str(c["gpu"]) for c in chunks))
        steady = chunks[1:]
        tot = {}
        for c in steady:
            for k, (ms, _) in c["phases"].items():
                tot[k] = tot.get(k, 0) + ms
        gpu_mean = sum(c["gpu"] for c in steady) / len(steady)
        wall_mean = sum(c["wall"] for c in steady) / len(steady)
        print(f"   steady mean: GPU {gpu_mean:.0f} ms, wall {wall_mean:.0f} ms, "
              f"{steady[0]['tokens'] / (wall_mean / 1000):.0f} tok/s (wall)")
        for k, ms in sorted(tot.items(), key=lambda kv: -kv[1]):
            m = ms / len(steady)
            print(f"   {k:<18} {m:8.1f} ms  {100 * ms / sum(tot.values()):5.1f}%  "
                  f"{100 * m / gpu_mean:5.1f}% of GPU")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
