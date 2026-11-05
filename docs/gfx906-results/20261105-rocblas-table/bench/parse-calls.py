#!/usr/bin/env python3
"""Turn a rocBLAS call trace of one engine run into the FP16 GEMM inventory of a chunk.

    python3 parse-calls.py <rocblas.trace.csv> [--probe tune-probe-1.log,...] [--chunk 2048]
                          [--chunk-ms 2295] [--table-gain]

The trace is rocBLAS's own `ROCBLAS_LOG_TRACE_PATH` file (`ROCBLAS_LAYER` bit 1), one CSV line
per library call:

    rocblas_gemm_ex,transA,transB,m,n,k,alpha,<a>,a_type,lda,<b>,b_type,ldb,beta,<c>,c_type,ldc,
                   <d>,d_type,ldd,compute_type,algo,solution_index,flags,atomics

Only rows that are the engine's `Gemm::f16_inplace` product are kept - f16_r A/B/C/D with f32_r
compute (gemm.cu), i.e. exactly the calls `STRATA_ROCBLAS_TUNING` would redirect.  In that call
rocBLAS's `m` is the projection's N (output features), `n` is T (tokens) and `k` is K, and
`ldc/2` is the engine's row stride `ldy` - the same key the tuning table uses.

`--probe` reads one or more `tools/hip/tune_rocblas` logs (the ones this record ships) for the
per-shape `default_ms` and best-of-164 `best_ms` at the recorded T, and prices the inventory:
`calls x (default_ms - best_ms)` is what a table would give back at that T.  Without `--probe`
the script prints the inventory only.

The verdict is the record's: GO at a saving of >= 1.5 % of the chunk (the shipped layer-3 kernel
is worth 1.4 %), BORDERLINE at >= 0.5 %, NO-GO below that.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

CASE = re.compile(r"^case T=(\d+) N=(\d+) K=(\d+) ldy=(\d+) default_ms=([\d.]+)")
SOLUTION = re.compile(r"^\s+solution=(-?\d+) ms=([\d.]+) gain=([\d.]+)")
KEEP = re.compile(r"^keep_default T=(\d+) N=(\d+) K=(\d+) ldy=(\d+) \(\d+ solutions; best ([\d.]+) ms"
                  r"(?: vs default ([\d.]+))?")

GO_PCT, BORDERLINE_PCT = 1.5, 0.5


def probe(paths: list[Path]) -> dict[tuple[int, int, int], tuple[float, float, int | None]]:
    """(n, k, t) -> (default_ms, best_ms, best_solution)."""
    out: dict[tuple[int, int, int], tuple[float, float, int | None]] = {}
    for path in paths:
        default: dict[tuple[int, int, int], float] = {}
        best: dict[tuple[int, int, int], tuple[float, int | None]] = {}
        shape: tuple[int, int, int] | None = None
        for line in path.read_text(errors="replace").splitlines():
            if (m := CASE.match(line)):
                shape = (int(m[2]), int(m[3]), int(m[1]))
                default[shape] = float(m[5])
            elif shape and (m := SOLUTION.match(line)):
                ms = float(m[2])
                if shape not in best or ms < best[shape][0]:
                    best[shape] = (ms, int(m[1]))
            elif (m := KEEP.match(line)):
                shape = (int(m[2]), int(m[3]), int(m[1]))
                if m[6]:
                    default[shape] = float(m[6])
                if shape not in best or float(m[5]) < best[shape][0]:
                    best[shape] = (float(m[5]), None)
        for shape, d in default.items():
            b, sol = best.get(shape, (d, None))
            out[shape] = (d, b, sol)
    return out



def calls(path: Path) -> dict[tuple[int, int, int, int], int]:
    """(n, k, ldy, t) -> how many times the engine called it."""
    count: dict[tuple[int, int, int, int], int] = defaultdict(int)
    for line in path.read_text(errors="replace").splitlines():
        f = line.split(",")
        if len(f) < 23 or f[0] != "rocblas_gemm_ex":
            continue
        if (f[8], f[11], f[15], f[18], f[20]) != ("f16_r", "f16_r", "f16_r", "f16_r", "f32_r"):
            continue
        if f[1] != "T" or f[2] != "N":     # the engine's C[N x T] = W^T . X
            continue
        count[(int(f[3]), int(f[5]), int(f[16]) // 2, int(f[4]))] += 1
    return count


def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    trace = Path(args[0])
    probe_paths: list[Path] = []
    chunk, chunk_ms, show_gain = 2048, 2295.0, "--table-gain" in args
    for i, a in enumerate(args[1:], start=1):
        if a == "--probe" and i + 1 < len(args):
            probe_paths = [Path(p) for p in args[i + 1].split(",")]
        elif a == "--chunk" and i + 1 < len(args):
            chunk = int(args[i + 1])
        elif a == "--chunk-ms" and i + 1 < len(args):
            chunk_ms = float(args[i + 1])
    rows = calls(trace)
    if not rows:
        print(f"no f16_r gemm_ex calls in {trace}", file=sys.stderr)
        return 2
    ms = probe(probe_paths) if probe_paths else {}

    per_t: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for (n, k, ldy, t), c in rows.items():
        per_t[t].append((n, k, ldy, c))

    print(f"trace: {trace.name} - {sum(rows.values())} f16_r gemm_ex calls, {len(rows)} distinct shapes, "
          f"T = {sorted(per_t)}")
    print(f"priced from: {', '.join(p.name for p in probe_paths) if probe_paths else 'nothing (inventory only)'}; "
          f"chunk reference {chunk_ms:.0f} ms")
    total_saving = 0.0
    for t in sorted((t for t in per_t if t > 64), reverse=True):
        block = per_t[t]
        print(f"\n--- {'the chunk' if t == chunk else f'T={t}'}: {sum(c for *_, c in block)} calls in "
              f"{len(block)} shapes, {t} tokens each ---")
        print(f"{'N':>6} {'K':>6} {'ldy':>6} {'calls':>6} {'default ms':>10} {'best ms':>9} "
              f"{'gain':>5} {'ms':>8} {'saving':>8}")
        block.sort(key=lambda r: r[3] * (lambda d: d[0] - d[1])(ms.get((r[0], r[1], t), (0.0, 0.0, None))),
                   reverse=True)   # biggest reclaimable saving first
        unmeasured = 0
        for n, k, ldy, c in block:
            d, b, _ = ms.get((n, k, t), (0.0, 0.0, None))
            if not d:
                unmeasured += c
                print(f"{n:6} {k:6} {ldy:6} {c:6}   unmeasured - run the tool on this shape")
                continue
            saving = c * (d - b)
            total_saving += saving
            print(f"{n:6} {k:6} {ldy:6} {c:6} {c * d:10.1f} {c * b:9.1f} {d / b:5.2f} {c * d:8.1f} "
                  f"{saving:8.1f}" + ("   (the default already wins)" if d <= b else ""))
        if unmeasured:
            print(f"  {unmeasured} calls of shapes with no probe row (left unpriced)")
    other = sum(c for t, block in per_t.items() if t <= 64 for *_, c in block)
    if other:
        print(f"\ndecode and other (T <= 64): {other} calls, not part of the chunk")

    if ms:
        pct = 100.0 * total_saving / chunk_ms
        verdict = "GO" if pct >= GO_PCT else ("BORDERLINE" if pct >= BORDERLINE_PCT else "NO-GO")
        print(f"\nsaving at the measured T: {total_saving:.0f} ms = {pct:.2f} % of a {chunk_ms:.0f} ms chunk"
              f" -> {verdict} (>= {GO_PCT} % GO, >= {BORDERLINE_PCT} % borderline)")
        if show_gain:
            print(f"prompt tok/s at the chunk: {1000 * chunk / chunk_ms:.1f} -> "
                  f"{1000 * chunk / (chunk_ms - total_saving):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
