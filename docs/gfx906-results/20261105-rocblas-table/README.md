# 2026-11-05 - a gfx906 rocBLAS solution table: feasible, worth an estimated 2-3.5 % of a chunk, not in this branch

`STRATA_ROCBLAS_TUNING` is the engine's table of rocBLAS solution indices for the prompt path's FP16-out
GEMMs ([`src/prefill/rocblas_tuning.hpp`](../../../src/prefill/rocblas_tuning.hpp), `Gemm::f16_inplace`). It
ships for **gfx1030** (`tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt`), where rocBLAS's own pick is up to
**6.8x** slower than the best kernel the library holds for the same shape. gfx906 has no table. This record
measures whether it wants one. **It does - but not in this branch**, and the win is on the shapes the
deployment keeps on the FP16 path on purpose.

Everything here was measured on this box the same day: rocBLAS **5.5.0.cd957402309** (the deployment's
`/opt/rocm/lib`), the deployment's two-card gfx906 split, T = 2048 (the deployed chunk size).

## 1. The mechanism is inert in this branch's gfx906 build

Before the numbers: a table would do **nothing** today. The only reader, `try_rocblas_f16o`, is inside
`#if defined(STRATA_ROCBLAS_AVAILABLE)` ([`gemm.cu:1350`](../../../src/prefill/gemm.cu)), and CMake defines
that macro only under `if(STRATA_ENABLE_HIP AND STRATA_ROCBLAS_AVAILABLE)`
([`CMakeLists.txt:838`](../../../CMakeLists.txt)). This branch's gfx906 build is `STRATA_ENABLE_CUDA=ON` +
hipcc -

```
$ grep -E 'STRATA_ENABLE_(HIP|CUDA)' rp-build/build-906/CMakeCache.txt
STRATA_ENABLE_CUDA:BOOL=ON
STRATA_ENABLE_HIP:BOOL=OFF
```

- so the macro is undefined, the reader is compiled out, and `STRATA_ROCBLAS_TUNING` is a **silent no-op**
on this platform (not a missing file: a missing branch). A gfx906 table is therefore the smaller half of the
work; un-gating the reader for the CUDA-path build is the other half.

## 2. The tool runs here unchanged, and rocBLAS offers 164 solutions a shape

`tools/hip/tune_rocblas.cpp` was written for RDNA2 but needs no change: it links against rocBLAS directly,
so it builds and runs in this box's images as-is.

```sh
# build (in the builder image; the tool only needs hipcc and librocblas)
hipcc -O3 --offload-arch=gfx906 tools/hip/tune_rocblas.cpp -o build-906/tune_rocblas -L/opt/rocm/lib -lrocblas
# time the default kernel and every solution rocBLAS lists for one shape, checking each against the default
tune_rocblas --case 2048,10240,2560,10240 --tuning-out /tmp/p.txt
```

The deployment's rocBLAS holds 210 Tensile files for gfx906 under `/opt/rocm/lib/rocblas/library/gfx906`, and
`rocblas_gemm_ex_get_solutions` returns **164 solutions** for every shape tried. Each candidate is checked
against the default kernel's own output: identical FP16 result within rounding (`relative_l2` <= 2e-3,
`max_abs` <= 4 ULP) and no write past the `N x T` result. The first shape came back **bit-identical**
(`relative_l2` 0, `max_abs` 0).

## 3. The shapes (T = 2048; the default kernel against the best of its 164 solutions)

| N | K | ldy | default ms | best ms | gain | what it is |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10240 | 2560 | 10240 | 7.347 | 7.250 | 1.01 | **the hyper-connection read** (36 x 7.35 = 264 ms; the phase reads 260) |
| 12288 | 2560 | 12288 | 8.518 | 8.730 | 0.98 | the GDN input projection |
| 2560 | 2560 | 2560 | 1.617 | 1.639 | 0.99 | |
| 6144 | 2560 | 6144 | 7.665 | 4.154 | **1.85** | a 6144-wide expert gate/up |
| 2560 | 6144 | 2560 | 8.458 | 4.942 | **1.71** | the matching expert down |
| 2560 | 10240 | 2560 | 15.14 | 8.519 | **1.78** | |
| 2560 | 640 | 2560 | 0.646 | 0.427 | **1.51** | the shared-expert down (K = 640: no int8 MMQ) |
| 640 | 2560 | 640 | 0.745 | 0.580 | **1.28** | |
| 512 | 2560 | 512 | 1.441 | 0.496 | **2.91** | **`attn_k` / `attn_v`** (`STRATA_DENSE_MMQ_SKIP` keeps them FP16) |
| 128 | 2560 | 128 | 0.295 | 0.173 | **1.71** | |

So gfx906's heuristic is *not* uniformly good. It is right on the three widest FP16 products
(1.01 / 0.98 / 0.99 - including the one phase still on the FP16 path in the deployed configuration, the
hc read) and up to **2.9x** wrong on the narrow ones. The RDNA2 story is real here too, on the other half of
the shape list.

## 4. Why it is still not in this branch

- **The shapes with the gap are the ones the branch already took off rocBLAS.** 6144x2560 and 2560x6144 are
  the experts', and layer 2's `STRATA_PF_RP` / `STRATA_DENSE_MMQ` route them to int8 (-132 and -58 ms of
  those two phases). Tuning them would be a second, worse answer to a question already answered: the port's
  -383 ms beats what a table could give on the same shapes.
- **But the shapes deliberately left on FP16 are where it wins, and that is free.** The K = 640 shared-expert
  down cannot use the int8 path (`native_mmq` rejects `K % 256 != 0`), the `attn_k`/`attn_v` pair is skipped
  on purpose (they write the K/V the decode reads, and int8 there halves the logits' KL), and the router is
  small. Those are the 1.28x / 1.51x / 1.71x / 2.91x rows. A table spends **no accuracy budget at all** - the
  same FP16 products, summed in another order and rounded once, verified row by row by the tool - which no
  other lever left on this box can say.
- **Sizing it, on the phase table:** `attn_k` + `attn_v` at T = 2048 are two 512x2560 GEMMs a layer, 1.44 ms
  each measured; 36 layers x 2 x 1.44 = **104 ms of the 165 ms `qsa proj`**, and a table cuts 0.94 ms of each
  - **~68 ms, 3.0 % of a 2295 ms chunk**. The shared-expert down adds ~8 ms (0.4 %). That is larger than
  layer 3's kernel (-1.4 %) and it costs no accuracy.
- **It is an artifact pinned to one library build.** The header carries `gfx906` + `5.5.0.cd957402309`; any
  ROCm refresh invalidates every row, exactly as `tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt` shows for the
  RDNA2 file - and the five separate gfx1100 hipBLASLt tables in `tools/hip/` are that cost, paid repeatedly.

## 5. What it would take, and what would change the verdict

1. One engine run on a single 2048-token chunk with `ROCBLAS_LAYER=2` + `ROCBLAS_LOG_BENCH_PATH`, summed by
   shape. That replaces the phase-name attribution in section 4 with the shapes and ms the deployment
   actually calls - the one number this record does not measure (it needs the server stopped; the model
   needs both cards).
2. Un-gate `try_rocblas_f16o` for a `STRATA_ENABLE_CUDA`-mode build (or link rocBLAS there), with
   `tests/hip/prefill_rocblas_gemm.cpp` as the gate - it already refuses a table for another architecture or
   rocBLAS build.
3. `tune_rocblas --tuning-out tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt` for the live shape list, and an
   A/B/A bracket on `qsa proj` / `router+shared` through
   [`bench/gfx906/decode-ab.sh`](../../../bench/gfx906/decode-ab.sh) reading those two phase columns.

The verdict flips to "worthless" only if step 1 shows the deployment does not call the narrow shapes at all,
which the `qsa proj` arithmetic above says is unlikely.
