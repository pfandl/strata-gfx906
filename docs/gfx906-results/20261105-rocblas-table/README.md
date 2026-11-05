# 2026-11-05 - a gfx906 rocBLAS solution table: 206 ms of a chunk (9.0 %), no accuracy cost, not in this branch

`STRATA_ROCBLAS_TUNING` is the engine's table of rocBLAS solution indices for the prompt path's FP16-out
GEMMs ([`src/prefill/rocblas_tuning.hpp`](../../../src/prefill/rocblas_tuning.hpp), `Gemm::f16_inplace`). It
ships for **gfx1030** (`tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt`), where rocBLAS's own pick is up to
**6.8x** slower than the best kernel the library holds for the same shape. gfx906 has no table. This record
measures whether it wants one: **it does - 206 ms of a 2295 ms chunk (9.0 %) with no accuracy cost at all**,
measured as a call census plus a per-shape probe, not estimated - but it is a follow-up of its own, because on
this branch the reader is compiled out and the artifact is pinned to one library build.

Everything here was measured on this box the same day: rocBLAS **5.5.0.cd957402309** (the deployment's
`/opt/rocm/lib`), the deployment's two-card gfx906 split (48 layers, 24 a card), T = 2048 (the deployed chunk
size), one 4000-token prompt = two chunks.

## How to read the numbers

rocBLAS picks a kernel for each GEMM shape from its own table, and the pick is a heuristic. A tuning table is
one line per shape naming **a different kernel out of rocBLAS's own catalogue**:

```
f16o 512 2560 512 2048 -605557544     # attn_k/attn_v: rocBLAS's pick 1.451 ms -> this solution 0.497 ms
```

`gain` below is the ratio between the two. It is not a precision trade: same FP16 inputs, same FP16 output,
same products, another kernel's summation order - every row the tool writes was checked against the default
kernel's own output (`relative_l2`, ULP, no write past the result).

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

## 3. The census: what one 2048-token chunk actually calls, and what a table gives back

Two measurements, both in `bench/`:

- **The inventory.** `bench/census.sh` runs one engine start (4000 tokens = two chunks of 2048) through
  [`bench/gfx906/decode-ab.sh`](../../../bench/gfx906/decode-ab.sh) with `ROCBLAS_LAYER=7` and
  `ROCBLAS_LOG_{TRACE,BENCH,PROFILE}_PATH` set, so **rocBLAS writes one CSV line per call itself** - shape,
  strides, algo, solution index (`bench/rocblas.trace.csv`). Nothing of ours sits in the path, so the counts
  are the library's.
- **The price.** `bench/parse-calls.py` keeps the lines that are `Gemm::f16_inplace` (f16_r A/B/C/D, f32
  compute, `transA=T`/`transB=N`) - exactly the calls a table would redirect - counts them by
  `(N, K, ldy, T)` and multiplies by the per-shape `default_ms`/`best_ms` from `bench/tune-census.log`, a
  `tune_rocblas` run on the same 10 shapes at T = 2048.

The first chunk (T = 2048, 48 layers) issues **554 such calls in 10 shapes**:

| N | K | ldy | calls | default ms | best ms | gain | GEMM ms | saving | what it is |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 2560 | 512 | 84 | 1.451 | 0.497 | **2.92** | 121.9 | **80.2** | `attn_k` / `attn_v` (`prefill.cpp:3324`) |
| 4 | 10240 | 4 | 96 | 0.876 | 0.381 | **2.30** | 84.1 | **47.5** | an hc mixing product (N = HC = 4 over the 10240-wide streams) |
| 320 | 10240 | 320 | 96 | 1.892 | 1.454 | **1.30** | 181.6 | **42.0** | the hc read's **down** projection, `LR=320` (`prefill.cpp:1869`) |
| 1 | 2560 | 1 | 48 | 0.365 | 0.035 | **10.57** | 17.5 | **15.8** | the shared-expert gate scalar, `gemm.bf16(..., 1, N)` (`prefill.cpp:3629`) |
| 2560 | 640 | 2560 | 48 | 0.645 | 0.427 | **1.51** | 31.0 | **10.5** | the shared-expert **down** (`K = 640`: `native_mmq` rejects `K % 256 != 0`) |
| 10240 | 320 | 10240 | 96 | 0.823 | 0.750 | 1.10 | 79.0 | 7.1 | the hc read's **up** projection (`prefill.cpp:1871`) |
| 128 | 2560 | 128 | 12 | 0.294 | 0.169 | 1.74 | 3.5 | 1.5 | shape only - the trace carries pointers, not names |
| 48 | 2560 | 96 | 72 | 0.129 | 0.115 | 1.12 | 9.3 | 1.0 | the router (48 experts; `ldy` 96 is the padded stride) |
| 10240 | 2560 | 10240 | 1 | 7.552 | 7.322 | 1.03 | 7.6 | 0.2 | one call a chunk (setup) |
| 2560 | 2560 | 2560 | 1 | 1.607 | 1.682 | 0.96 | 1.6 | -0.1 | the default already wins |
| **sum** | | | **554** | | | | **537** | **206** | **8.96 % of a 2295 ms chunk** |

Same run, second chunk (1951 tokens): the same 554 calls in the same 10 shapes - the inventory is steady state.
A second run of `bench/census.sh` (fresh directory) reproduced it **call for call** - 1108 calls, 554 a chunk,
the same 10 shapes and counts; the two traces differ only in the logged pointer addresses.

**What is not in the list is the news.** `12288x2560` (the GDN input projection), `6144x2560` and
`2560x6144` (expert gate/up and down), `640x2560` and `2560x10240` never appear: in the deployed configuration
(`STRATA_PF_RP=1 STRATA_DENSE_MMQ=1 STRATA_HIP_PROMPT_F16=1`) those shapes leave through the int8 port and
never touch rocBLAS. The FP16 GEMMs left on the library are the **narrow, awkward, per-layer** ones - which are
exactly the shapes gfx906's heuristic gets wrong (1.10x to 10.6x here).

## 4. Cross-check, and what could make it smaller

- **The inventory's ms agree with the engine's own phase counter.** The four hc shapes above sum to
  **362 ms**; the same arm's `STRATA_PREFILL_TIMING` phase table reads `hc read 351`. `attn_k`/`attn_v`'s
  121.9 ms sits inside `qsa proj 213`; the shared expert's 48.5 ms inside `router+shared 117`. Two
  independent instruments, one number.
- **These GEMMs are ~24 % of the chunk's GPU timeline** (537 of 2206 ms), which is more than the phase names
  suggest - the hc read alone is a sixth of the pass.
- **Caveats, in order of size.** (1) The arm was run with the trace on, so its phase table is inflated
  (2206 ms against the record's 2295 ms baseline with MTP on - do not compare them); the per-shape ms are
  from an idle GPU, so they are the conservative side. (2) The saving assumes the GEMMs are on the critical
  path: the 320-wide hc products and `attn_k`/`attn_v` are not streamed over the SSD, so they should be, but
  only an A/B/A bracket proves it. (3) The T = 1951 chunk reuses the T = 2048 prices.

## 5. Why it is still not in this branch

- **The reader is compiled out here** (section 1). A table file added to this branch would do nothing on this
  box until `try_rocblas_f16o` is un-gated for the `STRATA_ENABLE_CUDA` build, which is a code change to the
  GEMM header, not a data file - a different kind of change from everything else this branch ships.
- **The win is real and free, and that is the argument for doing it next.** 206 ms of a 2295 ms chunk is
  **9.0 %**, with **no accuracy budget at all**: the same FP16 products, summed in another order by another
  kernel rocBLAS already ships, each row verified against the default's own output (`relative_l2` <= 4e-5,
  `max_abs` <= 0.0625 on every row that matters here, no write past the result). For scale: layer 3's
  prompt-attention kernel is worth 1.4 %, the int8-score option 1.2 %, the GDN chunked scan 0 %. This is the
  largest lever left on this box and the only one that costs nothing numerically.
- **It is an artifact pinned to one library build.** The header carries `gfx906` + `5.5.0.cd957402309`; any
  ROCm refresh invalidates every row, exactly as `tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt` shows for the
  RDNA2 file - and the five separate gfx1100 hipBLASLt tables in `tools/hip/` are that cost, paid repeatedly.
- **And it is the same "two answers to one question" risk as before, only inverted.** The shapes the port
  already owns (experts, GDN, the hc read's int8 neighbours) *are* the table's biggest raw gains by ratio
  (1.85x at `6144x2560`); the census shows they never reach rocBLAS in this configuration, so the table
  cannot collide with them - but if `STRATA_DENSE_MMQ` or `STRATA_PF_RP` were ever turned off, most of this
  206 ms would go back to rocBLAS and the table's value would change shape. A table is only the right lever
  *because* the port came first.

## 6. What it would take

1. **Done, in this record.** The census (section 3): one engine run on two 2048-token chunks with the rocBLAS
   logs on, priced per shape. It replaces the earlier phase-name estimate (3 %) with the measured 9.0 %.
2. Un-gate `try_rocblas_f16o` for a `STRATA_ENABLE_CUDA`-mode build (or link rocBLAS there), with
   `tests/hip/prefill_rocblas_gemm.cpp` as the gate - it already refuses a table for another architecture or
   rocBLAS build.
3. `tune_rocblas --tuning-out tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt` for the live shape list (the 10
   rows of section 3 are what it must cover - the tool's 14-shape default list still contains `12288x2560`,
   `6144x2560` and `2560x6144`, which the deployment never calls), and an A/B/A bracket through
   [`bench/gfx906/decode-ab.sh`](../../../bench/gfx906/decode-ab.sh) reading `hc read`, `qsa proj` and
   `router+shared`. The bracket is the number that matters: section 4's caveat (2) says where it could
   dissipate.

