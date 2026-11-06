# 2026-11-05 - a gfx906 rocBLAS solution table: priced at 9.0 %, shipped, and the engine pays 5.5 %

`STRATA_ROCBLAS_TUNING` is the engine's table of rocBLAS solution indices for the prompt path's FP16-out
GEMMs ([`src/prefill/rocblas_tuning.hpp`](../../../src/prefill/rocblas_tuning.hpp), `Gemm::f16_inplace`). It
ships for **gfx1030** (`tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt`), where rocBLAS's own pick is up to
**6.8x** slower than the best kernel the library holds for the same shape. gfx906 had no table and - as
section 1 found - no reader either. This record prices the lever with a call census, and the branch it is
committed on carries it: **206 ms of a 2295 ms chunk in isolation (9.0 %), -5.5 % of the chunk measured on
the engine, and no accuracy cost at all** (the generated tokens are identical in every arm).


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

## 1. The mechanism was inert in this build (un-gated on this branch)

Before the numbers: a table did **nothing** here. The only reader, `try_rocblas_f16o`, is inside
`#if defined(STRATA_ROCBLAS_AVAILABLE)` ([`gemm.cu:1350`](../../../src/prefill/gemm.cu)), and CMake defined
that macro under `if(STRATA_ENABLE_HIP AND STRATA_ROCBLAS_AVAILABLE)` only. This build is
`STRATA_HIP_GFX906=ON` (the gfx906 compat path) with `STRATA_ENABLE_HIP=OFF` -

```
$ grep -E 'STRATA_ENABLE_(HIP|CUDA)' rp-build/build-906/CMakeCache.txt
STRATA_ENABLE_CUDA:BOOL=ON
STRATA_ENABLE_HIP:BOOL=OFF
```

- so the macro was undefined, the reader compiled out, and `STRATA_ROCBLAS_TUNING` a **silent no-op** on this
platform (not a missing file: a missing branch). `CMakeLists.txt` now calls `find_package(rocblas)` in the
`STRATA_HIP_GFX906` branch too, defines the macro and links `roc::rocblas`, so the switch works here and the
configure log says so.


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

## 5. It shipped, and the engine pays 5.5 % of a chunk

Sections 1-4 priced the lever. This branch now carries it: the CMake un-gating above, the table
(`tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt`, 80 rows - the 10 shapes of section 3 at the eight token
buckets the gfx1030 table uses, 67 with a solution), the script that makes one
(`tools/hip/make-gfx906-table.sh`) and the gate (`gfx906_prefill_rocblas_gemm`, 4/4 PASS at `relative_l2`
2.1e-4, skipping without the variable).

`bench/gfx906/decode-ab.sh`, 32,500-token prompt, `--prefill 2048`, MTP off, six interleaved arms with
`STRATA_PREFILL_TIMING=1` (the logs are `bench/table-e2e-phases.txt` and `bench/table-e2e-arms-timed.txt`):

| arm | GPU ms/chunk | pp tok/s | hc read | qsa proj | router+shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| base 1 | 2172 | 861.1 | 264.0 | 164.1 | 120.4 |
| base 2 | 2213 | 855.0 | 279.1 | 164.6 | 118.1 |
| base 3 * | 2403 | 796.7 | 255.5 | 164.7 | 118.3 |
| table 1 | 2059 | 903.4 | 197.9 | 150.8 | 82.6 |
| table 2 | 2072 | 867.3 | 189.5 | 149.3 | 83.7 |
| table 3 | 2076 | 904.9 | 189.6 | 154.8 | 100.5 |

**-5.5 % of a 2048-token chunk** (2172/2213 -> 2059/2072/2076 ms), **+5.4-6 % prompt throughput**, and the
deltas land exactly on the three phases the census said they would: `hc read` **-78 ms (-29 %)**,
`router+shared` -26 ms (-22 %), `qsa proj` -13 ms (-8 %), every other phase flat under 1 %
(`bench/table-e2e-applied.txt` is the arm's own pick log: 10 shapes, `tuned_launches=4415 fallbacks=1`).
\* base 3 is excluded from the deltas - its own phases show an expert-stream hiccup (`dequant` 132 against
69, `gemm gate/up` 446 against 364), so the arm was slow before the flag was considered. A second, untimed
bracket reads the same direction (854.9/866.9/765.7 against 884.0/922.5/809.7, `bench/table-e2e-arms.txt`),
and **the generated token ids are identical in all twelve arms**.

**Why 5.5 % and not 9.0 %.** Section 3's 206 ms prices each GEMM in isolation. The engine hides part of the
`qsa proj` saving behind the rest of that phase (the 512-wide pair is issued beside the selection and the
residual traffic), so that phase takes -8 % where its GEMMs alone allow -50 %; the `hc read` and the shared
expert are exposed and take -29 % and -22 %. The 9.0 % was the ceiling, 5.5 % is what the pipeline pays -
which is what section 4's caveat (2) said to expect, and why the bracket was the measurement that mattered.

## 6. What is left

1. **Done.** The census (section 3), the un-gating (section 1), the table and its script, the gate, and the
   bracket (section 5).
2. Regenerate after any ROCm refresh: `tools/hip/make-gfx906-table.sh` (~25 minutes) rewrites
   `tools/hip/gfx906-rocblas-5.5.0.<version>.txt`; a stale file is refused loudly by the engine, which then
   runs the default kernels.
3. Only if `STRATA_DENSE_MMQ` or `STRATA_PF_RP` were ever turned off would the shape list change: the wide
   products (experts, GDN) would come back to rocBLAS and the table would need their rows too - the tool's
   14-shape default list already covers them.



## 7. Status after the fork-branch sync (2026-11-06)

The branch was cut from `048effa1`; it now also carries the fork's v0.2.11-3 sync (1816 upstream
commits, merged in `3f22a974`/`b581a6b0`; the conflict table and the build check are in the
[2026-11-06 section of docs/GFX906.md](../../GFX906.md)).

- **The mechanism is re-verified on the merged tree.** The un-gate survived the sync - upstream's
  `if(STRATA_ENABLE_HIP AND STRATA_ROCBLAS_AVAILABLE)` is replaced by `find_package(rocblas QUIET)` inside the
  `STRATA_HIP_GFX906` block plus a define/link site gated only on `STRATA_ROCBLAS_AVAILABLE` - and
  `gfx906_prefill_rocblas_gemm` passes against the shipped file:
  `prefill gemm: rocBLAS tuning enabled (80 rows, 67 with a solution, gfx906, rocBLAS 5.5.0.cd957402309)`,
  4/4 cases PASS (relative_l2 2.1e-4, outputs within FP16 rounding of `hipBLASEx`). It needs
  `STRATA_HIP_PROMPT_F16=1` in the environment, because the table serves that route only.
- **The 5.5 % is not re-measured yet.** Sections 1-6 were taken on the `048effa1` tree; the merged tree is
  engine `0.1.57` with upstream's prompt path restructured (`prefill.cpp` +2691/-77) and its FP16 weight
  cache newly compiled into the gfx906 build (off by default: `w16_auto_default_()` starts false and
  `STRATA_PF_W16_CACHE_MIB` is unset). The bracket needs the cards, which the running service holds
  (1.7 / 0.3 GiB free on 2026-11-06); it is owed for the first window with the deployment stopped, and the
  numbers above stay attached to the engine they were measured on until then.

