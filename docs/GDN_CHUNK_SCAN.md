# GDN chunked scan for gfx906 (design)

Status: **design, then implementation** (2026-11-05, `gfx906-levers`).  Nothing here changes the
deployment until both halves of the gate are run.

## The lever, in the box's own numbers

The gfx906 prompt chunk (2048 tokens) spends, per `STRATA_PREFILL_TIMING=1` on the shipped stack:

| phase | ms/chunk | share |
| --- | ---: | ---: |
| `qsa attn` | 274.5 | 12.0 % |
| `gemm gate/up` | 415 | 18.2 % |
| `gemm down` | 186.5 | 8.2 % |
| `dequant` | 128 | 5.6 % |
| `hc read` | 260 | 11.4 % |
| **`gdn`** | **304** | **13.3 %** |
| - of which `gdn recurrence` | **199** | **8.7 %** |
| - `gdn out proj` | 123 | 5.4 % |
| - `gdn conv+gates` | 9 | 0.4 % |

`gdn recurrence` is the `gdn_recurrence(...)` call aggregated over the model's **36 GDN layers**
(`n_layers` 48, `qsa_interval` 4), i.e. **5.53 ms per layer** for 2048 tokens, 2.7 us per token.

That is far above the arithmetic: one layer's recurrence is
`48 heads x 128 rows x 128 cols x 2048 tokens x ~3 FMA` = **4.8 GFLOP**, which this box's ~6.5 TFLOPS of
FP32 would do in 0.75 ms, and its compulsory traffic is ~393 MB (q and k of each key head read by 4
column blocks; v, gate, beta) at ~1 TB/s.  The kernel is **dependency-bound**: `gdn_rec_cols_pipe_kernel`
exposes `HV x S/CB = 48 x 4 = 192` blocks (3.2 per CU on 60 CUs) and every one of them walks the 2048
tokens in order with two or three `__syncthreads()` per token, so the token loop - not the FLOPs - is the
clock.  Nothing about that is fixable inside the token-serial form.

## The algorithm: the delta rule in chunks (WY form)

Same as `src/prefill/kernels.cu:866-890` and flash-linear-attention's chunked delta rule.  For a chunk of
`GCH = 32` tokens, `g_t = exp(gate_t)`, `gamma_t` the gate sum from the chunk's start through `t`, `S0` the
state entering the chunk (`128 x 128` per value head):

```
A[t][i] = beta_t exp(gamma_t - gamma_i) k_t . k_i   (i < t)      T = (I + A)^-1   (unit lower triangular)
Y_t     = beta_t (v_t - exp(gamma_t) S0^T k_t)                   D = T Y
o_t     = exp(gamma_t) S0^T q_t + sum_{i<=t} exp(gamma_t - gamma_i) (q_t . k_i) D_i
S       = exp(gamma_last) S0 + sum_t exp(gamma_last - gamma_t) k_t D_t^T
```

Every operation inside a chunk is a small dense product (32x128, 32x32, 128x32), so the chunk exposes all
of the parallelism the token form hides; what stays serial is one 32-token state update per chunk - 32 of
them per super-block, done by the block that holds that state slice.  All exponents are `<= 0`, so nothing
overflows, and the form is FP32 like the serial kernels: the difference is the *order* of the sums.

## What already ships, and why it does not run here

`src/prefill/kernels.cu` carries both kernels (`gdn_chunk_prep_kernel`, `gdn_chunk_scan_kernel`),
`gdn_rec_chunked()`, the per-device scratch and the `STRATA_GDN_CHUNKED` switch, plus a gfx11 matrix-core
variant (`gdn_rec_wmma`).  Measured on an RTX 5090: 1.81x the token recurrence at 2048+ tokens
(`gdn_rec_parity --bench`).  None of it reaches gfx906:

1. the CUDA-core pair is inside `#if !defined(__HIPCC__)` - **not compiled into the AMD engine at all**;
2. `gdn_rec_chunked()` refuses a card below `cc_major 8`, below **128 SMs** (the scan's 128 blocks are
   meant to be one wave) and with less than its 93.7 KB opt-in LDS;
3. gfx906 has **64 KB** of LDS per block, so the scan kernel does not fit (91.5 KB) even with the other
   two gates skipped via `STRATA_GDN_CHUNKED=2`;
4. it stages with `cp.async` and prefetches with `prefetch.global.L2` - PTX, not HIP;
5. `gdn_chunk_prep_kernel` does `if (warp >= VPK) return;` and then `__syncwarp()`.  On a wave64 that is
   a **hazard**: threads 96-127 exit, the wave that contains them is partially exited, and a barrier
   inside it is illegal.  It is also why the gfx11 variant is a separate kernel and not this one.

## The gfx906 design

Geometry is fixed by `core/geometry.hpp`: `S = 128` (key and value head width), `HK = 16` key heads,
`HV = 48` value heads, `VPK = HV/HK = 3` value heads per key head, `C = 10240` row stride.
Choices: `GCH = 32` (a lane per token in the prep's solve), `GDV = 8` value columns per scan block,
`GSB = 1024` tokens per super-block (32 chunks).

**Prep kernel** - one block per (chunk, key head): 256 threads, `dim3(nch, HK)`.  The k/q stage, the
`k_t.k_i` / `q_t.k_i` products and the solve are the CUDA kernel's, with three porting changes:

- `__shfl_up_sync` / `__shfl_sync` take an explicit **width 32** (the solve is a 32-token prefix sum and a
  32-lane broadcast; HIP defaults to the wave width);
- no early return: all eight half-waves stay alive and the solve's row sequencing uses `__syncthreads()`
  (32 barriers per block-chunk) instead of `__syncwarp()`.  The solve is 3 of the 8 half-waves; the other
  five wait at the same barriers, which is what makes the wave64 case legal;
- LDS: `2 x 32 x 129 + 2 x 32 x 33` floats = **40.5 KB** (unchanged from the CUDA kernel, and it fits).

**Scan kernel** - one block per (key head, 8 value columns): `dim3(HK, S/GDV) = (16, 16) = 256 blocks`,
`128 x VPK = 384` threads (a 128-thread group per value head, sharing the block's staged k and q).  Each
thread owns one token's row of the products (2 columns) and a 4x2 tile of the state slice.  Per chunk:
`X = K S0`, `Q S0` -> `Y = beta (V - exp(gamma) X)` -> `D = T Y` -> `o = exp(gamma) Q S0 + P D` ->
`S = exp(gamma_last) S0 + K^T (D exp(gamma_last - gamma))`.  LDS:

```
k     [GCH][S+4]                32 x 132 x 4 = 16.9 KB      (single buffer: staged after the update)
q     [GCH][S+4]                32 x 132 x 4 = 16.9 KB
per value head (x3):  state [S][GDV] + Y + D + D*decay [GCH][GDV]
                        (128x8 + 3x32x8) x 4 = 7.2 KB       x3 = 21.5 KB
                                                     total = 55.3 KB   (fits 64 KB, one block per CU)
```

`GDV = 8` and a single k buffer are what buy the fit: the CUDA shape (`GDV = 16`, k double-buffered)
needs 91.5 KB.  The price is two extra reads of k and q per layer (each key head's rows are staged by one
block instead of three at `GDV = 16`) and one barrier where the double buffer had overlap: ~1.2 GB per
layer per chunk, i.e. ~1 ms, against a phase that costs 5.5 ms per layer today.

Because there is no `cp.async`, the CUDA kernel's "stage the next chunk while this one computes" is
replaced by an explicit order with five barriers per chunk, and the staging itself keeps the existing
idiom of issuing every load before the first store (`float4` copies, `t < n` zero-filled tails).

**Dispatch**: in `gdn_recurrence_variant`'s HIP branch, after `gdn_rec_wmma` declines (it needs gfx11
matrix cores), `if (chunked && T >= kGdnChunkedMin && gdn_rec_chunked_gfx906(...))` with
`STRATA_GDN_CHUNKED` as the switch.  `gdn_out_norm_kernel` still runs after it, unchanged, so `y`, `y16`,
`z` and `gamma` keep their contract, and the carried state keeps its `[HV][S][S]` layout between
super-blocks.  The function returns "nothing launched" (so the caller falls through to the serial kernels)
when the arch is not gfx906, `T < 128`, the scratch allocation fails, or `S % GDV` does not divide.

**Not in scope**: a matrix-core variant (gfx906 has none), FP16/BF16 operands (the FP32 reordering is
already the accuracy question; a half-precision one would be a different, much larger one), the decode
path (the recurrence there is 1 token - the chunked form degenerates), and `gdn out proj` (a GEMM that
belongs to the layered record's `bf16`/`native` work).

## Built and measured on the box (2026-11-05)

`src/prefill/gdn_chunk_906.cu` is the design above; `bench/gfx906_gdn_chunk_probe.cpp` calls the public
`gdn_recurrence` with `STRATA_GDN_CHUNKED` on and off and dumps `y`, `y16` and the carried state.
T = 2048, one call, 5 reps, median; `STRATA_GDN_CHUNK_PROF=1` splits the two launches.

| | total | prep | scan | norm + state (by subtraction) |
| --- | ---: | ---: | ---: | ---: |
| token recurrence (`STRATA_GDN_CHUNKED=0`) | 7.24 ms | - | - | - |
| chunked (`=1`) | 7.19 ms | 0.94 | 4.34 | 1.91 |
| component (recurrence only, no norm) | 5.28 vs ~5.3 | | | |

**Correctness: it is the same recurrence.** Against the token recurrence, over 2048 tokens x 48 heads x 128
columns plus the carried state:

| output | max_abs | rms | reference rms | relative |
| --- | ---: | ---: | ---: | ---: |
| `y` (pre-norm, fp32) | 5.53e-05 | 2.84e-07 | 0.202 | 1.4e-6 |
| `y16` (post-norm, fp16) | 1.95e-03 | 9.34e-06 | 0.556 | 1.7e-5 (a few FP16 ULP) |
| carried state | 5.72e-06 | 4.06e-07 | 0.266 | 1.5e-6 |

which is the "another order of the same sums" class this tree's harnesses expect (state inside 5e-7..2e-3,
`y` inside 1e-5..2e-2), on the tight end.

**Performance: at parity, so the lever does not pay as built.** The scan is 4.34 ms against the ~5.3 ms the
token recurrence takes for the same recurrence - no win - and the prep adds 0.94 ms back. T = 4096 shows the
same thing (15.00 vs 14.84 ms total, scan 9.1 ms). The three costs the design named as risks are all real and
they add up to the parallelism it gains:

- **occupancy**: 55.3 KB of LDS is one block per CU (6 waves), and every chunk has four `__syncthreads()`;
  32 chunks x 256 blocks over 60 CUs is a barrier chain, not a matmul.
- **the prep is 4 % of peak** (0.94 ms for 134 M FMA): its 32 `__syncthreads()` per block-chunk are the
  single-wave solve that could not use `__syncwarp()`.
- **the form does ~20 % more arithmetic** (the `T`, `P` and `D` products the token recurrence does not
  compute), and on this card the token recurrence already runs at ~1.9 TFLOPS, so a 20 % tax on 2.2 TFLOPS
  is a wash.

**Where the phase actually goes** (this is the useful finding): of the 7.2 ms, the recurrence is 5.3 ms and
**1.9 ms is `gdn_out_norm_kernel`**, which launches one 128-thread block per (token, head) - 98,304 blocks
per layer - and is the same kernel `src/prefill/kernels.cu:1588` already replaced with a grid-stride
version (`gdn_out_norm_loop_kernel`, "the old kernel launched a 128-thread block per row (T * 48 blocks;
~45 GB/s on the 8060S)"). That is a separate, bit-comparable lever on 3 % of the chunk and it does not need
this design at all.

Next steps, in order of expected value: (1) the norm kernel's grid-stride variant, (2) `GDV = 4` (27.5 KB,
two blocks per CU) for the scan, (3) the single-wave prep solve. The chunked scan itself is kept **opt-in and
off** (`STRATA_GDN_CHUNKED`, default off, and `docker-compose.yml` does not set it): it is correct and it is
not faster, and a numerical change that buys nothing is the `STRATA_GR_SPLIT` case.

### Reproduce

```sh
# one process per arm (the dispatch caches its decisions per process); the deployment is stopped by the caller
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --group-add "$(stat -c %g /dev/kfd)" --ipc=host --user "$(id -u):$(id -g)" --entrypoint bash \
  -v "$PWD/rp-build":/work -v "$PWD/data":/data strata-gfx906:local -lc \
  'env STRATA_GDN_CHUNKED=1 STRATA_GDN_CHUNK_PROF=1 \
     /work/build-906/gfx906_gdn_chunk_probe 2048 5 /data/gdn/c1-2048.bin'   # and the same with =0
```

The two dumps are `y` (fp32), `y16` (fp16) and the carried state, in that order; the numbers in the table
above are `|c1 - c0|` over each.


## The numerical contract

The chunked form computes the same recurrence with different summation orders: FP32 throughout, the
`32x32` solve and the `T Y`, `P D`, `K^T D` products replacing the token-by-token accumulation.  The
tree's own harness already states the bar (`src/kernels/gdn_chunk_parity.cpp`,
`src/prefill/gdn_rec_parity.cu`): the FMA variant is expected within **1e-5..2e-2 absolute / relative on
`y`** and **5e-7..2e-3 on the state** against a host double-precision recurrence, which is the "another
order of the same sum" class, not a format change.

The gate is the project's, unchanged: `hip_gdn_chunked` (dispatch + tails + carried state, against the
FP32 recurrence and a double-precision CPU reference), `gdn_chunk_parity`, then the prompt yardstick
(`tools/quant_quality.py prompt` + `pcompare`, top-1 and full-vocabulary KL) and the MTP draft-acceptance
arms.  A reordering that moves top-1 is a no.

## Measurement plan

1. **Correctness first, on the smallest thing that can fail**: `hip_gdn_chunked` with
   `STRATA_GDN_CHUNKED=1` against `=0`, then `gdn_chunk_parity`.
2. **Component**: per-layer time of `gdn_recurrence` at 2048 tokens (the phase table's `gdn recurrence`
   column).  The bar is the RTX 5090's 1.81x; anything under ~1.2x on this box is not worth the accuracy
   question.
3. **Engine A/B**: `bench/gfx906/decode-ab.sh`, four interleaved arms (`base, chunk, base, chunk`) with
   `STRATA_PREFILL_TIMING=1`, reading the `gdn recurrence` column - the phase is 199 ms of 2284, so a 2x
   phase win is ~4 % of the chunk, which *is* resolvable on this box (unlike the 1.2 % of the int8 score
   lever).
4. Only then the two gates, then the record under `docs/gfx906-results/`.

## Risks

- **Occupancy**: 55.3 KB of LDS means one block per CU (6 waves of 384 threads).  If that is not enough
  to hide the FP32 dependencies, `GDV = 4` (43 KB, two blocks per CU) is the fallback - same code, one
  constant.
- **The prep's 32 `__syncthreads()`** per block-chunk: 16 key heads x 64 chunks = 1024 blocks per layer,
  ~3200 cycles of barrier each at worst.  If it shows, the solve can go back to a single wave (64 lanes =
  2 value heads per pass) at the cost of one extra pass for the third head.
- **k/q traffic**: `GDV = 8` means each key head's k and q rows are read twice more than the CUDA shape.
  If the phase turns out DRAM-bound rather than issue-bound, `VPK` sharing at `GDV = 4` is the other
  corner of the trade.
- **Accuracy**: reordered FP32 sums may move the 20-prompt top-1/KL and the draft acceptance.  That is a
  gate result, not an implementation detail; if it moves, the lever is rejected like `STRATA_GR_SPLIT`.
