# 2026-11-03 - what is left after the RP port: an env-knob audit, a corrected phase profile, and the hyper-connection read priced

The [2026-11-02 record](../20261102-rp-prefill/README.md) ended with a claim that needed testing before any more
kernel work: *every remaining named phase is at its ceiling or algorithmically fixed, so prompt processing needs a
kernel-scale project (int8 attention scores) or algorithmic work (the GDN scan), not another env lever*. This audit
tests that claim on the same two-card box, and finds one phase the claim had wrong.

Everything here is one engine start per arm through [`bench/gfx906/rp-ab.sh`](../../../bench/gfx906/rp-ab.sh), the
same fixed random-word prompt, the four RP flags on in every arm (`BASELINE` was not re-run: the port is the
deployment now). Context 32768, prompt 32700 tokens, `--prefill 2048`, `--kv int8`, `--spec 4`, `--layer-split 24`,
`--resident-budget-gib 20`, `--conversation-cache-mib 4096`, 6 pool workers. Engine: this branch at `c14e3abf`
(13 commits ahead of `origin/gfx906`), built with the flags in `bench/gfx906/rp-check.sh`.

## Two measurement traps this audit had to fix first

Both cost real conclusions in the first pass, so they are worth stating for anyone repeating this.

1. **The first arm of a session is not comparable to the ones after it.** The GGUF is 62 GB and the box has 68 GB of
   RAM, so the *first* engine start reads it from the SSD and every later start finds it in the page cache. In the
   first pass (arms `t1-*`) the pp rose monotonically 514.0 -> 595.1 -> 616.9 -> 672.5 tok/s across arms that were
   *not* monotonically better. Re-running the plain RP arm after four warm arms (`t2-rp`, `t2-rp2`) gives 643.5 and
   664.3 tok/s - the same configuration, +25 % over its own cold start. **Every A/B on this box needs a warm-up arm
   first, and the blocked A/B/A below (rp, knob, rp) is the minimum.**
2. **A dense int8 phase absorbs the host's lateness.** `STRATA_PREFILL_TIMING` charges the gap between two marks to
   the phase of the first, so a phase where the GPU waits for the host to feed it *looks* slow. The cold-cache arm
   showed `qsa proj` 261 ms, `gdn` 481 ms and `dequant` 329 ms; the warm arm of the *same* configuration shows 153,
   333 and 128 ms. Reading the cold numbers as kernel costs would have sent the work at the wrong kernels.

With those two fixed, the run-to-run noise is +/-3 % on the end-to-end pp but **+/-0.7 % on the steady-state GPU
timeline** (the mean of the per-chunk `GPU timeline` line, first chunk excluded), which is what the verdicts below
use.

## The arms

Warm pass (`t2-*`, in this order, A/B/A around the knobs), steady-state GPU timeline per 2048-token chunk:

| arm | env | GPU ms/chunk | pp tok/s | verdict |
| --- | --- | ---: | ---: | --- |
| `t2-rp` | (the four RP flags) | 2770 | 643.5 | reference |
| `t2-sortne` | `+ STRATA_MMQ_RESIDENT_SORT_NE=1` | **2714** | 669.2 | -2.0 % GPU, inside noise: **not a win** |
| `t2-w32` | `+ STRATA_PF_RP_W=32` | 2929 | 620.7 | **+7.1 % GPU: a regression, rejected** |
| `t2-hcq8` | `+ STRATA_HC_Q8=1` | 2864 | 630.3 | **+4.8 % GPU: rejected for prompt** (see below) |
| `t2-rp2` | (the four RP flags, again) | 2734 | 664.3 | the noise band: 2734 vs 2770 for one config |

The cold first pass (`t1-*`) is kept in `bench/` for the record, not for comparison: `t1-rp` 514.0 tok/s is the same
configuration as `t2-rp` 643.5, and `t1-sortne` 672.5 vs `t2-sortne` 669.2 shows the knob's apparent +4 % was the
page cache, not the knob.

`STRATA_MMQ_RESIDENT_SORT_NE=1` is worth one sentence because it was the most promising candidate: it sorts each
chunk's experts by row count so the MMQ groups balance (`mmq_resident_sort_rows`, reachable on gfx906 - unlike
`STRATA_MMQ_GROUP_SORT`, which needs NVIDIA `whole_tiles` and returns false here). Its per-chunk timeline is visibly
the tightest of any arm (2588-2771 ms) but the mean lands 20 ms under the reference, i.e. inside the 36 ms spread
between the two reference runs. It stays off.

`STRATA_HC_Q8=1` is a **decode** knob, and this audit does not disturb that: it shrinks the hyper-connection weights
from `Bf16InF32` (4 B/element) to Q8_0, which frees ~0.95 GiB per card. What it does not do is speed the prompt
path, because `bf16_proj` expands the Q8_0 codes back to bf16 per call (`prefill.cpp`, `hc_q8_to_bf16`) and the hc
read phase measures the same 542-552 ms with and without it - see the next section.

## Corrected phase profile (warm, 2048-token chunks, 32700-token prompt)

`t2-rp2`, 2734 ms of steady-state GPU timeline per chunk:

| phase | ms/chunk | share | state in the code (checked, not assumed) |
| --- | ---: | ---: | --- |
| hc read | 546.6 | 20.0 % | `bf16_proj`: `Bf16InF32` weight -> bf16->f16 conversion per call -> hipBLAS FP16. **Not int8.** ~4.7 TFLOP/s effective |
| qsa attn | 427.6 | 15.6 % | gfx906 has no prompt-attention kernel: the decode kernel runs in 32-query batches |
| gemm gate/up | 414.0 | 15.1 % | int8, repacked GEMM (RP) |
| gdn (qkv+gate) | 332.7 | 12.2 % | int8 dense MMQ/RP |
| gemm down | 189.7 | 6.9 % | int8, repacked GEMM |
| qsa proj | 153.1 | 5.6 % | int8 dense (K/V deliberately on FP16) |
| gdn recurrence | 136.0 | 5.0 % | the sequential delta-net scan (chunked WY is CUDA/gfx11 only) |
| gdn out proj | 128.1 | 4.7 % | int8 dense |
| dequant | 127.9 | 4.7 % | the per-chunk expert gather + IQ3_S -> int8 repack |
| router+shared | 112.4 | 4.1 % | int8 dense |
| qsa select | 65.1 | 2.4 % | SIMT bf16 block scorer on gfx906 |
| host grouping | 34.4 | 1.3 % | host-side expert grouping |
| hc write / combine / embed / gdn conv / gather / wait copy | 48.7 | 1.8 % | small |

The 2026-11-02 section's row for the hyper-connection read said *"dense, int8 through `STRATA_DENSE_*`, at the int8
instruction ceiling (~11 TF/s)"*. That is **wrong**: the hc projections never enter `Gemm::native` (they are
`Bf16InF32` with no GGUF blocks) and they run at 4.7 TFLOP/s of FP16, i.e. ~18 % of the card's FP16 peak. It is the
single biggest phase, so the correction matters even though the conclusion below turns out to be the same.

## The hyper-connection read, priced (`dense_mmq_bench`, the two new hc cases)

`src/prefill/dense_mmq_bench.cu` gained the two shapes the read actually uses - `hc_*_down` `[10240 -> 320]` and
`hc_*_up` `[320 -> 10240]` - as Q8_0 cases, so the bench's three paths compare directly: f16 = dequant + hipBLAS
FP16 (today, including the conversion), mmq = int8 dot products, rp = the repacked int8 GEMM. T = 2048, medians:

| shape | f16 (today) | int8 MMQ (+quantize) | int8 repacked (+quantize) |
| --- | ---: | ---: | ---: |
| hc down [10240 -> 320] | 2.147 ms (6.3 TF) | 1.495 + 0.480 = 1.975 ms (+9 %) | 0.928 + 0.480 = **1.408 ms (+52 %)** |
| hc up [320 -> 10240] | 1.520 ms (8.8 TF) | 1.742 + 0.038 = 1.780 ms (**-15 %**) | not covered (K = 320) |

The read is 546.6 ms per chunk over 48 layers x 2 halves = 5.69 ms per layer-half, against 3.67 ms of GEMM in the
bench (the rest is the norm, the silu, the inject projection and the mix). The best int8 mix - repacked down plus
MMQ up - saves 1.36 ms of those 5.69, i.e. **~2.4 % of the chunk (~130 ms), for a project that has to repack a
K=320 shape the rp tiles do not cover and convert the resident weights.** That is not worth doing, and it is the
reason this audit's headline is a *rejection*: the phase that looked like the biggest leftover is at its practical
ceiling for this shape.

## What is actually left, re-ranked

1. **qsa prompt attention (15.6 %, 428 ms/chunk).** The only block still on a structurally wrong kernel: no
   gfx906 prompt kernel exists, so `prefill.cpp` loops `qsa_decode_attn_batch` over 32-query batches and every query
   re-reads its KV window. A query-blocked tiled kernel that stages K/V once per 64 cells is the lever; the int8
   scores question stays separate (the *stored* cache can stay FP16 - the int8 that matters for MTP acceptance is
   storage, not a transient on-the-fly quantization). Estimated 150-250 ms/chunk = **+6-10 %**.
2. **Expert gather/dequant (4.7 %, 128 ms) and the chunk-to-chunk variance it carries.** All 512 experts of a layer
   are touched every chunk (10 of 512 x 2048 tokens), so a repacked image would hit ~100 % - but the repacked form is
   ~2.5x the GGUF bytes and only ~4 GiB/card is spare. Needs a hot-expert LRU or a repacked-layout decode GEMV.
   ~+4 %.
3. **GDN recurrence (5.0 %, 136 ms).** Algorithmically open (chunked WY scan; exists for CUDA and gfx11, not
   gfx906), research-grade. ~+2-4 %.
4. Everything else is at its measured ceiling: the expert GEMMs run at ~16-17 TFLOP/s of int8 (the bench's dense
   shapes reach 18-19 TOPS repacked), the dense projections at 12-19 TOPS, and the remaining glue is 1.8 %.

So: the 2026-11-02 conclusion survives, with one phase moved from "at its ceiling" to "at its FP16 ceiling, and
int8 would not help either", and with the attention block confirmed as the next kernel-scale project.

## Reproducing

    # one arm (warm the page cache first: run it twice, read the second)
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
      -v "$PWD/data":/data -v "$PWD/rp-build":/work -v "$PWD/strata-gfx906":/repo:ro \
      -e MODEL=IQ3_S -e PACK=/data/packs/iq3_s -e OUT=/data/rp-opt -e CTX=32768 \
      -e PROMPT_TOKENS=32700 -e PREFILL=2048 -e STRATA_PREFILL_TIMING=1 \
      strata-gfx906:local /repo/bench/gfx906/rp-ab.sh t2-rp RP

    # a knob: the same line plus -e STRATA_MMQ_RESIDENT_SORT_NE=1 (and a second plain arm after it)

    # the per-chunk phase table of any arm
    python3 bench/parse-phases.py <arm dir>

    # the hc shapes on the three dense paths
    STRATA_MMQ_KQ=1 build-906/dense_mmq_bench 2048

`bench/t1-*` and `bench/t2-*` keep `summary.txt`, `info.txt` and the engine's `stderr.log` (the phase tables are in
their `stderr.log`; `bench/phase-tables.txt` is the parsed form). `bench/dense-mmq-hc-shapes.txt` is the whole bench
output for both token counts.

