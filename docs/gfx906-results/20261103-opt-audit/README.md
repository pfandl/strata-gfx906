# 2026-11-03 - what is left after the RP port: an env-knob audit (which found one), a corrected phase profile, and the hyper-connection read priced

The [2026-11-02 record](../20261102-rp-prefill/README.md) ended with a claim that needed testing before any more
kernel work: *every remaining named phase is at its ceiling or algorithmically fixed, so prompt processing needs a
kernel-scale project (int8 attention scores) or algorithmic work (the GDN scan), not another env lever*. This audit
tests that claim on the same two-card box, and the claim is **wrong in the most direct way**: the box's biggest lever
is an environment knob that was already in the tree - `STRATA_HIP_PROMPT_F16=1`, whose tip is printed only on gfx103x -
and it is worth **+10.4 %** of prompt processing here (650/670 -> 739.6 tok/s), because the largest phase is running
BF16 GEMMs on a card that has no BF16 instructions.

Everything here is one engine start per arm through [`bench/gfx906/rp-ab.sh`](../../../bench/gfx906/rp-ab.sh), the
same fixed random-word prompt, the four RP flags on in every arm (`BASELINE` was not re-run: the port is the
deployment now). Context 32768, prompt 32700 tokens, `--prefill 2048`, `--kv int8`, `--spec 4`, `--layer-split 24`,
`--resident-budget-gib 20`, `--conversation-cache-mib 4096`, 6 pool workers. Engine: this branch, `c14e3abf` for the
`t1`/`t2` passes (13 commits ahead of `origin/gfx906`) and the same binary rebuilt at `af503f57` for `t3` (the only
change in between is the tip below); built with the flags in `bench/gfx906/rp-check.sh`.

## Three measurement traps this audit had to fix first

All three cost real conclusions in the first pass, so they are worth stating for anyone repeating this.

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

3. **Price a phase against the call the engine makes, and divide it by one stage's work.** The first pass priced the
   hyper-connection read with the bench's `f16` column while the engine runs `Gemm::bf16` (BF16 hipBLAS, 1.8-2.5x
   slower on this card), and divided the phase by both cards' reads (96) instead of one stage's (48), halving the
   per-read cost. Both errors pointed the same way, and together they turned the box's largest remaining lever into a
   "rejected" item - see the priced section below. A bench column is evidence about the engine only if it is the same
   call, and a phase table is interpretable only against the layer count that produced it (the phases sum to the
   printed timeline, and that timeline is one stage's: 24 of the 48 layers, both halves).

With those three fixed, the run-to-run noise is +/-3 % on the end-to-end pp but **+/-0.7 % on the steady-state GPU
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

A third blocked pass (`t3-*`) closed the question the bench opened - *is the hyper-connection read really at its
ceiling* - by running the knob that turns the prompt path's 16-bit GEMMs into FP16 (`STRATA_HIP_PROMPT_F16=1`):

| arm | env | GPU ms/chunk | hc read | pp tok/s |
| --- | --- | ---: | ---: | ---: |
| `t3-rp` | (the four RP flags) | 2694 | 544.9 | 650.0 |
| `t3-f16io` | `+ STRATA_HIP_PROMPT_F16=1` | **2441** | **261.4** | **739.6** |
| `t3-rp2` | (the four RP flags, again) | 2699 | 544.0 | 670.1 |

**-255 ms of GPU timeline (-9.5 %), -283 ms in the hc read alone (-52 %), +10.4 % pp against the better reference
(+12.1 % against their mean).** The references repeat to 0.2 % on the timeline and 0.1 % on the phase, so this is far
outside the noise band. Every other phase is roughly unchanged (qsa attn 421 -> 424, gate/up 409 -> 422, gdn 313 ->
307, qsa proj 156 -> 165): the whole win is the read. The full tables are in `bench/phase-tables.txt` and the arms in
`bench/t3-*`. (This session's references are 1.4 % faster than `t2`'s - 2696 against 2734 ms on the same
configuration - so arms are only compared inside their own pass, as everywhere else in this record.)

`STRATA_MMQ_RESIDENT_SORT_NE=1` is worth one sentence because it was the most promising candidate: it sorts each
chunk's experts by row count so the MMQ groups balance (`mmq_resident_sort_rows`, reachable on gfx906 - unlike
`STRATA_MMQ_GROUP_SORT`, which needs NVIDIA `whole_tiles` and returns false here). Its per-chunk timeline is visibly
the tightest of any arm (2588-2771 ms) but the mean lands 20 ms under the reference, i.e. inside the 36 ms spread
between the two reference runs. It stays off.

`STRATA_HC_Q8=1` is a **decode** knob, and this audit does not disturb that: it shrinks the hyper-connection weights
from `Bf16InF32` (4 B/element) to Q8_0, which frees ~0.95 GiB per card. What it does not do is speed the prompt
path, because `bf16_proj` expands the Q8_0 codes back to bf16 per call (`prefill.cpp`, `hc_q8_to_bf16`), so the GEMM
stays BF16 hipBLAS and the hc read phase measures the same 542-552 ms with and without it - see the next section for
what actually costs there.

## Corrected phase profile (warm, 2048-token chunks, 32700-token prompt)

`t2-rp2`, 2734 ms of steady-state GPU timeline per chunk:

| phase | ms/chunk | share | state in the code (checked, not assumed) |
| --- | ---: | ---: | --- |
| hc read | 546.6 | 20.0 % | one stage's **48 reads**, i.e. **11.39 ms per read** (this table is 24 of the 48 layers - dividing by both stages' 96 halves it and hides the lever). `bf16_proj` -> `Gemm::bf16`: the `Bf16InF32` weight goes straight into `hipblasGemmEx` as **BF16 -> FP32** (no conversion outside the kernel, no hipBLASLt in this build). **Not int8, and not FP16 either.** 2.7 / 4.2 TFLOP/s on the two GEMMs, i.e. 2.2-2.5x *below* the same matrices in FP16. Measured with `STRATA_HIP_PROMPT_F16=1`: **261.4 ms, 10.7 %**, 5.44 ms per read |
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
`Bf16InF32`, with no GGUF blocks). They run through `Gemm::bf16`, which on gfx906 is `hipblasGemmEx` with BF16 inputs
and an FP32 accumulator - gfx906 has no BF16 instructions, so rocBLAS widens every element inside the kernel, and the
two GEMMs run at **2.7 and 4.2 TFLOP/s where the same matrices in FP16 reach 6.5 and 9.0** (priced below).
It is the single biggest phase *and* the one place where this audit's first pass was wrong in its own favour.

## The hyper-connection read, priced (`dense_mmq_bench`: the two hc cases and a BF16 column)

`src/prefill/dense_mmq_bench.cu` gained the two shapes the read actually uses - `hc_*_down` `[10240 -> 320]` and
`hc_*_up` `[320 -> 10240]` - as Q8_0 cases, **and a `bf16` column**: the same product on a resident BF16 weight through
`Gemm::bf16`, which is the call the read makes. An earlier version of this audit compared the int8 paths against the
bench's `f16` column (`Gemm::native`: dequantize to FP16, then hipBLAS FP16) and so priced them against a path the
read never runs. T = 2048, medians (T = 4096 in `bench/dense-mmq-hc-shapes.txt`):

| shape | bf16 = today | f16 (dequant + hipBLAS FP16) | int8 MMQ (+ its quantize) | int8 repacked (+ its quantize) |
| --- | ---: | ---: | ---: | ---: |
| hc down [10240 -> 320] | **5.063 ms (2.7 TF)** | 2.068 ms (6.5 TF): **x2.45** | 1.495 + 0.480 = 1.975 ms (-61 %) | 0.917 + 0.480 = **1.397 ms (-72 %)** |
| hc up [320 -> 10240] | **3.201 ms (4.2 TF)** | 1.491 ms (9.0 TF): **x2.15** | 1.732 + 0.040 = 1.772 ms (-45 %) | not covered (K = 320) |

Two things fall out. First, **BF16 costs 1.8-2.5x on this card on every shape in the bench** (all twelve cases, both
token counts): the same matrices in FP16 are 2.2-2.5x faster, because FP16 has `v_dot2` and BF16 has to be widened to
FP32 - so this is not an int8 question at all. Second, the read's own budget only closes on the bf16 numbers: the
timing table is **one stage's** (24 layers; its phases sum to the 2734 ms timeline, and the gdn phase's 313 ms matches
289 ms of bench-rate GEMM over 24 layers), so 546.6 ms is **48 reads** and one read costs **11.39 ms** -
8.26 ms of BF16 hipBLAS (73 %) plus ~3.1 ms of norm, silu, inject (N = 4, negligible) and mix.

Priced per read against the 2734 ms chunk, from the bench (before the `t3` arm):

| change | per read | per chunk | share |
| --- | ---: | ---: | ---: |
| **the hc weights in FP16** (`Gemm::f16`; bf16 -> f16 is exact, and 16 bits stay 16 bits) | 11.39 -> 6.69 ms | **-226 ms** | **-8.3 %** |
| ... plus the down projection repacked at int8, paying its activation quantize | -> 6.02 ms | -258 ms | -9.4 % |
| ... plus the up projection at int8 as well | -> 6.26 ms | -246 ms | -9.0 % (worse - the up side is slower on int8) |

**Measured**: the `t3-f16io` arm above applies exactly that idea to every 16-bit GEMM on the prompt path, and the read
goes **11.35 -> 5.44 ms per read** (chunk 2696 -> 2441). That is *more* than the bench predicted, and the read's own
budget says why: solving 11.35 = G_bf16 + E and 5.44 = G_f16 + E + conv for the ~1.4 ms of norm/silu/inject/mix
memory traffic gives G_bf16 ~ 9.9 ms and G_f16 ~ 4.0 ms - a ratio of **2.46**, the same 2.45 the bench measures on
`hc down`, but ~20 % slower in absolute terms than the isolated bench (two stages share the memory system).
The error that hid this is trap 3: the phase is one stage's 48 reads, not both stages' 96, and the first pass priced
the int8 alternatives against a benchmark column that the read never runs.

So the largest phase does not need a kernel. What it needs is the *activations* in FP16 as well (`xn16`, `lo16`, and
`xn16` again for the inject projection), which is either the `xf16` writers that the bf16->f16 CUDA path already uses
or a 0.2 ms conversion per read; the weights are already 16-bit, the conversion is exact, and `bf16_to_f16_rows` /
`Gemm::f16` exist for exactly it. Two variants are on the table: the shipped knob as it is (`STRATA_HIP_PROMPT_F16=1`,
measured above, converts each weight per call and moves every 16-bit GEMM on the path to FP16), or the same thing
narrowed to the three hc matrices with a load-time FP16 image. The narrow one keeps the numerics of every other
projection intact and lands the same ~283 ms.

**Neither variant is numerically free**, and that was the gate before any of this ships: `STRATA_HIP_PROMPT_F16=1` is
documented as not bit-identical to the default, and on this arm's fixed random-word prompt it changes the generated
tokens from the third one on (both references produce `259, 76038, 22700` repeating; the FP16 arm produces
`259, 76038, 49433` repeating). The prompt is gibberish, so that is not evidence of damage - it is evidence that the
knob has to go through the acceptance harness rather than being flipped on because it is faster. That harness has now
been run: [the section below](#acceptance-the-gate-run-2026-11-03-same-box) passes it on every metric that exists.

With this commit the engine prints that tip on gfx906 too (`prompt_f16()` in `gemm.cu`; verified in a 4096-token
smoke arm, which prints it once per card). It was gated to gfx103x - which is why the knob was invisible on the one
card where BF16 costs the most.

The int8 line stays as the second-order item: +1.1 % for repacking one matrix, on top of either variant.

## Acceptance: the gate, run (2026-11-03, same box)

Two measurements, both against the default (`STRATA_HIP_PROMPT_F16=0`) on the same pack, the same engine image and the
four RP flags in both arms. Artefacts in `bench/accept/`.

**The batched prompt path's own yardstick** - `tools/quant_quality.py prompt` + `pcompare`: the 5 text segments of
`data/quant_quality/v1` at 256 / 512 / 768 / 1000 tokens (20 prompts), the *full vocabulary's* KL of the next token's
distribution, one engine start per arm with the harness's fixed configuration (experts on the CPU, `--expert-cache 1`):

| run | prompts | top-1 vs default | KL mean | median | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `STRATA_HIP_PROMPT_F16=1` | 20 | **100.00 %** | 0.01185 | 0.00035 | 0.13436 |

Every one of the 20 prompts picks the default path's top token. For scale, on the same yardstick's prompt path in
`docs/QUANTS.md`: the K-quant MMQ kernels against the FP16 path are 0.095 mean / 0.0027 median (UD-Q4_K_XL) and stayed
opt-in over it, and the same quant file on a second backend is 94.5 % top-1 at KL 0.038. This change is ~8x smaller
than the first on both statistics and does not move the top token at all.

**MTP draft acceptance** - through the deployment (`docker compose`, MTP on, `--spec 4`, greedy, three request types),
reading `timings.draft_n_accepted / draft_n` out of each response:

| request | generated | default drafts | accepted | FP16 drafts | accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| docs, 1911-token prompt | 252 / 382 | 134/213 | 62.9 % | 221/307 | **72.0 %** |
| factual, 70-token prompt | 128 / 134 | 75/102 | 73.5 % | 79/114 | 69.3 % |
| code, 1532-token prompt | 600 / 600 | 299/408 | 73.3 % | 311/425 | 73.2 % |
| **all three** | | **508/723** | **70.3 %** | **611/846** | **72.2 %** |

Aggregate acceptance rises 1.9 points over 1,569 tested drafts. Per request it moves both ways, which is what one
greedy sample each can resolve - the drafting itself is deterministic (the same arm's counts repeat exactly across
runs, `mtp2-*`) - so the read is "not reduced", not "+1.9 points of quality". The earlier not-bit-identical
observation is confirmed as wording: the `docs` answer opens "Strata distributes the model across GPU, RAM, and SSD"
where the default opens "Strata splits work across...", with the same three points behind it.

**Prompt speed on these requests is not the number to read**: 280 vs 266 tok/s on the 1911-token prompt and 205 vs 323
on the code prompt, and an earlier pass of the same 1911-token request gave 125 vs 217 tok/s and then 146 vs 113 - the
first request after a container restart streams the expert tiers and swings by ±40 % on this box. The controlled
number stays the blocked A/B/A above (`t3`, warm, steady state, references within 0.2 %): **650/670 -> 739.6 tok/s**.

Verdict: the gate passes on the metrics that exist - 100 % top-1, KL an order of magnitude under the opt-in K-quant
precedent, MTP acceptance not reduced. That is why the deployment's `docker-compose.yml` now sets
`STRATA_HIP_PROMPT_F16=1` (with `=0` as the opt-out) and the engine prints the gfx906 tip where it used to print
nothing. The full-vocabulary logits (24 MB) stay on the box at `data/rp-opt/accept/arm{0,1}`; everything else is in
`bench/accept/`.

## What is actually left, re-ranked

1. **`STRATA_HIP_PROMPT_F16=1`: +10.4 % pp, measured, in the tree, and now on in the deployment.** The knob a
   gfx103x-only tip advertises takes the hc read from 544.5 to 261.4 ms/chunk and the timeline from 2696 to 2441 ms
   (`t3` pass above). Its gate was numerical, not engineering, and it **passed** (100 % top-1, KL 0.0119 / 0.00035,
   MTP acceptance 70.3 -> 72.2 %; `bench/accept/` and the acceptance section above). The narrower variant - a load-time
   FP16 image of `hc_*_down`, `hc_*_up` and `hc_*_inject` with `Gemm::f16` - is now a fallback, not the plan: it lands
   the same ~283 ms and leaves every other projection's numerics alone, if the wider knob ever has to be narrowed.
2. **qsa prompt attention (15.6 %, 428 ms/chunk).** The only block still on a structurally wrong kernel: no
   gfx906 prompt kernel exists, so `prefill.cpp` loops `qsa_decode_attn_batch` over 32-query batches and every query
   re-reads its KV window. A query-blocked tiled kernel that stages K/V once per 64 cells is the lever; the int8
   scores question stays separate (the *stored* cache can stay FP16 - the int8 that matters for MTP acceptance is
   storage, not a transient on-the-fly quantization). Estimated 150-250 ms/chunk = **+6-10 %**.
3. **Expert gather/dequant (4.7 %, 128 ms) and the chunk-to-chunk variance it carries.** All 512 experts of a layer
   are touched every chunk (10 of 512 x 2048 tokens), so a repacked image would hit ~100 % - but the repacked form is
   ~2.5x the GGUF bytes and only ~4 GiB/card is spare. Needs a hot-expert LRU or a repacked-layout decode GEMV.
   ~+4 %.
4. **The hc read's down projection at int8 (+1.1 % on top of (1))**: 1.397 ms per read against 2.068 in FP16, the
   quantize eating most of the repack's +48 %. The up projection stays 16-bit under every option (int8 is *slower*
   there).
5. **GDN recurrence (5.0 %, 136 ms).** Algorithmically open (chunked WY scan; exists for CUDA and gfx11, not
   gfx906), research-grade. ~+2-4 %.
6. Everything else is at its measured ceiling: the expert GEMMs run at ~16-17 TFLOP/s of int8 (the bench's dense
   shapes reach 18-19 TOPS repacked), the dense projections at 12-19 TOPS, and the remaining glue is 1.8 %.

The 2026-11-02 claim this audit set out to test fails at its own first sentence. There *is* an env lever left, it is
the largest one on the box, and it shipped all along - with a tip that only gfx103x prints, because
`prompt_f16()` assumes the FP16 advantage is an RDNA3 property. On a card with no BF16 instructions it is worth more,
not less: **the reason the phase looked like an int8/FP16 ceiling was that nobody had measured the BF16 call it
actually makes.**

## Reproducing

    # one arm (warm the page cache first: run it twice, read the second)
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
      -v "$PWD/data":/data -v "$PWD/rp-build":/work -v "$PWD/strata-gfx906":/repo:ro \
      -e MODEL=IQ3_S -e PACK=/data/packs/iq3_s -e OUT=/data/rp-opt -e CTX=32768 \
      -e PROMPT_TOKENS=32700 -e PREFILL=2048 -e STRATA_PREFILL_TIMING=1 \
      strata-gfx906:local /repo/bench/gfx906/rp-ab.sh t2-rp RP

    # a knob: the same line plus -e STRATA_MMQ_RESIDENT_SORT_NE=1 (and a second plain arm after it)

    # the pass-3 arm: the same line plus -e STRATA_HIP_PROMPT_F16=1, bracketed by two plain arms

    # the per-chunk phase table of any arm
    python3 docs/gfx906-results/20261103-opt-audit/bench/parse-phases.py data/rp-opt/<arm>

    # the hc shapes on the four dense paths (f16 | bf16 | int8 mmq | int8 repacked), at 2048 and 4096 tokens
    for T in 2048 4096; do STRATA_MMQ_KQ=1 build-906/dense_mmq_bench $T; done

`bench/t1-*`, `bench/t2-*` and `bench/t3-*` keep `summary.txt`, `info.txt` and the engine's `stderr.log` (the phase
tables are in their `stderr.log`; `bench/phase-tables.txt` is the parsed form). `bench/dense-mmq-hc-shapes.txt` is the
whole bench output for both token counts, with the `bf16` column that started this.

