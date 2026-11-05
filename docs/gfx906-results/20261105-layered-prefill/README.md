# 2026-11-05 - the prompt-prefill ladder, cut into three layers

The [2026-11-02 record](../20261102-rp-prefill/README.md) ends with `549.0 -> 769.7 tok/s (+40 %)` and
the [2026-11-03 audit](../20261103-opt-audit/README.md) prices the knobs on top of it. Both are true and
both mislead as a *change under review*, because that ladder climbs over four different things at once:

1. **levers the engine already ships** - `STRATA_HIP_PROMPT_F16` (opt-in, #835) and the two gfx906
   attention switches, which are upstream PRs [#1661](https://github.com/Niko1221/Strata/pull/1661)
   (`STRATA_GFX906_ATTN_QUERY_SWIZZLE`) and [#1718](https://github.com/Niko1221/Strata/pull/1718)
   (`STRATA_GFX906_ATTN_REDUCE12`), merged off by default. None of that is code in this branch: on the
   base commit the switches live in `src/kernels/cuda/qsa_decode_attn.cu` and the FP16 path in
   `src/prefill/gemm.cu:581`, and the deployment turns them on by environment, not by a patch.
2. **a port of someone else's kernel stack** - the seven repacked-int8 commits authored by
   **Janus Reith** (`3d2d65f2` … `e0fb5093`).
3. **one kernel written here for this card** - `src/kernels/cuda/qsa_prompt_attn_gfx906.cu`.
4. **a measurement artefact** - the `549.0` baseline was the cold first arm of its session at
   8192-token chunks; the same stock build reads 448 tok/s warm at the chunk size it picks for itself,
   and 532 tok/s at the deployed 2048.

A reviewer cannot tell which of the +40 % they are being asked to accept. So this record re-cuts the
ladder into **three layers and measures each layer against the layer below it**, in one session, on one
prompt, with one harness - and it reports the box's own noise band instead of a figure carried over
from an earlier session.

| layer | what it is | where the code comes from | measured against |
| --- | --- | --- | --- |
| 1 | every lever the engine already had, enabled | base commit `048effa1` on `webzone/strata-gfx906@gfx906`, env only | as shipped |
| 2 | the repacked-int8 prompt stack | Janus Reith (`janus-reith/strata-gfx906`), seven commits, ported | layer 1 |
| 3 | the gfx906 prompt-attention kernel | this branch | layer 2 |

Layer 2 is **Janus Reith's** work, from <https://github.com/janus-reith/strata-gfx906>: seven commits,
carried unmodified with his authorship. His repack kernels derive from `mx-llama.cpp`, and that MIT
notice travels with them as `src/prefill/rp/LICENSE.mx-llama.cpp`. None of his fork is merged upstream,
and neither is the fork base these layers are measured on - see [`docs/GFX906.md`](../../GFX906.md) for
what the fork contains beyond upstream.

## The box, the prompt, the harness

Two Instinct MI50 32 GB (gfx906), 68 GB of RAM, IQ3_S (~84 GB on disk, ~62 GB resident). One arm = one
engine start inside the runtime image with the layer's build mounted at `/work`
([`bench/gfx906/rp-ab.sh`](../../../bench/gfx906/rp-ab.sh)): the same seeded random-word prompt of
**32,700 tokens**, `--max-context 32768`, `--prefill 2048`, `--kv int8`, `--spec 4`, `--layer-split 24`,
`--resident-budget-gib 20`, `--conversation-cache-mib 4096`, 6 pool workers, and
`STRATA_PREFILL_TIMING=1` so every arm also prints its per-chunk phase table.

Two numbers per arm:

- the engine's own end-to-end line, `prompt N tokens = 0 reused + N read in M ms (X tok/s)` - what a
  user sees;
- the phase table's `GPU timeline` per chunk - the mechanism, and the number that says whether a lever
  moved the GPU or the host.

`bench/parse-phases.py` prints both plus the medians of the phases that move between layers, over
full-size chunks, first chunk dropped, chunks slower than 1.5x the median dropped as outliers;
`bench/phase-diff.py <arm-a> <arm-b>` prints the two arms' full 20-phase tables side by side, sorted by
the delta, which is how each layer below is attributed. Every arm's raw stdout/stderr is in
`bench/<arm>/` and the session's harness logs (which arm ran when, with which environment) in
`bench/session/`; the prompt file is not kept because the harness regenerates it from a fixed seed.

**The noise band of this session, measured, not assumed.** Five arms of *one* configuration (layer 1 with
everything on) read 570.4 / 633.5 / 649.8 / 670.0 / 676.7 tok/s - an 18.6 % spread end to end - while
their GPU medians move only 2666-2752 ms (3.2 %). The 2026-11-03 audit reported +/-0.3 % on its
references; that did not hold here (this session crossed a container kill that evicted part of the 62 GB
GGUF, and single-chunk outliers appear in almost every arm). **Every verdict below therefore rests on
interleaved A/B/A arms plus the phase table, never on a single arm**, and a difference smaller than the
band is reported as no effect.

## Layer 1 - every lever the engine already has (base commit `048effa1`)

The binary is the base commit compiled with the flags of
[`bench/gfx906/rp-check.sh`](../../../bench/gfx906/rp-check.sh); **no source from this branch is in it**.
Only the environment changes between arms.

| arm | what it adds | pp tok/s | GPU ms/chunk | hc read | qsa attn | gdn |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `L1-warm` | warm-up arm (reads the 62 GB GGUF off the SSD), discarded | 460.1 | 11062 | 2117 | 1680 | 1717 |
| `L1-stock` | **nothing** - as shipped: `--prefill auto`, no env | 448.2 | 11032 | 2115 | 1677 | 1711 |
| `L1-geo` | `--prefill 2048` (the deployed chunk) | 532.4 | 3239 | 555 | 421 | 492 |
| `L1-f16` | `+ STRATA_HIP_PROMPT_F16=1` | 628.6 | 2809 | 269 | 420 | 390 |
| `L1-sw` | `+ STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1` | 573.8 | 3070 | 553 | 304 | 493 |
| `L1-all` | geometry + the FP16 knob + both switches | 633.5 | 2728 | 268 | 306 | 390 |
| `L1-all2` | the same configuration, again | **676.7** | **2666** | 270 | 304 | 391 |

**The chunk size is the first lever, and the engine says why.** At `--prefill auto` this context picks
8192, and the same run warns:

> `a prompt chunk of 8192 tokens lends 1891 cache slots to the prompt path, but only 1690 of them fit
> in RAM: the others are read from the SSD on every chunk`

At 2048 it borrows 568 slots and every one of them stays in RAM. That is the whole +18.8 % of
`L1-stock -> L1-geo`: not a faster kernel, fewer SSD reads. (It also retires the older reading that the
chunk size was worth +1.3 % - that comparison was made with the port and the knobs already on, where
the lend fits either way. Both readings are in [20261104-chunk-sweep](../20261104-chunk-sweep/README.md).)

**`STRATA_HIP_PROMPT_F16=1` is worth +18.1 %, and the phase table says where it goes**: the
hyper-connection read 555 -> 269 ms/chunk (-51 %) and `gdn` 492 -> 390 ms (-21 %). gfx906 has no BF16
instructions, so the prompt path's 16-bit GEMMs run BF16 hipBLAS; the knob moves them to FP16.

**The two upstream attention switches are worth +7.8 %**: `qsa attn` 421 -> 304 ms/chunk (-28 %), and
nothing else moves.

**They compose.** Adding the phase deltas to the geometry arm predicts
`3239 - 286 (hc read) - 102 (gdn) - 117 (qsa attn) = 2734 ms`; the arms that turn both on measure 2728
and 2666. The end-to-end numbers agree inside the noise band: `532.4 x 3239/2809 x 3239/3070` = 678
predicted, 633.5 and 676.7 measured.

Layer 1's baseline is therefore **geometry 2048 + `STRATA_HIP_PROMPT_F16=1` + both attention switches =
633.5 / 649.8 / 670.0 / 676.7 tok/s over the five arms that ran it (median 649.8; `L1-all3`'s 570.4 is
the low outlier of the same configuration)** - **+45 % over the same binary as shipped (448.2 tok/s)**,
none of it code from this branch.

### The two levers never tried on this box, and what they turned out to be

The 2026-11-03 audit found no env lever left, but it priced its candidates *on top of the port* and
inside a single pass. Two are worth re-testing at this baseline, because this box is a two-card layer
split on gfx906:

| arm | lever | pp tok/s | GPU ms/chunk | verdict |
| --- | --- | ---: | ---: | --- |
| `L1-sortne` | `STRATA_MMQ_RESIDENT_SORT_NE=1` | 670.0 | 2684 | no effect |
| `L1-sortne2` | the same | 647.4 | 2703 | |
| `L1-sortne3` | the same, closing the bracket | 595.1 | 2722 | |
| `L1-nosort` | the layer-1 baseline, bracketed | 670.0 | 2692 | reference |
| `L1-pipe` | `STRATA_PREFILL_PIPE=1` ([`docs/MULTI_GPU.md`](../../MULTI_GPU.md), the layer-split prompt pipeline) | 500.2 | 2677 | **rejected: -21 % end to end** |

`STRATA_MMQ_RESIDENT_SORT_NE=1` sorts each chunk's experts by row count so the MMQ groups balance.
Arm-by-arm it first looked like +5.8 % (`670.0` against `633.5`) - which is exactly the trap the audit
warned about, because the reference arm that followed it read `676.7`. With the bracket around it the
end-to-end numbers are `670.0 / 647.4 / 595.1` against the baseline's `633.5 / 676.7 / 670.0`, i.e. the
two ranges overlap completely, and the GPU medians are 2703 against 2692 ms (0.4 %). **No phase moves**:
`hc read` 269-270, `qsa attn` 304-305, `gate/up` 541-543 with and without the flag. It stays off - the
audit's verdict, now with the bracket that proves it.

## Layer 2 - the repacked-int8 port (seven commits by Janus Reith)

The seven commits under test, in order, are `3d2d65f2` (IQ4_NL gate/up experts through the repacked
GEMM), `464dbb15` (`STRATA_PF_RP=1`: expert products), `2d8254dd` (`STRATA_DENSE_MMQ=1`: dense
projections on the int8 MMQ path), `57e47e1a` (`STRATA_DENSE_RP=1`), `7864b43d` (a per-product choice
for the down projection), `238a6d23` (IQ4_XS, `STRATA_DENSE_MMQ_PROFILE`), `e0fb5093` (IQ3_XXS/IQ3_S
gate/up, which is what this pack's experts are). **They are ported, not written here**: authorship is
his, the commit messages are his, and the four knobs are the deployment's environment, exactly as
above. This layer is measured as one step, with each commit's own bench
([`src/prefill/rp_expert_bench.cpp`](../../../src/prefill/rp_expert_bench.cpp),
[`src/prefill/dense_mmq_bench.cu`](../../../src/prefill/dense_mmq_bench.cu)) as the per-kernel evidence.

| arm | configuration | pp tok/s | GPU ms/chunk |
| --- | --- | ---: | ---: |
| `L1-all` / `L1-all2` / `L1-all3` / `L1-all4` / `L1-nosort` | layer 1 (reference, five arms) | 633.5 / 676.7 / 570.4 / 649.8 / 670.0 | 2728 / 2666 / 2752 / 2690 / 2692 |
| `L2-rp` | `+ STRATA_PF_RP=1 STRATA_DENSE_MMQ=1 STRATA_DENSE_RP=1 STRATA_MMQ_KQ=1` | 729.8 | 2358 |
| `L2-rp2` | the same | 738.2 | 2314 |
| `L2-rp3` | the same | 760.9 | 2318 |

**+13.6 % end to end (median 649.8 -> 738.2 tok/s) and -13.9 % of the GPU timeline (2692 -> 2318 ms per
2048-token chunk).** The phase table says where, and it is not one kernel but five phases:

| phase | layer 1 | layer 2 | delta | share of the chunk |
| --- | ---: | ---: | ---: | ---: |
| gemm gate/up | 542 | 410 | **-132** | -4.9 % |
| gdn (qkv+gate) | 391 | 304 | **-87** | -3.2 % |
| gdn out proj | 199 | 124 | **-75** | -2.8 % |
| gemm down | 245 | 187 | **-58** | -2.2 % |
| qsa proj | 196 | 165 | **-31** | -1.2 % |
| hc read | 270 | 260 | -10 | -0.4 % |
| router+shared | 127 | 118 | -9 | -0.3 % |
| dequant | 93 | 128 | **+35** | +1.3 % |
| qsa select | 67 | 76 | +9 | +0.3 % |
| host grouping / combine | 28 / 13 | 33 / 17 | +5 / +4 | +0.3 % |
| `qsa attn`, `gdn recurrence`, everything else | | | 0 | 0 % |
| **GPU timeline** | **2689** | **2352** | **-337** | **-12.5 %** |

The five int8 GEMM phases give up 383 ms and the per-chunk expert repack (`dequant`) takes 35 ms back -
the port pays for itself 11 times over inside one chunk. Two things are worth noting because they were
easy to get wrong earlier: `qsa attn` does **not** move (the attention is not part of this port, and the
two switches from layer 1 already took their 117 ms), and the `dequant` increase is the honest cost of
repacking IQ3_S into the GEMM's layout per chunk.

This reproduces the 2026-11-02 number (+12.7 % at the same chunk size) on a *different* baseline - that
run compared against a bare stock build at a 65,500-token prompt, this one against layer 1 at 32,700
tokens with the FP16 knob and both switches already on. The port's gain does not depend on either.

### Interaction: the port and the FP16 knob are orthogonal, not additive by luck

`L2-nof16` is the port without `STRATA_HIP_PROMPT_F16` (both attention switches on):

| arm | pp tok/s | GPU ms/chunk | hc read | gate/up |
| --- | ---: | ---: | ---: | ---: |
| `L2-nof16` | 619.9 | 2594 | **541** | 408 |
| `L2-rp` / `L2-rp2` / `L2-rp3` | 729.8 / 738.2 / 760.9 | 2358 / 2314 / 2318 | **260** | 409-413 |

The port's own phases are identical in both configurations (`gate/up` 408 against 409-413, `down` 186
against 186, `gdn` 312 against 304-306) and the whole difference is the hyper-connection read, 541
against 260 ms: **the FP16 knob is worth +19.1 % on top of the port** (median 738.2 against 619.9), the
same order as the +18.1 % it was worth in layer 1. The two levers touch different GEMMs, so the ladder
multiplies them: `649.8 x 1.18 x 1.14` is where layer 2 lands.

## Layer 3 - the prompt-attention kernel (this branch)

This is the one layer whose code is written here: `src/kernels/cuda/qsa_prompt_attn_gfx906.cu` plus the
dispatch in `qsa_prompt_attn.cu` / `qsa_prompt_attn.hpp`, opt-in as `STRATA_PROMPT_ATTN_GFX906=1` (the
deployment sets it). The layer-2 binary and the layer-3 binary differ by nothing else.

Seven arms, interleaved, all with the layer-1 baseline and the port on:

| arm | `STRATA_PROMPT_ATTN_GFX906` | pp tok/s | GPU ms/chunk | qsa attn |
| --- | --- | ---: | ---: | ---: |
| `L3-ko` | 0 | 682.3 | 2340 | 305.5 |
| `L3-k` | 1 | 751.1 | 2300 | 274 |
| `L3-k2` | 1 | 760.3 | 2295 | 275 |
| `L3-ko2` | 0 | 702.5 | 2298 | 305 |
| `L3-ko3` | 0 | 665.5 | 2341 | 305 |
| `L3-k3` | 1 | 717.1 | 2270 | 272.5 |
| `L3-ko4` | 0 | 769.4 | 2314 | 304.5 |

**The phase table is unambiguous and the end-to-end number is not.** `qsa attn` reads 304-305.5 ms/chunk
in all four kernel-off arms and 272.5-275 in all three kernel-on arms - **-31 ms per 2048-token chunk, a
10 % cut of that phase and 1.4 % of the chunk** (`2327 -> 2295` ms on the chunk medians) - and every other
phase is identical to the millisecond:

| phase | `L3-ko2` | `L3-k2` | delta |
| --- | ---: | ---: | ---: |
| qsa attn | 305 | 275 | **-30** |
| qsa proj | 165 | 164 | -1 |
| dequant | 129 | 126 | -3 |
| host grouping | 30 | 28 | -2 |
| gemm gate/up | 413 | 414 | +1 |
| everything else | | | 0 |
| GPU timeline (median) | 2298 | 2289 | -9 |

End to end the same seven arms read **665.5-769.4 tok/s with the kernel off and 717.1-760.3 with it on** -
overlapping ranges, a median of 692.4 against 751.1. That looks like +8.5 % and it is not: the GPU-side
medians differ by 1.4 % (2327 against 2295), and the pp numbers are dominated by the 2.5-13.8 s of
per-prompt work the engine does outside its per-chunk timing (see `L1-pipe`, where that gap is 13.8 s and
the GPU timeline is unchanged). **So this layer's honest number is the phase delta, -1.4 % of a chunk
(2327 -> 2295 ms), and its end-to-end contribution is inside this box's noise band.** At 32,700 tokens a
1.4 % chunk saving
is ~0.4 s out of ~43 s; nothing on this box can resolve that in one arm.

**This corrects the 2026-11-03 record.** That one reported "-2.9 to -4.1 % of the chunk" from chunk walls of
2362 -> 2255 ms, on two arms per side. Two things say it is too large. Its **own** phase measurement there
was -40 ms of `qsa attn`, which is 1.7 % of a 2340 ms chunk, not 3-4 % - a chunk-level figure twice its own
phase delta needs a second mechanism, and there is none in the phase table. And with seven arms here the
chunk-level delta is `2327 -> 2295` ms = **-32 ms, -1.4 %**, which is the phase delta and nothing more. The
kernel is real, its size is 1.3-1.4 % of the chunk at this prompt size, and the 4 % reading was two fast
draws against a quiet reference.

**The quality gate is unchanged** and was run on exactly this kernel code: top-1 20/20 (100 %) against
the default path at KL 0.01364 / 0.00062 on the 20-prompt yardstick, and MTP draft acceptance 72.2 -> 72.0 %
over 1,550 drafts, with both reference arms reproducing their counts and their text exactly - see
[20261103-qsa-prompt-attn](../20261103-qsa-prompt-attn/README.md).




`STRATA_PREFILL_PIPE=1` is the more interesting negative: the GPU timeline is **identical** (2677 ms
against 2666-2728) while the end-to-end number collapses to 500.2 tok/s. The pipeline overlaps stage 1's
read of chunk *c-1* with stage 0's read of chunk *c*, which is the right shape for a stream of prompts
and the wrong shape for one 32,700-token prompt: each chunk now waits for both stages before the next
starts, and the host staging that `--prefill 2048` was hiding behind the GPU becomes visible. Two cards,
a layer split and a single long prompt is not the case it was written for.

hold here (this session crossed a container kill that evicted part of the 62 GB GGUF, and the audit's
arms ran back-to-back inside one warm window). **Every verdict below therefore rests on interleaved
A/B/A arms plus the phase table, never on a single arm**, and a difference smaller than the band is
reported as no effect.

## The ladder

All at a 32,700-token prompt, context 32768, `--prefill 2048`, medians of the arms named above:

| step | pp tok/s | GPU ms/chunk | gain |
| --- | ---: | ---: | --- |
| **as shipped** - base commit, no env, `--prefill auto` (8192) | 448.2 | 11032 | - |
| **layer 1** - every lever the engine already had | 649.8 (633.5-676.7) | 2692 | **+45.0 %** |
| **layer 2** - the ported repacked-int8 stack | 738.2 (729.8-760.9) | 2318 | **+13.6 %** |
| **layer 3** - the gfx906 prompt-attention kernel | 751.1 (717.1-760.3) | 2295 | -1.4 % GPU (pp inside the band) |

**448.2 -> 751.1 tok/s is +67.6 % end to end. Of that, +15 % is code under review (the port, +13.6 %, and
the kernel, -1.4 % of the GPU timeline); the other +45 % is configuration the engine already supported** -
one chunk-geometry decision, one opt-in GEMM precision (upstream #835) and two attention switches
(upstream #1661, #1718).

These are this prompt size on this box. At the deployment's 262,144-token context and a 64K prompt the
absolute tok/s are higher for every layer (the 2026-11-02 record read 769.7 tok/s with layer 1 + layer 2
at 65,500 tokens), but the ratios are what the layers claim and those are measured here.

## What this record changes in the earlier ones

- **`549.0 -> 769.7 (+40 %)` becomes `448.2 -> 751.1 (+67.6 %)`, of which +15 % is code.** The old
  headline compared a cold arm at 8192-token chunks against a warm one, and counted three upstream levers
  as part of the change.
- **`STRATA_HIP_PROMPT_F16=1`**: +18.1 % in layer 1, +19.1 % on top of the port (the 2026-11-03 audit
  measured +10.4 % at 65,500 tokens and 8192-token chunks; the knob's value tracks the share of the
  hyper-connection read, which is a larger share at the smaller chunk).
- **The two attention switches**: +7.8 % at layer 1 against the audit's +5.8 % on top of the port - the
  same effect inside this box's band, bit-identical output either way.
- **The prompt-attention kernel**: -31 ms of `qsa attn` per 2048-token chunk, -1.4 % on the chunk medians
  (`2327 -> 2295`), rather than the -2.9 to -4.1 % that its own record read off two chunk walls - there the
  phase itself moved only 1.7 % of a chunk. Its end-to-end effect is inside this box's noise band.
- **`STRATA_MMQ_RESIDENT_SORT_NE=1` stays off** - the audit's verdict, now with a bracket around it.
- **`STRATA_PREFILL_PIPE=1` is a -21 % regression on this workload** (new; the GPU timeline is unchanged,
  the wall clock is not).
- **"No env lever is left" holds for the runtime levers that reach the prompt path on gfx906.** The
  candidates that still look like levers are CMake options, not environment: `STRATA_MMQ_TAB`,
  `STRATA_PREFILL_FUSED`, `STRATA_MMQ_KQUANTS`, `STRATA_PREFILL_MMQ`. `STRATA_HCD_EXACT` needs the gfx1151
  hipBLASLt tuning table (`src/core/arch_defaults.cpp:17`), and `STRATA_PF_GEMM` / `STRATA_PF_PAD` are
  documented as gfx11-only in `docs/AMD_HIP.md`.

## What is still open

- **int8 attention scores** (~+6-10 % on the old estimate, never built): the largest remaining phase at
  the kernel level, `qsa attn` 274 ms of 2295 after layer 3.
- **The GDN scan** (`gdn recurrence` 131 ms; the chunked WY form is CUDA/gfx11 only) - algorithmic work,
  not a knob.
- **The 8192-chunk over-lending**: the engine's own warning says the prompt path borrows more cache slots
  than fit in RAM. That is the +18.8 % of layer 1, and it is a *memory-budget* problem
  (`--resident-budget-gib`), not a kernel one; 2048 is the deployed workaround, not a fix.

