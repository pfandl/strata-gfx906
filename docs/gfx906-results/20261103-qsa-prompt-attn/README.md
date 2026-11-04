# 2026-11-03 - the prompt attention: two in-tree switches (+5.8 % measured) and a gfx906 prompt kernel

The 2026-11-03 [opt audit](../20261103-opt-audit/README.md) ended with one kernel-scale item: `qsa attn`, 427.6 ms of a
2734 ms chunk, is not a GEMM on this card at all - `qsa_prompt_attn_batch()` refuses gfx906 (its kernels are
m16n8k16 MMAs and cp.async), so `prefill.cpp` loops the *decode* kernel over 32-query batches and every query re-reads
its KV window. This record covers both halves of that work: the switches that were already in the tree and never
measured on this deployment, and a prompt kernel written for this card.

Everything below is the same two-card box, the same fixed random-word prompt (32,700 tokens, context 32768,
`--prefill 2048`, `--kv int8`, `--spec 4`, `--layer-split 24`), one engine start per arm through
[`bench/gfx906/rp-ab.sh`](../../../bench/gfx906/rp-ab.sh), and the shipped configuration in every arm: the four RP flags
**and** `STRATA_HIP_PROMPT_F16=1` (the +10.4 % knob from the audit). Artefacts: `data/rp-opt/s0-*` (the switch arms),
`data/rp-opt/s1-*` (the kernel's arms), `data/rp-opt/accept-attn/` (the acceptance gate).

## 1. The two gfx906 attention switches: +5.8 % of prompt, bit-identical

`STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` (PR #1661) and `STRATA_GFX906_ATTN_REDUCE12=1` (PR #1718) shipped as off-by-default
community opt-ins, and the notes in their results directory measured them on the *component* (the batched attention at
`n_q=32`, `ctx=65536`) and once end to end at 65K. They had never been measured on this deployment's configuration, and
nothing in the deployment turned them on.

**Component, A/B/C/C/B/A, one process per arm** (`bench/gfx906_qsa_query_swizzle_probe`, 50 reps, medians; the
dispatcher caches the environment, so each setting needs its own process):

| arm | env | 4K ctx | 32K ctx | 65K ctx (20 reps) |
| --- | --- | ---: | ---: | ---: |
| `a1` | (default) | 0.9706 ms | 0.9737 ms | 0.9131 ms |
| `b1` | `+ QUERY_SWIZZLE=1` | 0.8898 | 0.8935 | 0.8540 |
| `c1` | `+ QUERY_SWIZZLE + REDUCE12` | **0.6547** | **0.6591** | **0.6402** |
| `c2` | (repeat of `c1`) | 0.6542 | 0.6602 | 0.6415 |
| `b2` | (repeat of `b1`) | 0.8890 | 0.8934 | 0.8330 |
| `a2` | (repeat of `a1`) | 0.9738 | 0.9718 | 0.9001 |

-27.6 % at 4K, **-32.3 % at 32K**, -30.0 % at 65K for the pair; the swizzle alone is -8.3 % here (the community record's
-14.6 % was at `n_q=32`/65K on their build, where the batch's own overhead differs). The two repeats of every arm agree
to 0.2 %, so these are not clock-drift artifacts.

**Every arm's output hash is identical** (`OUTPUT_FNV1A64=841abe1e61cb58a7`, 786,432 output bytes, at 32K; the 4K and
65K runs likewise agree arm for arm), i.e. the switches leave the arithmetic unchanged - the swizzle only permutes the
query tile inside LDS, and `reduce12` is the same butterfly as twelve `warp_sum` calls with 16 exchanges instead of 60.

**Engine, blocked A/B/A, warm** (the second `a` arm is the reference: the first start of a session reads the 62 GB GGUF
from disk, which flatters whatever runs next):

| arm | env | GPU ms/chunk | qsa attn | hc read | pp tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `s0-a1` | shipped config (cold start) | 2835 | 431.6 | 269.1 | 654.4 |
| `s0-c1` | `+ the two switches` | **2304** | **304.7** | 262.1 | **769.7** |
| `s0-a2` | shipped config (warm reference) | 2432 | 425.0 | 263.3 | 727.3 |

The whole delta is the attention: `qsa attn` 425.0 -> 304.7 ms (-28.3 %), every other phase unchanged (hc read 263.3 ->
262.1, gate/up 413.5 -> 410.1). Against the warm reference that is **-128 ms of a 2432 ms chunk and +5.8 % of prompt**
(727.3 -> 769.7 tok/s); `s0-a1` is the cold-start trap the audit documents (654.4 tok/s for the same configuration as
`s0-a2`).

Verdict: ship them on. They are bit-identical at both levels, they need int8 KV (which the deployment runs), and they
cost nothing. `docker-compose.yml` now sets both with `=0` as the opt-out.

## 2. A prompt kernel for this card (`qsa_prompt_attn_gfx906.cu`)

The decode kernel's shape is the problem: it splits each query's cells across `n_chunks` blocks and writes
`n_chunks x 24 heads x 256` floats of partials per query - 811 KB at a 32K context - then a merge kernel reads them all
back, once per 32-query batch. On this box that is ~80 GB of scratch traffic per 2048-token chunk.

`src/kernels/cuda/qsa_prompt_attn_gfx906.cu` keeps the per-cell arithmetic (the same int8 codes and per-64 scales, the
same query-LDS swizzle, the same 12-head butterfly) and changes only the traversal: **one block per (query, KV head)**
walks all of that query's selected cells with a running softmax, so the partials stay in registers and LDS and there is
no scratch write, no merge kernel and no batching of queries. `qsa_prompt_attn_batch()` asks it first on this build;
`STRATA_PROMPT_ATTN_GFX906=1` turns it on (unset = the old loop - it is opt-in, see the status line below), and a pool
it does not serve (Q4_0, k8v4) or another geometry falls back to the old loop as well.

**Accuracy against the decode kernel** (`bench/gfx906_qsa_prompt_attn_probe`, `n_q=2048`, `ctx=32768`, the same
synthetic selection the swizzle probe builds): `max_abs=5.960e-06`, `rms=9.477e-08`, 515,839 of 12,582,912 elements
bitwise equal (`max_rel=1.389` is one element whose reference is ~0, which the absolute bound covers). The scores and the
per-dimension value products are the same operands in the same order; the difference is when the exponentials are taken
(running maximum instead of each chunk's own, with the merge's rescale folded in), i.e. the last bits. The same probe
with `--masked-pages` (683 of 2048 pages left non-resident, `n_q=512`, `ctx=8192`): `max_abs=4.768e-06`, 159,187 of
3,145,728 bitwise - the page -1 masking path is the decode kernel's as well.

**Speed on the same geometry**: 45.66 ms -> **39.93 ms (1.144x)**, 22.30 -> 19.50 us per query - against the decode path
*with both switches on*. Without them the same probe measures 64.48 -> 39.78 ms (1.621x), which is what the phase would
look like had the switches not shipped first. (Two runs of the 32K probe: prompt 39.68 and 39.93 ms, i.e. 1.151x /
1.144x.)

**A bug the engine gate caught that the probe did not** (worth recording): the first version read the selection width
once per *call* (`step + kStepWidth`) instead of once per *query* (`step + query * kStepCount + kStepWidth`). A chunk's
first and last query differ by up to a chunk's worth of cells, so the engine's prompt read the wrong cell counts and the
gate came back **top-1 80 %, KL 0.609** - while both probe geometries (4K and 32K) differed by only a few cells per
query and stayed at `max_abs` ~5e-06. Lesson: a component probe whose inputs vary less than the engine's inputs cannot
gate what the engine does.

**The gate, run** (`tools/quant_quality.py prompt` + `pcompare`, int8 KV, the four RP flags, `STRATA_HIP_PROMPT_F16=1`
and both switches in *both* arms, `--gpu-experts`, one engine start per arm; `pcompare`'s KL is over the full
vocabulary):

| arm | env | top-1 vs the decode path | KL mean | median | max |
| --- | --- | ---: | ---: | ---: | ---: |
| `arm1` (kernel) | `STRATA_PROMPT_ATTN_GFX906=1` | **100.00 %** (20/20) | 0.01364 | 0.00062 | 0.13760 |

Re-run on the committed binary (the degenerate-input guards added after the first pass - a zero-width selection and the
`L > 0` divisor, both of which the gated inputs never reach) gave the identical line to the digit, and the probe's
accuracy figures are bit-identical too (`max_abs` 5.960e-06, 515,839 bitwise), so the shipped binary is the gated one.

For scale, on the same yardstick: `STRATA_HIP_PROMPT_F16=1` is 0.01185 / 0.00035 and shipped; the opt-in K-quant MMQ
kernels against the FP16 path are 0.095 / 0.0027 and stayed opt-in. This change sits with the first, on a stricter
input (int8 KV, i.e. the mode the kernel actually acts in).

**Status**: the kernel ships **off by default** (`STRATA_PROMPT_ATTN_GFX906=1` opts in). What is still missing before
it could become a default is the other half of the gate this box uses - MTP draft acceptance through the deployment:
three request types, greedy, `timings.draft_n_accepted / draft_n` out of each response, one compose recreate per arm
(the recipe and its artefacts are `20261103-opt-audit/bench/accept/`, `mtp3-*`). Its speed half is section 3 and clears
the prediction; the image's engine binary also has to be rebuilt from this commit before the deployment can set the
variable at all.

## 3. The engine A/B/A: `s1-a1` / `s1-k1` / `s1-a2`

Same box, same prompt (32,700 tokens, context 32768, `--prefill 2048`), the deployment's configuration in every arm
(four RP flags, `STRATA_HIP_PROMPT_F16=1`, both attention switches), `STRATA_PROMPT_ATTN_GFX906=0` in the two `a` arms.
One engine start per arm, the decode path bracketing the kernel. Chunks whose GPU timeline is a disk-stall outlier
(> 2800 ms: first-touch reads of the 62 GB GGUF show up as one or two such chunks per arm, trap 1 of the audit) are
excluded from the means; the first chunk is always excluded.

| arm | path | chunks kept | GPU ms/chunk | `qsa attn` | wall tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `s1-a1` | decode | 12 of 14 | 2336 | 310.8 | 784 |
| `s1-k1` | **kernel** | 14 of 14 | **2268** | **270.6** | **844** |
| `s1-a2` | decode | 12 of 14 | 2346 | 312.8 | 815 |

The two references agree to 0.4 % on the chunk and 0.6 % on the phase, so the read is: **`qsa attn` 310.8 -> 270.6 ms
(-40.2 ms, -12.9 %) and the chunk 2336 -> 2268 ms (-68 ms, -2.9 %)** - the phase saving is exactly what the component
probe predicted (45.66 - 39.68 = 36.0 ms there, 40.2 ms here), and the chunk gains a little more than the phase, which
is what removing a merge-kernel launch per 32-query batch should look like. Wall tok/s 784/815 -> 844.

The reference arms' own spread (310.8 vs 312.8 on the phase, 2336 vs 2346 on the chunk) is the resolution limit of one
run per arm, and the kernel arm's 270.6 is 13 % below both - well outside it.


## What this changes for the next step

The audit ranked the attention as the box's second lever (15.6 % of the chunk, ~428 ms) and put it at "a query-blocked
tiled kernel". The switches were the cheap half of that: -120 ms for two environment variables that change no bits. The
kernel is the other half, and the honest reading of its component number is that the *structure* the decode kernel
imposes (split-K partials + merge, one 32-query batch at a time) is worth ~1.15x, not the 3x the "no prompt kernel
exists" framing suggested - the decode kernel's inner loops were already the swizzled/reduce12 form.

What would move the phase again is arithmetic, not structure: the scores are FP32 FMAs over int8 codes today, and gfx906
has `v_dot4_i32_i8` (already mapped to `__dp4a`, `hip_compat/intrinsics.hpp`). An int8 score phase (q quantized per
64-dimension group, exact column sums carried through the P·V side) is a ~4x instruction-rate claim that costs real
numerics - a much larger gate than this kernel's - and it is the only remaining lever of that size in the phase.

## Reproducing

    # the switches, component: ONE PROCESS PER ARM (the dispatcher caches the environment)
    #   plain, +STRATA_GFX906_ATTN_QUERY_SWIZZLE=1, +both - run each at 4K / 32K / 65K and repeat the series backwards
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
      -v "$PWD/strata-gfx906":/repo:ro -e HIP_VISIBLE_DEVICES=0 \
      strata-gfx906:local /repo/build-906/gfx906_qsa_query_swizzle_probe 32 32768 50 /tmp/out.bin
    #   ... plus -e STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 [and -e STRATA_GFX906_ATTN_REDUCE12=1]

    # the engine arms (one engine start per arm; the deployment must be stopped: `docker compose stop`)
    #   s0: plain, then +both switches, then plain again (blocked A/B/A)
    #   s1: STRATA_PROMPT_ATTN_GFX906=0, then =1, then 0 again
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
      -v "$PWD/data":/data -v "$PWD/rp-build":/work -v "$PWD/strata-gfx906":/repo:ro \
      -e MODEL=IQ3_S -e PACK=/data/packs/iq3_s -e OUT=/data/rp-opt -e CTX=32768 \
      -e PROMPT_TOKENS=32700 -e PREFILL=2048 -e STRATA_PREFILL_TIMING=1 \
      -e STRATA_HIP_PROMPT_F16=1 strata-gfx906:local /repo/bench/gfx906/rp-ab.sh s0-a1 RP
    #   ... plus -e STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 -e STRATA_GFX906_ATTN_REDUCE12=1 for the switch arm

    # the kernel: accuracy and time against the decode path on one synthetic chunk
    #   run once WITHOUT the two switch vars to price it against the pre-switch decode path, once WITH them
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
      -v "$PWD/rp-build":/repo:ro -e HIP_VISIBLE_DEVICES=0 \
      -e STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 -e STRATA_GFX906_ATTN_REDUCE12=1 \
      -e STRATA_PROMPT_ATTN_GFX906=1 \
      strata-gfx906:local /repo/build-906/gfx906_qsa_prompt_attn_probe 2048 32768 10 /tmp/cand.bin
    #   without STRATA_PROMPT_ATTN_GFX906=1 the probe prints DECLINED and exits 3 (it measures the decode path)

    # the gate (int8 KV, the deployment's flags, one engine start per arm; `--gpu-experts` so the arms differ only in
    # the attention): arm0 `--env STRATA_PROMPT_ATTN_GFX906=0`, arm1 `--env STRATA_PROMPT_ATTN_GFX906=1`, then pcompare.
    # It needs `--kv int8` on BOTH sides, which tools/quant_quality.py gained in this change (it had no way to pin the
    # KV format, so an arm could silently run fp16 KV and answer a different question).

The phase tables of the engine arms are parsed with
`python3 docs/gfx906-results/20261103-opt-audit/bench/parse-phases.py data/rp-opt/<arm>`; the arms keep `summary.txt`,
`info.txt` and the engine's `stderr.log` (`s0-a1`, `s0-c1`, `s0-a2`; the kernel's engine arms land as `s1-*`).

