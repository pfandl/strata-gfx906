# 2026-11-05 - int8 (dp4a) prompt attention scores: -9.9 % of `qsa attn`, -1.2 % of a chunk, kept opt-in

The gfx906 prompt attention kernel of
[`20261103-qsa-prompt-attn`](../20261103-qsa-prompt-attn/README.md) computes its **score phase in fp32**: per
selected cell, each lane of a warp holds 8 of the 256 dimensions, dequantizes the 8 int8 K codes to float
(`load_k8`) and does `12 heads x (8 mul + 7 add)`. With `--kv int8` the codes are already int8 and
`gfx906:sramecc` has `v_dot4_i32_i8`, so the phase can be a **dot product**: quantize the query tile per 32
dimensions (Q8_0 rounding, `d = amax / 127`) and run **2 `v_dot4_i32_i8` per head-cell** instead of 15 fp32
ops. `STRATA_QSA_INT8_SCORES=1` is that path. It is **measured, correct, and not shipped**:

| | `qsa attn`, ms/chunk (median, 27-28 chunks) | full chunk, ms | kernel, ctx 32K | kernel, ctx 4K |
| --- | ---: | ---: | ---: | ---: |
| fp32 scores (shipped) | 274.0 | 2284 | 40.02 ms | 39.64 ms |
| `STRATA_QSA_INT8_SCORES=1` | **247.0** | 2282 | **36.28 ms** | **35.73 ms** |
| delta | **-27.0 ms = -9.9 %** | -2 ms = -0.1 % (noise) | -9.35 % | -9.85 % |

The phase is genuinely faster - the two arms' per-chunk distributions do not even touch (fp32 253-289,
dp4a 224-252) - and the component probe says the same thing from the other end (-9.4 % / -9.9 % of the
kernel). What it is **not** is a chunk: the score phase is about a tenth of this kernel, the kernel is about
a tenth of a chunk, so **-27 ms of 2284 ms = -1.2 %** is the ceiling, and the arms only read -7 and -8 ms
pairwise. On this box, that is inside the noise band.

Numerically it is fine: **top-1 100 % (20/20) at KL 0.01403 mean** against the decode path, where the
shipped fp32 kernel reads 0.01112 on the same run and 0.01364 in the 2026-11-03 session. So this lever is
**kept as an opt-in** (`docker-compose.yml` does not set it, the code is behind the flag) rather than
shipped or reverted: it is the last free thing in that kernel, and it costs nothing while it is off.

## The change

`src/kernels/cuda/qsa_prompt_attn_gfx906.cu`, one template parameter and one branch:

- `prompt_attn_kernel<KV_MODE, SCORES8>`; `SCORES8` is only instantiated with `KV_MODE == 1` (a
  `static_assert` says so), because the path reads `p.k_q` / `p.k_scale` directly. fp16 pools and the Q4_0
  KV that must stay on the decode kernel are untouched, and so is the whole fp32 phase: same tile, same
  swizzle, same arithmetic.
- **Per-block query quantization** replaces the fp32 tile in shared memory (the same 12 KB buffer, reused):
  each of the 12 query heads of a KV head is quantized per 32 dimensions with Q8_0's rounding
  (`rintf(x / d)`, clamped to +-127) and written through the swizzle the fp32 tile uses, so both phases
  address the tile the same way and the fp32 path's bytes are unaffected.
- **The score** becomes `dp4a(qw[0], k0, dp4a(qw[1], k1, 0))` per head with `part[h] * (sqscale * kscale)`.
  A lane's 8 dimensions sit inside **one** 32-dimension query group and **one** 64-dimension K group
  (`KV_Q8_GROUP`), so one scale product per head-cell is exact - no per-element correction and no leftover
  dequantize; the tile's store order (`d0+qflip` then `d0+(qflip^4)`) is why the two K words are swapped for
  the odd half-warps.
- Dispatch: `mode == 1 && score_int8_enabled()`, read once per process
  (`STRATA_QSA_INT8_SCORES=1`), beside the existing `STRATA_PROMPT_ATTN_GFX906=1`.

## The component probe

`bench/gfx906_qsa_prompt_attn_probe.cpp` (2048 queries, 2051 cells, 20 reps, one process per path), both
paths on int8 pools:

| ctx | path | prompt ms | vs the decode path | max_abs | rms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32768 | fp32 scores | 40.0180 | 1.618x | 5.960e-06 | 9.477e-08 |
| 32768 | dp4a scores | **36.2751** | **1.786x** | 2.185e-02 | 6.930e-04 |
| 4096 | fp32 scores | 39.6387 | 1.626x | 4.888e-06 | 9.341e-08 |
| 4096 | dp4a scores | **35.7349** | **1.803x** | 2.180e-02 | 6.965e-04 |

The kernel is not arithmetic-bound, which is the whole story of the lever: the dp4a swap removes the score
phase's multiplies but not its K traffic (8 int8 codes per lane per cell, the same 256 bytes per block-cell
either way), so a 5x cheaper score phase is a 9 % cheaper kernel. The accuracy columns are the probe's
usual "against the decode kernel" line and they move (rms 9.5e-08 -> 6.9e-04) because the queries are now
quantized; the yardstick below is what prices that.

## The engine A/B

Four arms in one session, interleaved (`base, dp4a, base, dp4a`), one engine start per arm, both arms with
`STRATA_PROMPT_ATTN_GFX906=1` and `STRATA_PREFILL_TIMING=1` on top of the layer-3 stack
(`STRATA_PF_RP / DENSE_MMQ / DENSE_RP / MMQ_KQ / HIP_PROMPT_F16 / GFX906_ATTN_QUERY_SWIZZLE /
GFX906_ATTN_REDUCE12`), `--max-context 32768 --prefill 2048 --kv int8 --spec 4 --spec-min-p 0.5
--layer-split 24 --resident-budget-gib 20 --pool-workers 6`, the seeded 32,500-token prompt, 16 generated:

| arm | prompt tok/s | chunk ms | `qsa attn` ms | `qsa attn`, every kept chunk |
| --- | ---: | ---: | ---: | --- |
| `base1` | 775.8 | 2290 | 274.5 | 253-289 |
| `i8_1` | 755.1 | 2283 | **248.0** | 232-252 |
| `base2` | 723.7 | 2284 | 273.5 | 254-281 |
| `i8_2` | 778.8 | 2276 | **246.5** | 224-251 |

`bench/arms-phases.txt` is the full per-chunk listing (`bench/qsa-phases.py`, the layered record's outlier
rule: first chunk dropped, chunks slower than 1.5x the arm's median dropped - `i8_1` lost one to a
1,549 ms `hc read` outlier, which is the host copy and not this kernel).

Three readings, and they disagree by design:

- **The phase**: 274.0 -> 247.0 ms, **-9.9 %**, and the two arms' spreads do not overlap. That is the
  kernel, and it agrees with the probe (-9.4 %).
- **The chunk**: 2284 -> 2282 ms pooled, -7 and -8 ms pairwise. The 27 ms that left `qsa attn` did not all
  leave the chunk; the other phases scatter by more than that between arms (this box's chunk-to-chunk
  band, the one the layered record's -1.4 % reading also lives in).
- **The `prompt tok/s` column is not a reading** either way: the pairs say -2.7 % and +7.6 % (base1 vs
  i8_1: 41.9 s vs 43.0 s; base2 vs i8_2: 44.9 s vs 41.7 s). Two arms of the *same* configuration span 7 %
  here, so 1 % of the chunk cannot be resolved end to end. That is why the verdict rests on the phase table
  and the component, not on the pp column.

## The gate (numerical): run

`tools/quant_quality.py prompt` + `pcompare`, three arms in one session on **one binary (the one this branch
builds, `md5 36149aa2...`)**, int8 KV, both gfx906 attention switches in all three arms, `--gpu-experts`,
20 prompts, KL over the full vocabulary against the decode path:

| arm | env | top-1 vs the decode path | KL mean | median | max |
| --- | --- | ---: | ---: | ---: | ---: |
| `ref` (decode path) | `STRATA_PROMPT_ATTN_GFX906=0` | - | - | - | - |
| `fp32` (shipped kernel) | `STRATA_PROMPT_ATTN_GFX906=1` | **100.00 %** (20/20) | 0.01112 | 0.00062 | 0.05995 |
| `int8` (this flag) | `=1` + `STRATA_QSA_INT8_SCORES=1` | **100.00 %** (20/20) | 0.01403 | 0.00078 | 0.09617 |

The reading: **top-1 does not move at all, and the KL step (+26 %) is the size of the harness's own
run-to-run scatter on this pair** - the same fp32-vs-decode comparison read 0.01364 on 2026-11-03 and
0.01112 here. For scale, on the same yardstick `STRATA_HIP_PROMPT_F16=1` reads 0.01185 and ships, and the
opt-in K-quant MMQ kernels read 0.095 and stayed opt-in; the dp4a arm sits between them, on the shipped
side.

**The acceptance half was not run**, because nothing is being shipped: `docker-compose.yml` keeps the fp32
score phase, so the deployment's numbers, the layered record and the MTP acceptance of the 2026-11-03
record all stand unchanged. If the switch is ever turned on in the deployment it needs that gate first -
the one whose arms are in `data/rp-opt/accept-attn/` (three request types, one recreate per arm, blocked
A/B/A, `draft_n` / `draft_n_accepted` out of the responses).

## Verdict

- **Kept, off by default, out of the deployment.** The lever is real on its phase (-9.9 %) and free
  numerically (top-1 100 %, KL in the shipped kernel's band), but 27 ms of a 2284 ms chunk cannot be
  resolved on this box and cannot be felt through a prompt that takes 42 s.
- **What would change it**: a configuration where the attention is a bigger share of the chunk - more
  selected cells per query, a shorter chunk, or a box whose `qsa attn` is not already ~12 % of the prompt.
  Nothing in the current deployment is close.
- **What is left in the kernel**: the *value* phase, which was measured afterwards - see the section below:
  it is 58 % of the kernel and it is latency-bound, not instruction-bound, so it does not take a dp4a-style
  rewrite (that would mean quantizing the probabilities); it takes more memory-level parallelism at no
  register cost.

## The value phase, checked

The dp4a win is bounded by the other phase, so the kernel was instrumented to split it (a temporary `skip`
argument: `skip=1` removes the value loop, `skip=2` replaces the scores with 0 and keeps the softmax; the
knob is not committed). Component probe, nq 2048, ctx 32768, int8 pools, 20 reps, ms:

| scores | full | no value loop | no score phase | value phase | score phase | rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp4a | 36.46 | 15.30 | 23.80 | **21.16 (58 %)** | 12.66 (35 %) | 2.64 |
| fp32 | 39.93 | 19.73 | 23.63 | **20.20 (51 %)** | 16.30 (41 %) | 3.43 |

The two arms' score-phase difference (16.30 - 12.66 = 3.64 ms) is the kernel's own dp4a delta (3.47 ms), so
the split is consistent. Per cell per thread the value phase is: 1 `srow[c]` LDS read whose result *is* the
next load's address, 1 int8 V byte + 1 fp16 scale (broadcast) + 2 converts + 1 multiply, 12 broadcast
`LDS.32` of `sp[h][c]`, 12 FMAs - about 30 instructions per cell per thread, 2051 cells per query, 2 KV
heads, 2048 queries per chunk.

Two attempts, both measured on the component and both **byte-identical** to the probe's reference output:

| attempt | dp4a | fp32 | verdict |
| --- | ---: | ---: | --- |
| cell-major probability tile: `spv[64][16]`, written by the softmax, read as 3 broadcast `LDS.128` instead of 12 broadcast `LDS.32` | 36.16 (-0.8 %) | 40.27 (+0.9 %) | **neutral** - the phase is not LDS-issue bound |
| 4-deep gather pipeline (`VDEPTH=4`: four V rows loaded before any is consumed) | 48.31 (**+33 %**) | 49.86 (+25 %) | **a loss** - and the same build's *no-value* variant reads 19.54 vs 15.30, i.e. the extra live registers cost occupancy for the whole kernel |

So the phase is **latency- and dependency-bound, not instruction-bound**: the per-cell chain is `srow` LDS
read -> global address -> 256-byte V row gather -> convert -> 12 FMAs, with ~12 resident waves per CU to hide
it, and neither halving its LDS instructions nor deepening the gather moves it. What is left worth trying is
**cooperative staging of the pass's V rows into LDS** (64 x 256 B = 16 KB per pass: 4 coalesced `LDG.128` +
4 `STS.128` per thread, so the gathers issue as one batch and the loop's dependency chain becomes LDS-only).
It costs 16.5 KB of LDS per block, which takes the fp32-score path from 3 to 2 blocks per CU - and the two
experiments above say the occupancy is what decides it. Not attempted here: this lever's end-to-end ceiling
is 1.2 % of a chunk, and that rewrite would need the whole gate again for a phase that is already 58 % of a
kernel that is 12 % of a chunk.

## Reproduce

```sh
# component probe: 2 paths on int8 pools, one process each (the dispatch caches per process)
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --group-add "$(stat -c %g /dev/kfd)" --ipc=host --user "$(id -u):$(id -g)" \
  -v "$PWD/rp-build":/work -v "$PWD/data":/data strata-gfx906:local bash -lc \
  'for m in 0 1; do env STRATA_PROMPT_ATTN_GFX906=1 STRATA_QSA_INT8_SCORES=$m \
     /work/build-906/gfx906_qsa_prompt_attn_probe 2048 32768 20 /data/i8/p-$m.bin; done'

# the engine A/B: four arms, phases on, deployment stopped and restarted by the harness
ARMS=$(printf 'base1|STRATA_PREFILL_TIMING=1 STRATA_PROMPT_ATTN_GFX906=1\ni8_1|STRATA_PREFILL_TIMING=1 STRATA_PROMPT_ATTN_GFX906=1 STRATA_QSA_INT8_SCORES=1\nbase2|STRATA_PREFILL_TIMING=1 STRATA_PROMPT_ATTN_GFX906=1\ni8_2|STRATA_PREFILL_TIMING=1 STRATA_PROMPT_ATTN_GFX906=1 STRATA_QSA_INT8_SCORES=1\n') \
  OUT=/data/qsa8 GEN_TOKENS=16 bench/gfx906/decode-ab.sh
python3 docs/gfx906-results/20261105-int8-scores/bench/qsa-phases.py \
  /home/me/Documents/strata/data/qsa8/base1 /home/me/Documents/strata/data/qsa8/i8_1 \
  /home/me/Documents/strata/data/qsa8/base2 /home/me/Documents/strata/data/qsa8/i8_2

# the yardstick: exe pointed at the binary this branch builds (the image ships its own)
python3 tools/quant_quality.py prompt --config /data/config/strata-iq3_s-work.json \
  --out /data/gate-i8/ref  --gpu-experts --kv int8 --env STRATA_PROMPT_ATTN_GFX906=0
python3 tools/quant_quality.py prompt --config /data/config/strata-iq3_s-work.json \
  --out /data/gate-i8/fp32 --gpu-experts --kv int8 --env STRATA_PROMPT_ATTN_GFX906=1
python3 tools/quant_quality.py prompt --config /data/config/strata-iq3_s-work.json \
  --out /data/gate-i8/int8 --gpu-experts --kv int8 --env STRATA_PROMPT_ATTN_GFX906=1 \
  --env STRATA_QSA_INT8_SCORES=1
python3 tools/quant_quality.py pcompare /data/gate-i8/ref /data/gate-i8/fp32 /data/gate-i8/int8
```

`bench/` holds the probe output (`probe.txt`), the phase table (`arms-phases.txt`) with the script that
prints it (`qsa-phases.py`), the gate (`gate.txt`), and the value-phase investigation
(`value-phase-split.txt`, `value-phase-sp-transposed.txt`, `value-phase-pipeline-d4.txt` - the last two are
the two attempts' raw runs, byte-identity checked with `cmp` against `/data/i8/int8-32768.bin` and
`/data/i8/fp32-32768.bin`). The raw arm directories (`/data/qsa8`, 15 chunks each) and the three gate
directories (`/data/gate-i8/*`, 21 files each) stay on the box.

## Files

- `src/kernels/cuda/qsa_prompt_attn_gfx906.cu` - the `SCORES8` path and its dispatch.
- `docs/GFX906.md` - one paragraph in the deployment section, beside `STRATA_SPLIT_OVERLAP` and
  `STRATA_GR_SPLIT`.
