# 2026-11-05 - `STRATA_GR_SPLIT=1` (gfx906 K-split hyper-connection down read): the read is 10 % faster, the text is not the text, and in the deployment's regime it is slower

`STRATA_GR_SPLIT=1` is a **gfx906-only** branch in the hyper-connection read: the down projection is
split along K into 8 rows x 5 slices of 2048 (`gr_down_split_kernel`) with a finish kernel that sums
the slices, instead of one pass (`src/kernels/cuda/fused_gr.cu:2302-2325`, inside
`#if defined(STRATA_HIP_GFX906)`). `docs/AMD_HIP.md` prices it at "~3 % per window" and says why it
is off: "sums in another order than the single-token read, so off by default". It has never been run
on this deployment.

Measured here, in three parts:

| | what it says |
| --- | --- |
| the read itself (`fused_gr_bench`, T 1..5, 3 runs each mode, same inputs) | **-10.4 % median** (-7.4 % .. -16.2 %), and **every output hash differs** |
| the contract (`gr_parity`) | **0 failures off, 3 on**: the split no longer matches T single-token reads at multi-row windows |
| end to end, MTP off (4 arms, comparable windows) | **-1.1 % per token** (26.51 -> 26.23 ms), same drafts (2 of 7) on both sides |
| end to end, the deployment's regime (MTP on, 6 arms) | **+2.6 % per token** *and* a different token stream (125 vs 120 tokens, 17 vs 22 draft proposals) |

**Verdict: not a lever to ship.** What it buys is ~1 % per token in the clean shape, it changes the
numbers by design (so it needs the full gate, which it fails at multi-row windows), and where the
deployment actually runs - MTP on - the changed text triggers more draft work than the read saves.
The `-10 %` of the read is the ceiling of the kernel itself, not of the model.

## What it is, and what it can touch

- gfx906 only, opt-in, and it needs **bf16 down rows**: `a[0].s_down == nullptr` (i.e. not
  `STRATA_HC_REQ8`) or the branch is skipped. It rides `STRATA_GR_FAST` (on by default on HIP) for
  the norm/up halves; `gr_norm_multi`/`gr_up_multi` otherwise.
- **Decode only.** The prompt's hyper-connection read lives in `src/prefill/prefill.cpp` (with its
  own `STRATA_PF_HCDOWN`/`STRATA_HC_UPMIX` opt-ins) and never calls `fused_gr_read*`.
- It applies to **any** window with `n_tok >= 1` (`fused_gr_read_multi` has no T>1 guard), so unlike
  the overlapped hand-off of the [split-overlap record](../20261105-split-overlap/README.md) it is
  not limited to multi-row windows.
- It is in the binary the image ships: `gr_down_split` symbols are in
  `/work/build-906/strata` (`md5 85693dc7...`); `docs/GFX906.md:2321-2326` records the clang-23
  `return;` fix in this block and that the deployed configuration has never set the variable.

## The read, measured against itself (`fused_gr_bench`)

`fused_gr_bench 50 1 5 1` runs the read per shape twice - the older kernels and the `STRATA_GR_FAST`
ones, `fused_gr_set_fast(0/1)` - and the split replaces the down read **inside both**. So the pair
to compare across the two modes is `fast` against `fast`:

| T | apply | inject | fast, flag off | fast, flag on | delta | FNV of the output |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 1 | 55.0 us | 49.5 us | -10.0 % | differs |
| 2 | 1 | 1 | 63.9 us | 58.3 us | -8.8 % | differs |
| 3 | 1 | 1 | 76.8 us | 66.7 us | -13.2 % | differs |
| 4 | 1 | 1 | 88.4 us | 74.8 us | -15.4 % | differs |
| 5 | 1 | 1 | 102.1 us | 89.4 us | -12.4 % | differs |

Over all 20 shapes: **median -10.4 %, range -7.4 % .. -16.2 %** (`bench/parse-component.py`). The
docs' "~3 % per window" is this read's share of a whole window, and the arithmetic says what share
that would need: the read runs once per layer, so 48 x 55-100 us is 2.6-4.8 ms of the ~30 ms window
this box decodes in (9-16 %), and 10 % off the read is 1-1.6 % of the window - which is what the
end-to-end block below measured (-1.1 %). 3 % of a window needs the read at about a quarter of it,
i.e. the shape it was measured in (131K context, multi-row windows, 16 GB cards).


## The contract (`gr_parity`)

```
flag off:  fused GR read + q8_1 (STRATA_QFUSE), T 1..8, direct and graph replays: pass
           gr_read/gr_write: 0 failures
flag on:   fused GR multi max-T differs from single-token calls
           fused GR changed graph replay differs from single-token calls
           fused GR read + q8_1 (STRATA_QFUSE), T 1..8, direct and graph replays: FAIL
           fused GR multi max-T=8 LDS launch and changing graph replay FAIL
           gr_read/gr_write: 3 failures
```

That is the documented cost, now with a number: the sums are re-associated, so the multi-row read is
no longer bitwise equal to T single-token calls. Anything that ships it needs the real gate (prompt
top-1/KL, MTP draft acceptance), and here the failures are exactly at the windows that gate would
score.

## End to end: two sessions, because a lever that changes the text is not a flag swap

Everything else is the harness of the split-overlap record (`bench/gfx906/decode-ab.sh`, renamed from
`overlap-ab.sh` and given `MTP=`/`SPEC=` knobs for this): the layer-3 stack in every arm, 32,500-token
seeded prompt, 32K context, `--prefill 2048`, split 24, 128 generated tokens, one engine start per
arm, one container, arms interleaved base/flag.

### Session A - the deployment's regime (MTP on, `--spec 4`), 6 arms

| arm | generated | decode ms | ms/token | drafts | ids md5 |
| --- | ---: | ---: | ---: | ---: | --- |
| base1 | 125 | 3808 | 30.46 | 0/17 | `2323bd2cb25f` |
| gr1 | 120 | 3882 | 32.35 | 0/22 | `f6da826cc889` |
| base2 | 125 | 3819 | 30.55 | 0/17 | `2323bd2cb25f` |
| gr2 | 120 | 3761 | 31.34 | 0/22 | `f6da826cc889` |
| base3 | 125 | 3880 | 31.04 | 0/17 | `2323bd2cb25f` |
| gr3 | 120 | 3751 | 31.26 | 0/22 | `f6da826cc889` |

Median ms/token: **30.55 -> 31.34 (+2.6 %)**. Read as a regression it would be wrong: the two sides do
not run the same work. The lever changes the sums, the text diverges (125 vs 120 tokens - the EOS
comes five tokens earlier), and with MTP on the *draft layer* then proposes more often (22 against 17
for a shorter answer): each proposal is a draft chain plus a wider verify window. The +2.6 % is that
extra work, not the read. It is also the honest answer for the deployment: with the drafter on, a
numerically-different lever spends its gain on a changed token stream.

### Session B - MTP off, so the windows are comparable, 4 arms

With the draft layer off (`MTP=` empty) the engine falls back to `--spec 4` prompt lookup
(`spec=6 mtp_max=4 lookup=3` in its INFO line), and both sides proposed **2 of 7** lookahead tokens -
the same window mix, so ms/token is a like-for-like number again:

| arm | generated | decode ms | ms/token | drafts | ids md5 |
| --- | ---: | ---: | ---: | ---: | --- |
| base1 | 125 | 3327 | 26.62 | 2/7 | `2323bd2cb25f` |
| gr1 | 120 | 3138 | **26.15** | 2/7 | `f6da826cc889` |
| base2 | 125 | 3299 | 26.39 | 2/7 | `2323bd2cb25f` |
| gr2 | 120 | 3156 | **26.30** | 2/7 | `f6da826cc889` |

Median ms/token: **26.51 -> 26.23 = -1.1 %**: real, and as small as the component arithmetic predicts
(10 % off a read that is 9-16 % of a window). Decode is faster overall without the drafter
(37.6-38.2 tok/s against 32-33) - a separate observation about this prompt, where MTP proposes and
never lands.

## Verdict

- **Not shippable as it is.** It changes the numbers by design and fails `gr_parity` at exactly the
  multi-row windows an acceptance gate would score; the gate itself (top-1, KL, draft acceptance) has
  not been run, and the end-to-end number it would be bought with is ~1 %.
- **The kernel is fine**: -10.4 % median on its own output, at every shape T 1..5, both `apply` and
  `inject` variants. What breaks is that a re-associated sum is a different model output, not a
  faster identical one.
- **In the deployment's regime it is a loss** (+2.6 % per token, session A), because the changed text
  moves the draft layer's work. That is a lesson for every numerically-different knob on this box,
  and the reason both sessions exist.

## The next decode lever, and exactly what it needs: `--pipeline-windows 2`

Checked in the source (`src/program/generate.cpp:4256-4305`, `4410-4414`, `4517-4518`, `9044-9047`),
not read off the docs, because the first version of this note had it wrong:

- **Default only on gfx1200/gfx1201.** The auto path needs `decode_defaults_arch_ok()` =
  `gfx_arch_is_gfx12_wmma()` = `gfx1200 || gfx1201` (`include/strata/kernels/gfx_arch.hpp:30`); a
  non-HIP build forces it off (NVIDIA keeps it opt-in). `STRATA_PIPELINE_AUTO=1` only switches it
  *off* - on gfx906 it has to be asked for as `--pipeline-windows 2`.
- **Two stages are the special case**: `min_st` is 2 only when `mtp_first_ok && mtp_first_default_on()`
  (gfx12 again), otherwise 3. So even on gfx12 a two-stage split only gets the default with the draft
  layer already on the first card.
- **The gate is longer than "--serve and a draft layer"**: `--serve`, each stage on its own GPU, two or
  more stages, `--mtp` with `--spec >= 2`, and *not* `--batch`, `--peer-device`,
  `--expert-cache-device1..3`/`--remote-expert-opt`, `--adapt-async` (3+ stages), or
  `--pipeline-windows 1` on 3+ stages. This deployment passes all of them (MTP on, `--spec 4`,
  batch 0, no peer, no helper caches).
- **`STRATA_MTP_STAGE0=1` is a requirement here, not a pairing.** The drafter moves to the first card
  only when `pipeline_windows >= 2` *and* (`STRATA_MTP_STAGE0=1` or `mtp_first_default_on()`, gfx12
  only), and only if `mtp_first_ok`: `--mtp` + `--spec >= 2` + the draft head's subset file
  (`/data/mtp/rt/draft_vocab.bin` is present here) + no `STRATA_MTP_FULL_HEAD=1`. Without it the flag
  is kept and the engine prints the #2637 note (the pipeline lost to the serial loop on 2x R9700
  before 0.1.44; `docs/MULTI_GPU.md`, "Where the draft layer sits").
- **Costs and conflicts**: 160 MiB per card out of the expert cache plus ~96 MiB of GDN snapshots on
  card 0 (`kPipeWindowMib`/`kPipeSnapMib`), and `--pipeline-windows > 0` **switches off**
  `STRATA_SPLIT_OVERLAP` (`generate.cpp:9044-9047`): it trades the shipped +1.2 % decode for whatever
  it brings. The live engine reports 2004 MiB free on CUDA0 and 712 MiB on CUDA1 at the 25/23 split.
- **No entrypoint knob**: `docker/entrypoint.sh` has nothing for it; through the deployment it takes
  `STRATA_SET="--pipeline-windows 2"` and a saved preset, or a direct engine run.

That is what a session for it has to hold fixed, and why it is the one decode lever left that is
worth its VRAM: it does not change the text (the tokens are the serial loop's up to CPU rounding).

## Harness and files

- `bench/gfx906/decode-ab.sh` - the A/B driver (was `overlap-ab.sh`), now generic: `ARMS`, `OUT`,
  `MTP=` (empty drops the draft layer), `SPEC=`, `EXTRA_INSIDE=` (a script inside the container run
  after the arms), the `WRONG-PACK` assertion and the hand-off line in every arm's summary.
- `bench/gfx906/gr-component.sh` - the component half: `fused_gr_bench` and `gr_parity`, flag off and
  on, into `$OUT/component/`.
- `bench/gfx906/parse-arms.py` - the end-to-end table and the ids md5 (moved out of the split-overlap
  record so both records use one copy).
- `bench/parse-component.py` - the component table above.
- `bench/session-a/` (6 arms), `bench/session-b/` (4 arms), `bench/component/` (7 logs): every arm's
  `cmd.txt`, `info.txt`, `stdout.log` (with its `T <id>` lines), `stderr.log`, `summary.txt`.

Reproduce:

```sh
# A: the deployment's regime, with the component half after the arms
ARMS=$(printf 'base1|\ngr1|STRATA_GR_SPLIT=1\nbase2|\ngr2|STRATA_GR_SPLIT=1\nbase3|\ngr3|STRATA_GR_SPLIT=1\n') \
  OUT=/data/grsplit EXTRA_INSIDE=/work-bench/gr-component.sh bench/gfx906/decode-ab.sh
# B: comparable windows (no draft layer)
ARMS=$(printf 'base1|\ngr1|STRATA_GR_SPLIT=1\nbase2|\ngr2|STRATA_GR_SPLIT=1\n') \
  OUT=/data/grsplit2 MTP= bench/gfx906/decode-ab.sh
python3 bench/gfx906/parse-arms.py /home/me/Documents/strata/data/grsplit2
python3 docs/gfx906-results/20261105-gr-split/bench/parse-component.py \
  /home/me/Documents/strata/data/grsplit/component
```


