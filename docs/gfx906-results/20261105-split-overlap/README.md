# 2026-11-05 - `STRATA_SPLIT_OVERLAP=1` on the two-card split: +1.2 % decode, bit-identical

`STRATA_SPLIT_OVERLAP=1` is the engine's **overlapped hand-off** (`docs/MULTI_GPU.md`, "Overlapped
hand-off"): instead of the host syncing stage 1, then staging and launching stage 2's graph, the
first stage's GPU writes the hand-off with volatile stores and raises a ready word, the next
stage's graph is staged and launched on a helper thread, and its GPU spins on that word
(`wait_flag_ge`). It takes one stream sync and one graph launch per window off the critical path.
It was measured on 2x RX 7900 XTX only - `53 tok/s -> 70.1` decode on a code prompt, `45 -> 55.9` on
prose - and it had never been run on this box.

On the two MI50s it is worth **+1.2 % decode** and nothing else:

| | decode tok/s (median of 3) | decode ms/token | prompt tok/s (median of 3) |
| --- | ---: | ---: | ---: |
| serial order (default) | 32.60 | 30.68 | 762.3 |
| `STRATA_SPLIT_OVERLAP=1` | **33.00** | **30.32** | 711.4 |
| delta | **+1.2 %** | **-1.2 %** | no reading (see below) |

The flag **changes no numbers**: all six arms generated the same 125 token ids (md5 `2323bd2cb25f`),
accepted the same 0 of 17 drafts and wrote the same checkpoint. It is a timing lever only.

## What the flag can and cannot touch

- It exists only in the **decode** verify-window loop: `Verifier::chain_overlap()`
  (`src/core/verify.cpp:2834`) is true only with a next stage on **another GPU**, a hand-off flag
  pair, and no PLE stage; the chain arm of `Verifier::run` (`verify.cpp:3102`) then hands the window
  to `next_->run` without a host sync and syncs its own stream once the chain is done. The flags are
  allocated when the variable is set and the plan is printed at startup
  (`src/program/generate.cpp:9036-9048`).
- It is **off beside `--pipeline-windows`** (that loop overlaps the stages its own way), on one GPU,
  and for **batch windows** (several conversations keep the serial order).
- It is **not on the prompt path**: the prompt's chunk hand-off is a different object in
  `src/prefill/prefill.cpp` (its own pieces/events, gated by `STRATA_PREFILL_PIPE`, which is the
  -21 % regression of the layered record). The `prompt tok/s` column of these arms therefore cannot
  read the flag; what it shows is this box's drift, which is why it is reported as no reading.

## The box, the prompt, the harness

Two Instinct MI50 32 GB (gfx906), 68 GB RAM, IQ3_S (~62 GB resident), the **layer-3 stack of the
[2026-11-05 layered record](../20261105-layered-prefill/README.md)** in every arm - the ported
repacked-int8 prompt path plus the levers the engine already had:

```
STRATA_PF_RP=1 STRATA_DENSE_MMQ=1 STRATA_DENSE_RP=1 STRATA_MMQ_KQ=1
STRATA_HIP_PROMPT_F16=1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1
```

One arm = one engine start (`engine=0.1.49.2`, the L3 build, `md5 85693dc7...`) with
`--max-context 32768 --prefill 2048 --kv int8 --spec 4 --spec-min-p 0.5 --layer-split 24
--resident-budget-gib 20 --conversation-cache-mib 4096 --pool-workers 6 --pcie-frac 0
--vram-reserve-mib 0`, the seeded 12-word prompt of `bench/gfx906/rp-ab.sh`, and **125-128 tokens
generated** so the decode window can be timed (the layered record's arms generated 8 and could not).

The generation window is what this needs: the arms ask for 128 and the engine reports 125 (the
lookup/MTP path consumes the rest). That is also why the prompt is **32,500 tokens** and not the
layered record's 32,700: `prompt + max_new` must fit the context, and 32,700 + 128 > 32,768 - the
session's first attempt was refused by the engine (`ERR prompt (32700 tokens) + max_new (128)
exceeds the context (32768)`). A 0.6 % shorter prompt moves the context mix by less than the spread
below.

Six arms, interleaved so that a drift in the box cannot read as the flag:
`base1, ovl1, base2, ovl2, base3, ovl3`, in one container (one engine start per arm, one page
cache). Harness: `bench/gfx906/overlap-ab.sh` (new; one container, arms in one run, a WRONG-PACK
assertion and the hand-off line in every arm's summary). Summary: `bench/parse-arms.py`.

## The arms

| arm | hand-off | prompt tok/s | decode tok/s | decode ms | ms/token | drafts | ids |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base1 | serial | 701.3 | 32.6 | 3829 | 30.63 | 0/17 | 125 |
| ovl1 | overlapped | 711.4 | 33.0 | 3784 | 30.27 | 0/17 | 125 |
| base2 | serial | 769.9 | 32.6 | 3835 | 30.68 | 0/17 | 125 |
| ovl2 | overlapped | 732.7 | 33.0 | 3790 | 30.32 | 0/17 | 125 |
| base3 | serial | 762.3 | 31.1 | 4013 | 32.10 | 0/17 | 125 |
| ovl3 | overlapped | 652.4 | 32.5 | 3848 | 30.78 | 0/17 | 125 |

Paired inside each bracket: **-1.18 %, -1.17 %, -4.11 %** of decode ms/token. The three base arms
span 30.63-32.10 ms (4.8 %), the three flag arms 30.27-30.78 (1.7 %).

Two things make the pairing trustworthy rather than a coincidence of medians:

- the arms generate the **same tokens** (identical ids in all six), so the work per window is
  identical and the only variance left is machine noise;
- the flag's effect is a **fixed amount per window**, and every window here is one row (below), so
  it is a fixed amount per token.

## The reading

**+1.2 % decode**, i.e. **0.36 ms off a 30.7 ms token** - one stream sync plus one graph launch per
window, which is the whole of what the flag removes. It does not travel from the 7900 XTX because
the share it removes is not the same:

- there, serial decode was **53 tok/s against a single card's ~70** - the hand-off and the host
  syncs were roughly a quarter of the token time, so removing them recovered nearly all of it;
- here a token is **30.7 ms of GPU work across both cards** (24 + 24 layers over PCIe 3.0 x16), and
  one sync per window costs 0.36 ms of it. The slower the cards, the smaller the host's share, and
  gfx906 is slow per token.

**0 accepted drafts makes this the ceiling, not the typical case.** On this prompt the MTP draft
proposes 17 times and is never accepted, so every decode window is a single row and the flag saves
one hand-off per *token*. Where drafts are accepted the same window carries 2-5 rows - one hand-off
per window, i.e. per 2-5 tokens - and the relative gain shrinks accordingly. A drafts-accepted
prompt is the next thing to measure, and it should read *smaller* than +1.2 %, not larger.

**The prompt column reads nothing.** 762.3 against 711.4 is -6.7 %, inside this box's noise: the
arms themselves span 652.4-769.9 (18 %), the layered record's identically configured arms spanned
633.5-676.7 (6.8 %), and the flag's code is not in the prompt's hand-off path at all.

## A trap this session walked into (fixed in the harness)

The runtime image's entrypoint exports the deployment's defaults before it runs any command -
`docker-gfx906/entrypoint.sh:24`:

```sh
export MODEL="${MODEL:-IQ2_XS}" PACK="${PACK:-$(printf '%s' "${MODEL:-IQ2_XS}" | tr 'A-Z' 'a-z')}"
```

so inside `docker run <image> <script>` the environment already holds **`PACK=iq2_xs`** and
`MODEL=IQ2_XS`. A harness that writes `PACK=${PACK:-/data/packs/iq3_s}` (as
`bench/gfx906/rp-ab.sh` does) then benchmarks the **previous quant**. The first arm of this session
did exactly that and died at startup with the misleading pair

```
iq2_xs has no native_experts.txt, so it is read as the canonical Q2_0 pack, whose CPU kernel needs AVX-512
strata: this CPU cannot run the expert kernel: missing AVX512F AVX512BW AVX512VL AVX512-VNNI AVX512-VBMI
```

which reads like a CPU problem and is a pack problem. Two fixes, both in this branch:

- `bench/gfx906/overlap-ab.sh` reads `PACK_DIR` (never `PACK`), logs its argv in every arm's
  `cmd.txt`, and prints `WRONG-PACK` in the arm's summary if the engine did not resolve the folder;
- `bench/gfx906/rp-ab.sh` now ignores an inherited `PACK` that is not an absolute path, so a re-run
  of the 2026-11-02 record cannot silently measure IQ2_XS.

Also fixed there: an arm whose `prompt + max_new` does not fit the context is refused before the
engine starts, instead of after two and a half minutes of loading.

## What this changes

- **The deployment turns it on**: `docker-compose.yml` now sets `STRATA_SPLIT_OVERLAP` with `1` as
  the default (`=0` opts out), because it is free, bit-identical and worth 1.2 % of decode.
- **`docs/GFX906.md`** gets one line in the deployment section for it.
- Nothing in the layered record moves: this is a decode lever, measured on top of the layer-3 stack,
  and its 1.2 % is not a prompt gain.

## What is still open

- **A drafts-accepted prompt** (chat, code, tool call) - MTP accepts drafts there, windows are 2-5
  rows, and the expected gain is smaller than +1.2 %. That is the number a deployment sees.
- **`--pipeline-windows 2`** - the other overlapped loop, mutually exclusive with this one and off by
  default outside gfx1200/gfx1201; it needs its own A/B, not a combination.
- **Three or more stages / three cards**: middle stages take two hand-offs, and `handoff_failed`
  covers them (`verify.cpp:3110`); untested here.
- **Batch windows** (`--batch-groups`), which keep the serial order by design - if several
  conversations matter more than one, that loop is where the decode lever would have to go.
- Decode at the deployment's 262,144-token context: these arms are at 32K on purpose (the benchmark
  configuration), and the per-window sync cost does not grow with context, so the *relative* gain
  there should be smaller still - the token is longer, the sync is the same.

## Reproduce

```sh
# on the host, from the checkout: stops the deployment, runs six arms in one container, brings it back
ARMS=$(printf 'base1|\novl1|STRATA_SPLIT_OVERLAP=1\nbase2|\novl2|STRATA_SPLIT_OVERLAP=1\nbase3|\novl3|STRATA_SPLIT_OVERLAP=1\n') \
  bench/gfx906/overlap-ab.sh
python3 docs/gfx906-results/20261105-split-overlap/bench/parse-arms.py /home/me/Documents/strata/data/overlap
```

`bench/` holds every arm's `cmd.txt`, `info.txt`, `stdout.log`, `stderr.log` and `summary.txt`, plus
the session log (`bench/session.log`). The token ids of every arm are in `stdout.log` (`T <id>`
lines); `parse-arms.py` prints their md5, which is the bit-exactness check.
