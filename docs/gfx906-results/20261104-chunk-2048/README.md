# The 2048-chunk arm: what the chunk change was worth, and what the int8 stack is worth at the shipped chunk size — 2026-11-04

Machine: the same box as the 2026-11-02/03 records (Ryzen Threadripper 2950X, 68 GB RAM, 2× MI50 32 GB,
150 W cap per card, HIP 7.14 from `mixa3607/rocm-gfx906:7.14-complete`), the same image
(`sha256:10349bcd…`), and the **shipped binary** `sha256 a17227badbba9702…` — `docker cp`'d out of the
running container and mounted at `/work/strata-shipped`, so this is the engine the deployment serves.
Same harness as the earlier records (`bench/gfx906/rp-ab.sh`), same fixed flags (`--layer-split 24`,
`--resident-budget-gib 20`, `--conversation-cache-mib 4096`, `--spec 4`, `--kv int8`,
`--pcie-frac 0`, `--vram-reserve-mib 0`, `--pool-workers 6`), same 65,500-token seeded-random prompt,
8 generated tokens, MTP drafting off. Only `--prefill 2048` differs from the 2026-11-02 A/B.

**Why this run exists.** The 2026-11-02 A/B ran at `--prefill auto`, which resolves to **8192** tokens
on this box (`strata serve: prompt chunk auto: 8192 tokens`), while every later arm — the FP16 knob, the
two attention switches, the kernel — ran at 2048. So the 549.0 → 769.7 ladder carried an unpriced chunk
step, and the int8 stack had only been measured at 8192 (+6.7 %) and 4096 (+18.3 %). This closes both.

## The arms

One engine start per arm, in this order (`bench/`):

| arm | mode | pp tok/s | note |
| --- | --- | ---: | --- |
| `warm2k` | BASELINE | 556.0 | first arm of the session — the cold-start trap |
| `base2k` | BASELINE | 588.1 | |
| `rp2k` | RP | **650.4** | the four `STRATA_*` int8 flags |
| `base2k2` | BASELINE | 566.5 | |

The two warm baselines are 3.7 % apart (588.1 / 566.5), which is this box's session noise band; the cold
first arm sits 3.8 % under their mean (577.3). The numbers are the engine's own
`prompt 65500 tokens = 0 reused + 65500 read in M ms (X tok/s)` line of each arm's `stderr.log`; the
per-chunk `PP` line of `stdout.log` agrees to 0.2 % (e.g. `rp2k`: 100502 ms / 651.7 tok/s).

## Verdict

1. **The int8 stack at the shipped chunk size is +12.7 %** — 650.4 against the baseline mean 577.3,
   i.e. +10.6 % against the higher reference and +14.8 % against the lower. The 2026-11-02 headline
   `+6.7 %` was measured at 8192 chunks, where *both* arms were partly reading experts off the SSD
   (#669: an 8192-token chunk lends 1891 slots and only 1690 of their experts fit in RAM, so the rest
   are read per chunk) — that masked the stack's advantage. At 4096 it was +18.3 %, at 2048 it is
   +12.7 %: the stack's gain is chunk-dependent, and 2048 is the chunk the deployment ships. The
   dependence does **not** rank the chunk sizes — that is measured in `…/20261104-chunk-sweep/`, where
   2048, 3072 and 4096 are one population in absolute terms (770.0 / 778.3 / 777.0 tok/s).
2. **The chunk change 8192 → 2048 is worth ~+1.3 %**, not the ~12 % an elimination over the ladder
   suggested: the cold baseline at 8192 (549.0) against the cold baseline at 2048 (556.0). The rest of
   that residual is the stack's larger gain at 2048 plus page-cache warmth.
3. **Page-cache warmth is worth ~+4 %** at 2048 (cold 556.0 against the warm mean 577.3). The
   2026-11-02 baseline 549.0 was likewise the first arm of its session, so the `+40 %` headline carries
   roughly +5 % of conditions — chunk size and warmth — that are not code.
4. **The four lever A/Bs compose to the measured total**: 577.3 × 1.127 (stack) × 1.104 (FP16) × 1.058
   (switches) = **760.0** against 769.7 measured, +1.3 % — inside the noise band. So the code part of
   549.0 → 769.7 is **+31.6 %** by composition and **+33.3 %** end to end (577.3 → 769.7), and the
   remaining ~+5 % is the conditions above.

## What this changes in the write-ups

- The stack's quoted gain becomes **+12.7 % at the shipped chunk size** (+6.7 % at 8192, +18.3 % at
  4096), instead of +6.7 % or the record's cross-chunk +33 %.
- The ladder's chunk step is **measured, not inferred**: +1.3 %, with the warmth caveat attached.
- The `549.0` baseline is identified as a **cold first arm**, which is why the like-for-like before is
  577.3 and the code gain is +33 %, not +40 %.

## Reproduce

```sh
docker compose stop          # the engine needs the cards to itself
cd /home/me/Documents/strata
docker cp strata:/opt/strata/build-906/strata rp-build/strata-shipped   # before stopping, or from the image
sha256sum rp-build/strata-shipped   # a17227badbba970241f233571c6044c868b76839088b34f09a10c9f87d52a5f7

for arm in "warm2k BASELINE" "base2k BASELINE" "rp2k RP" "base2k2 BASELINE"; do
  set -- $arm
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
    -v "$PWD/data":/data -v "$PWD/rp-build":/work -v "$PWD/strata-gfx906":/repo:ro \
    -e MODEL=IQ3_S -e PACK=/data/packs/iq3_s -e OUT=/data/rp-opt -e CTX=65536 \
    -e PROMPT_TOKENS=65500 -e PREFILL=2048 -e STRATA_PREFILL_TIMING=1 \
    -e ENGINE=/work/strata-shipped \
    strata-gfx906:local /repo/bench/gfx906/rp-ab.sh "$1" "$2"
done
docker compose start
```

Each arm is ~4 minutes on this box once the page cache is warm (load ~2 min, 65,500 tokens ~2 min).
