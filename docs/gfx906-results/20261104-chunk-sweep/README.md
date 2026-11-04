# 2026-11-04 — chunk size on the deployment harness: 2048, 3072 and 4096 are the same, and the harness itself was worth ±10 %

**Question.** The 2026-11-02 sweep picked `PREFILL=2048` from one arm per size (632.6 / 686.1 / 701.1 /
636.5 for 1024 / 2048 / 3072 / 4096) and the number has been shipped since. Two things then put that
reading in doubt. First, the 2026-11-04 bench arms (`…/20261104-chunk-2048/`) put the int8 stack at
**+18.3 %** at 4096 against **+12.7 %** at 2048 — if the stack is worth more at 4096, the natural
question is why not ship 4096. Second, the serve sweep's `a4096` arm turned out to be the **cold first
arm** of its block (it starts a block three hours after the previous one ends, with no warm 4096
reference anywhere in that session) — the same trap the 2026-11-04 bench record documents for the
549.0 baseline.

**Answer: no, and the 4096 number in the shipped docs is withdrawn.** At the deployment's own harness,
2048, 3072 and 4096 are **indistinguishable** (5 / 3 / 5 arms, means 770.0 / 778.3 / 777.0 tok/s;
arms of one size spread 83-156 tok/s, 11-20 % of the mean). The stack's larger *relative* gain at 4096
is a gain against a *slower same-chunk baseline*, not a faster absolute: the bench has 617.1 stock at
4096 against 730.2 with the stack, and 577.3 stock at 2048 against 650.4 — the stack moves 4096 further
because 4096 stock is further behind. Nothing here says 4096 is faster, and the old "4096 loses ~10 %"
was the cold arm.

## Setup

Deployment harness (`bench/gfx906/serve-ab.sh` + `context-bench.py`, HTTP, one engine recreate per arm,
63,267-token fresh seeded prompt, `temperature 0`, `max_new 128`), the shipped config as
`docker compose` runs it (RP + `STRATA_DENSE_MMQ` + `STRATA_DENSE_RP` + `STRATA_HIP_PROMPT_F16` +
the two attention switches + `STRATA_PROMPT_ATTN_GFX906` + MTP + int8 KV), `CONTEXT=65536`, only
`PREFILL` varies. Engine binary = the one the image ships (`sha256 a17227badbba9702…`).

Every arm reports `cache_n 0` with `prompt_n 63267` in its `raw_timings`, i.e. **no prompt was reused**
and all 13 arms read the same prompt — the differences below are compute, not cache.

13 arms in three blocks. Blocks 1 and 2 used the script's default restart
(`docker compose up -d --force-recreate`); block 3 restarted through `docker compose down` first.

| block | # | arm | chunk | pp tok/s | prompt ms | ms/token | tg tok/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | `a2048` | 2048 | **673.5** | 93931 | 1.485 | 34.1 |
| 1 | 2 | `a4096` | 4096 | **806.6** | 78438 | 1.240 | 44.5 |
| 1 | 3 | `a2048b` | 2048 | **829.6** | 76265 | 1.205 | 43.2 |
| 1 | 4 | `a4096b` | 4096 | **802.2** | 78869 | 1.247 | 44.6 |
| 1 | 5 | `a3072` | 3072 | **690.8** | 91587 | 1.448 | 49.9 |
| 1 | 6 | `a3072b` | 3072 | **804.0** | 78692 | 1.244 | 50.2 |
| 2 | 7 | `b4096` | 4096 | **724.8** | 87294 | 1.380 | 42.7 |
| 2 | 8 | `b3072` | 3072 | **840.0** | 75320 | 1.191 | 50.3 |
| 2 | 9 | `b2048` | 2048 | **766.8** | 82504 | 1.304 | 43.2 |
| 2 | 10 | `b4096b` | 4096 | **743.7** | 85069 | 1.345 | 44.4 |
| 3 | 11 | `c2048` | 2048 | **773.6** | 81783 | 1.293 | 41.2 |
| 3 | 12 | `c4096` | 4096 | **807.6** | 78338 | 1.238 | 42.1 |
| 3 | 13 | `c2048b` | 2048 | **806.7** | 78423 | 1.240 | 43.1 |

| chunk | n | arms | mean | median | min-max |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2048 | 5 | 673.5, 829.6, 766.8, 773.6, 806.7 | 770.0 | 773.6 | 673.5-829.6 |
| 3072 | 3 | 690.8, 804.0, 840.0 | 778.3 | 804.0 | 690.8-840.0 |
| 4096 | 5 | 806.6, 802.2, 724.8, 743.7, 807.6 | 777.0 | 802.2 | 724.8-807.6 |

**No chunk-size effect is resolvable.** The between-size spread of the means is 8 tok/s (1.1 %) while
arms of one size spread 83-156 tok/s (11-20 %). Splitting the 13 arms at 1.28 ms/token puts every size
in both groups — fast (1.19-1.25 ms/token, n=7): 2048 two arms, 3072 two, 4096 three; slow (1.29-1.49,
n=6): 2048 three arms, 3072 one, 4096 two — so the split is a property of the **arm**, not of the chunk
size. Within the fast group the means are 818.2 / 822.0 / 805.5 tok/s for 2048 / 3072 / 4096, i.e. 2.1 %
apart with the ranking 3072 > 2048 > 4096. That is inside the group's own spread and is no reason to
move the default; it does rule out 4096 as *better*.

## The harness was the variance

Blocks 1-2 spread 673.5-840.0 tok/s (**25 %**) with the config changing between only two flag values,
which is far more than this box's usual ±3 % and far more than the chunk size could explain. The
suspected mechanism is startup overlap: `docker compose up -d --force-recreate` starts the new engine
while the old one may still be tearing down 60+ GB (the service sets `stop_grace_period: 120s`), so the
new engine builds its 20 GiB pinned expert copy and places its six pool workers while the host is
busy. The box is 2-NUMA-node (47.9 GB on node 0, 22.1 GB on node 1, node distance 16) and nothing in
the deployment pins memory or threads, so a bad placement is paid by every expert read for the whole
arm. Block 3 restarted through `docker compose down && sleep 25 && docker compose up -d`, i.e. the old
container is gone before the new one starts, and the spread fell to **773.6-807.6 (4.4 %)** — the same
shape this box shows elsewhere.

That is a fix for the harness, not for the engine: `bench/gfx906/serve-ab.sh` now defaults to
`docker compose down && docker compose up -d`. Any earlier serve-harness number with a wide bracket
(including the 2026-11-02 sweep this record corrects) should be re-read with that in mind: with one
arm per size and a 25 % per-arm hazard, the "plateau, not a slope" reading was mostly the hazard.

Mechanism not proven — no NUMA or placement counters were recorded. What is proven is that the spread
is not the chunk size: identical config (block 1 arms 2 and 4, both 4096) gives 806.6 and 802.2, while
the same config in block 2 gives 724.8 and 743.7.

## `--prefill` and the lend budget (#669)

Every chunk size overflows the lend budget at `RESIDENT_GIB=20`, by the same ~11 % of the lent slots:
2048 lends 639 slots and 568 fit, 4096 lends 1059 and 941 fit, 8192 lends 1891 and 1690 fit (all from
the arms' `stderr.log`: `…/20261104-chunk-2048/bench/*/stderr.log`,
`…/20261102-rp-prefill/bench/*/stderr.log`). So a bigger chunk does not buy a clean lend region — the
overflow scales with the chunk and the cost per token stays flat. #669 is therefore not a reason to
prefer 2048 over 4096. The reasons to keep 2048 are that there is no throughput to gain, that it is the
config the 256K validation and the whole 2026-11-02/03 ladder ran at, and that a smaller chunk gives a
shorter first chunk (first-token latency) and finer prompt-cache checkpoints.

## Verdict

- **Keep `PREFILL=2048`.** No measured throughput difference from 3072 or 4096, one less config change
  to re-validate, and the finer checkpoints.
- **Withdraw "4096 loses ~10 %"** from `docs/GFX906.md` and `README-gfx906.md`: it came from one cold
  arm. The corrected statement is that 1024 was the only size that looked worse (632.6, and that arm's
  request stopped after 28 tokens, so it is not a clean data point either) and that 2048-4096 are one
  population on this box.
- **Do not read the stack's +18.3 % at 4096 as "4096 is faster".** The gain is relative to the same
  chunk size's baseline; the absolute numbers are flat.

## Reproduce

```sh
# block 3, the harness form to use (old container gone before the new one starts)
cd /home/me/Documents/strata
OUT=$PWD/data/serve-chunk-sweep LOG=$PWD/data/serve-chunk-sweep/serve-ab3.log \
COMPOSE_UP="docker compose down && sleep 25 && docker compose up -d" \
  sh strata-gfx906/bench/gfx906/serve-ab.sh 65536 \
    c2048 "PREFILL=2048" c4096 "PREFILL=4096" c2048b "PREFILL=2048"
```

Each arm is ~3.2 min (engine restart and load ~1 min, 63,267-token prompt 75-95 s, 128-token
generation). The arm's JSON (`ab-65536-<arm>.json`) is the record; the engine's own stderr is not kept
because the container is recreated per arm.
