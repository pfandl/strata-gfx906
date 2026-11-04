# RP int8-GEMM prompt stack: A/B, chunk sweep and phase profile — 2026-11-02

Machine: AMD Ryzen Threadripper 2950X (Gigabyte X399 AORUS PRO), 68 GB RAM, 2× MI50 (gfx906, 32 GB),
150 W cap per card, ROCm 10 — **not** the Dell T5810 of the earlier records in `docs/GFX906.md`, so the
two datasets are not comparable. Engine built from branch `rp-port` (this fork's `gfx906` at v0.2.4.2 +
the 9 cherry-picked RP commits of
[janus-reith/strata-gfx906](https://github.com/janus-reith/strata-gfx906)). Pack: IQ3_S native
(experts resident, `--layer-split 24`, `--kv int8`, `--prefill auto` unless stated). Prompt: 65,500
tokens of seeded random words, so no prompt-cache or n-gram reuse.

**Verdict: the RP stack (repacked int8 GEMMs on the `dp4a` path) is +6.7 % prefill at the default
auto chunks and +33 % at 4K chunks, bitwise-parity clean, and chunk size — not the stack — is the
dominant knob once RP is on (2048-3072 form a ~700 tok/s plateau, 1024 and 4096 both sit ~633).**
*Chunk claim corrected 2026-11-04: the "+6.7 %/+33 %" verdicts stand, but chunk size is **not** a
throughput knob — see the correction box in §2 (`…/20261104-chunk-sweep/`).*

## 1. Controlled A/B, one engine start per arm (`bench/`)

Same binary, same prompt, one arm per process; `rp-ab.sh` mode selects the flags (`BASELINE` = none,
`RP` = `STRATA_PF_RP`/`DENSE_MMQ`/`DENSE_RP`/`MMQ_KQ`); 8 generated tokens, MTP drafting off. The
numbers are the `prompt N tokens = … read in M ms (X tok/s)` line of each arm's `stderr.log`:

| arm | flags | prompt | pp tok/s | vs BASELINE |
| --- | --- | ---: | ---: | ---: |
| `base64k` | BASELINE | 65500 | **549.0** (119304 ms) | — |
| `rp64k` | RP | 65500 | **585.9** (111802 ms) | **+6.7 %** |
| `base4k` | BASELINE, `--prefill 4096` | 65500 | 617.1 (106146 ms) | — |
| `rp4k` | RP, `--prefill 4096` | 65500 | **730.2** (89696 ms) | **+33.0 %** vs `base64k`, +18.3 % vs `base4k` |
| `base4kmm` | BASELINE, `--prefill 4096 --mmap-experts` | 65500 | 597.3 (109669 ms) | — |
| `rp4kmm` | RP, `--prefill 4096 --mmap-experts` | 65500 | 716.5 (91415 ms) | +20.0 % vs `base4kmm` |

Read off that table: the RP gain scales with the chunk (`--prefill 4096` widens it from +7 % to
+18 %), and `--mmap-experts` **costs** 1.9 % with RP and 3.2 % without it on this box — the higher
absolute numbers reported on other gfx906 machines come from their configuration, not from mmap.

The parity arm (`bench/check/`) runs the same config with `STRATA_PF_RP_CHECK=1` and compares every
product against the reference path; both cards report:

```
rp check: 2000 products compared, 0 differ
rp check: 2000 products compared, 0 differ
rp check: 4000 products compared, 0 differ
rp check: 4000 products compared, 0 differ
```

## 2. Serve-harness chunk sweep (`serve/`)

Through the HTTP API, one container restart per arm, fixed 64K context, `PREFILL` = chunk tokens,
`max_new` 128 (`bench/gfx906/serve-ab.sh`; the JSON of each arm is shipped, plus the harness log):

| chunk | pp tok/s | tg tok/s | drafts | note |
| ---: | ---: | ---: | ---: | --- |
| 1024 | 632.6 | 39.0 | 14/23 | also stops early (`finish: stop`, 28 tokens) — tg not comparable |
| **2048** | **686.1** | 48.3 | 69/105 | default since 2026-11-02 |
| **2048** (re-run) | **700.2** | 48.4 | 69/105 | replicate |
| 3072 | 701.1 | 48.4 | 69/105 | on the plateau with 2048 |
| 4096 | 636.5 | 48.0 | 69/105 | |

So the curve is a plateau, not a slope: 2048 and 3072 agree within noise at ~700, while 1024 and
4096 each lose ~10 %. 2048 is the shipped default because it sits on the plateau with a smaller
first chunk (shorter first-token latency, finer prompt-cache checkpoints); `PREFILL` stays a one-line
env override. A `base` arm (defaults, 544.3 tok/s) is in `serve/serve-ab.log`; it is not used as a
baseline above because it is the same-config run that wandered 545-604 tok/s across that afternoon
(the `bench/` A/B in §1 measures both arms in one process precisely to avoid that).

> **Correction, 2026-11-04 — the "1024 and 4096 each lose ~10 %" reading is withdrawn.** The `a4096`
> arm is the **first arm of this block** (`serve-ab.log` 00:11:51, three hours after the previous
> block), so it is the cold-start trap this repo documents elsewhere and it has no warm 4096 reference
> to bracket it. Re-measured on the full stack with 5 / 3 / 5 arms in one session
> (`…/20261104-chunk-sweep/`): 2048, 3072 and 4096 are **one population** (770.0 / 778.3 / 777.0 tok/s,
> within-size spread 12-23 %). The absolute level above (~700) is also RP-only: the later switches take
> that harness to ~780-830. `PREFILL=2048` is unchanged, but for the checkpoints and the first chunk,
> not for throughput. The mechanism that made these five single arms look like a curve is the harness:
> `up -d --force-recreate` overlaps the new engine with the old one's teardown; `down` first cut the
> spread from 25 % to 4.4 %.

## 3. Phase profile (`phase-profile-pre-rp-2026-11-01.txt`)

`STRATA_PREFILL_TIMING=1` capture taken **before** the port landed (IQ2_XS, chunk 1024, seven
requests from 1.5K to 12.4K tokens, verbatim engine lines). It is here as the "before" side of the
profile: expert dequant+GEMM+combine is 24-27 % of the GPU timeline, gate/up GEMM ~11 TFLOP/s (~41 %
of the f16 pipe peak, i.e. compute-bound, not launch-bound), host staging 0 ms. The post-RP profile
and the "what if" audit (36K tokens, 2048 chunks, IQ3_S, with the `ms_experts_host` counter
explained) are in [docs/GFX906.md](../../GFX906.md#post-rp-phase-profile-and-what-if-audit-36k-tokens-2048-chunks-iq3_s).

## Provenance and caveats

- Each `bench/<arm>/summary.txt` is **empty**: the harness of that day grepped for `prompt
  processed`, a string this engine version does not print, so the result lines stayed in
  `stderr.log`. Nothing was edited afterwards — the numbers above are read out of those logs, and
  the logs are shipped verbatim, apart from one redaction: an aborted arm in `serve/serve-ab.log`
  (`bigchunk`, dropped after `RemoteDisconnected`, not used above) left a Python traceback carrying
  this workstation's absolute path, whose four occurrences are replaced by `<workspace>`.
- `prompt.txt` was not archived: it is deterministic (seed 3, 65,536 words from a 12-word
  vocabulary, truncated to 65,500 ids) and `rp-ab.sh` regenerates it identically.
- One IQ3_S pack, one box (the machine line above), one ROCm build: these are not cross-machine
  numbers and another gfx906 box can land elsewhere. The port moves prompt processing only — decode
  and the chat template are untouched by it, so this record makes no claim about either.
