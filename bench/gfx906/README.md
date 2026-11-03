# gfx906 RP-port harnesses

The scripts that produced [docs/gfx906-results/20261102-rp-prefill](../../docs/gfx906-results/20261102-rp-prefill/README.md)
and the numbers in [docs/GFX906.md](../../docs/GFX906.md). They are MI50/MI60 (gfx906) harnesses: written
for a two-card box serving one IQ3_S pack, with the engine built for gfx906 and the RPC bits a normal
`docker compose` deployment already has.

| script | what it measures |
| --- | --- |
| `rp-check.sh` | compile check of the engine with the RP port, host-side, no image build (incremental in `WORK`) |
| `rp-ab.sh` | one A/B arm on a fixed prompt: the pp tok/s of one engine start, plus the `RP_CHECK` bitwise parity line |
| `serve-ab.sh` | the same A/B through the HTTP API, one arm per container restart (`COMPOSE_UP`), one JSON per arm |
| `context-bench.py` | one request at a context size, prints + writes the pp/tg record (`serve-ab.sh` calls it) |
| `parab.py` | N concurrent completions with distinct prompts: the parallel-serving plateau |

## Requirements

- **`rp-check.sh`** needs a builder image with ROCm, cmake, ninja and clang, tagged
  `strata-gfx906-builder` (build it from `docker/Dockerfile.builder`, or set `IMAGE=`). `WORK` is a
  host directory it keeps the build in (`~/.cache/strata-gfx906-rp-check` by default); it mounts the
  checkout read-only at `/src`.
- **`rp-ab.sh`** runs *inside the runtime image*, which is where `/opt/strata` (the venv tokenizer,
  `serve/chat_golden.json`, `data/expert-profile.bin`) exists. Mount the pack+models at `/data` and the
  build at `/work`:

  ```
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
    -v "$PWD/data":/data -v "$PWD/rp-build":/work strata-gfx906:local \
    /work/bench/gfx906/rp-ab.sh rp4k RP
  ```

  `OUT` (default `/data/rp`) collects `<label>/{info,prompt.txt,stdout.log,stderr.log,summary.txt}`.
  The number to read is the `prompt N tokens = 0 reused + N read in M ms (X tok/s)` line of
  `stderr.log`. Arms: `BASELINE` (no flags), `RP` (all four), `RP_CHECK` (plus
  `STRATA_PF_RP_CHECK=1`). `PREFILL=` and `EXTRA=` select the 4K (`PREFILL=4096`) and mmap
  (`EXTRA=--mmap-experts`) arms; `PACK`, `GGUF`, `PLE`, `MTP`, `PROFILE`, `SPLIT`, `RESIDENT_GIB`,
  `CACHE_MIB`, `POOL_WORKERS`, `CTX`, `PROMPT_TOKENS` are all overridable.
- **`serve-ab.sh`** restarts the deployment between arms, so it needs the compose command:
  `COMPOSE_UP` (default `docker compose up -d --force-recreate`; where docker needs the group,
  `COMPOSE_UP="sg docker -c 'docker compose up -d --force-recreate'"`). One arm is
  `<name> "<ENV=V ...>"`, e.g.

  ```
  bench/gfx906/serve-ab.sh 65536 a4096 "PREFILL=4096" a2048 "PREFILL=2048" a3072 "PREFILL=3072"
  ```

  It restores the defaults (`COMPOSE_UP` with no extra env) at the end. `URL` (default
  `http://127.0.0.1:8080`), `OUT`, `LOG` are overridable; `--model`/`--max-new` live in
  `context-bench.py`.
- **`parab.py`** takes `--url`; no build needed.

## Caveats worth knowing before comparing numbers

- `serve-ab.sh` records one JSON per arm and nothing else: the arm's flags live in the log line, not
  in the JSON. Keep the log (`OUT/serve-ab.log`) next to the JSONs.
- The prompt is a random-word sequence seeded from the context size, so arms at the same context are
  comparable to each other, not necessarily to a chat prompt.
- `max_tokens` is fixed; if an arm stops early (`finish: stop`) its tg/drafts numbers are not
  comparable to the arms that filled the window (1024 did that in the 2026-11-02 sweep).
- `rp-ab.sh` polls for `READY`/`DONE` on a plain file, so it sleeps rather than blocking on a pipe; a
  cold 64K run takes ~20 minutes per arm on this box.
