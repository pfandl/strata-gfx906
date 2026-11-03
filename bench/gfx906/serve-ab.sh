#!/usr/bin/env bash
# Serve-harness prefill A/B at a fixed context: one arm per engine restart, the same prompt for
# every arm (context-bench.py seeds its vocabulary rng from the context), one JSON per arm.
#
#   bench/gfx906/serve-ab.sh <context> "<arm>" "<ENV=V ...>" ["<arm>" "<ENV=V ...>"]...
#   e.g. bench/gfx906/serve-ab.sh 65536 a4096 "PREFILL=4096" a2048 "PREFILL=2048"
#
# The arm's flag string is the environment of the restart, so the arm is what the deployment
# would run with (`PREFILL=2048`, `STRATA_PF_RP=0`, ...). Defaults are restored at the end.
#
# Overridable:
#   COMPOSE_UP  the restart command (default "docker compose up -d --force-recreate"; where docker
#               needs the group, e.g. COMPOSE_UP="sg docker -c 'docker compose up -d --force-recreate'")
#   OUT         where the JSON and the log go (default: this directory)
#   URL         the engine (default http://127.0.0.1:8080)
set -u -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE}
COMPOSE_UP=${COMPOSE_UP:-docker compose up -d --force-recreate}
URL=${URL:-http://127.0.0.1:8080}
LOG=${LOG:-$OUT/serve-ab.log}
CTX="$1"; shift
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }

while [ "$#" -ge 2 ]; do
  arm="$1"; shift; envs="$1"; shift
  say "=== arm '$arm' at $CTX: ${envs:-defaults} ===" | tee -a "$LOG"
  # shellcheck disable=SC2086
  env CONTEXT="$CTX" $envs sh -c "$COMPOSE_UP" >>"$LOG" 2>&1 || { say "up failed"; continue; }
  if python3 "$HERE/context-bench.py" --context "$CTX" --url "$URL" --out "$OUT/ab-$CTX-$arm.json" >>"$LOG" 2>&1; then
    summary=$(python3 -c "import json; d=json.load(open('$OUT/ab-$CTX-$arm.json'));
print('arm $arm: pp %s tg %s drafts %s/%s' % (d['pp_tok_s'], d['tg_tok_s'], d['drafts_accepted'], d['drafts']))")
    say "$summary" | tee -a "$LOG"
  else
    say "arm $arm failed"
  fi
done

say "restoring defaults" | tee -a "$LOG"
sh -c "$COMPOSE_UP" >>"$LOG" 2>&1
