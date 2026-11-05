#!/usr/bin/env bash
# Decode A/B on the two-card gfx906 split: one arm per engine start, the layer-3 stack in every arm,
# one flag difference between the arms, one prompt, one container.
#
#   bench/gfx906/decode-ab.sh
#
# Written for STRATA_SPLIT_OVERLAP (2026-11-05 split-overlap record) and reused for every decode
# lever since (`STRATA_GR_SPLIT`, ...): the arm's own string is the only thing that differs, and
# GEN_TOKENS is large enough to time the decode window (the prompt records generate 8 tokens and
# cannot).  Two things in an arm's summary decide whether it is a measurement at all:
#
#   - the hand-off line: "overlapped: STRATA_SPLIT_OVERLAP=1" or "serial order", so an arm whose
#     flag did not take is reported NOT-APPLIED instead of being read as a null result;
#   - the WRONG-PACK line: the runtime image's entrypoint exports PACK=iq2_xs, so a harness that
#     reads ${PACK} benchmarks the previous quant (this file reads PACK_DIR only).
#
# The arms: 32,500-token prompt (the layered record's seeded word list), --max-context 32768,
# --prefill 2048, --kv int8, --spec 4, --layer-split 24, 128 generated tokens, in one container (one
# engine start per arm, page cache shared across the arms).
#
# Overridable: COMPOSE_DIR DATA BUILD IMAGE for the host half; CTX PROMPT_TOKENS GEN_TOKENS
# PREFILL SPLIT RESIDENT_GIB CACHE_MIB POOL_WORKERS ARMS OUT EXTRA_INSIDE for the session.
# EXTRA_INSIDE is a script path inside the container run once after the arms (component benches,
# parity checks) - its output goes to $OUT/extra.log.


set -u -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- host half
if [ "${1:-}" != "inside" ]; then
  COMPOSE_DIR=${COMPOSE_DIR:-/home/me/Documents/strata}
  DATA=${DATA:-$COMPOSE_DIR/data}
  BUILD=${BUILD:-$COMPOSE_DIR/rp-build}
  IMAGE=${IMAGE:-strata-gfx906:local}
  mkdir -p "$DATA/overlap"
  say "stopping the deployment (the arms need both cards and all of the VRAM)"
  ( cd "$COMPOSE_DIR" && docker compose down ) >>"$DATA/overlap/host.log" 2>&1 || exit 1
  # an engine that is still loading holds the pack and 20 GiB of RAM; the first session's first
  # arm read the wrong pack folder while the deployment was still coming up behind it
  if docker ps --format '{{.Names}}' | grep -qx strata; then
    say "the deployment container is still up - stopping it"
    docker stop strata >>"$DATA/overlap/host.log" 2>&1
  fi
  sleep 5
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --group-add "$(stat -c %g /dev/kfd)" --ipc=host --user "$(id -u):$(id -g)" \
    -v "$DATA":/data -v "$BUILD":/work -v "$HERE":/work-bench:ro \
    -e CTX -e PROMPT_TOKENS -e GEN_TOKENS -e PREFILL -e SPLIT -e RESIDENT_GIB \
    -e CACHE_MIB -e POOL_WORKERS -e ARMS -e OUT -e EXTRA_INSIDE -e MTP -e SPEC \
    "$IMAGE" /work-bench/decode-ab.sh inside "$@"
  rc=$?
  say "bringing the deployment back"
  ( cd "$COMPOSE_DIR" && docker compose up -d ) >>"$DATA/overlap/host.log" 2>&1
  exit $rc
fi
shift

# ------------------------------------------------------------ container half
OUT=${OUT:-/data/overlap}
# PACK_DIR, not PACK: the image ships ENV PACK=iq2_xs (and MODEL=IQ2_XS) for its entrypoint,
# so a script that reads ${PACK} benchmarks the previous quant.  An arm whose engine reads
# another folder is caught below (WRONG-PACK).
PACK_DIR=${AB_PACK:-/data/packs/iq3_s}
GGUF=${GGUF:-/data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
PLE=${PLE:-/data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf}
# MTP= (empty) drops the draft layer: a numerically-different lever changes the tokens, and with
# the drafter on that also changes how many draft chains run, so the window mix stops being
# comparable.  Without it every window is one row and the arms are comparable again.
MTP=${MTP-/data/mtp/rt}
# SPEC=1 (with MTP= empty) makes every verify window one row: for a lever that changes the sums and
# therefore the tokens, that is the only shape in which decode tok/s stays comparable.
SPEC=${SPEC:-4}
PROFILE=${PROFILE:-/opt/strata/data/expert-profile.bin}
ENGINE=${ENGINE:-/work/build-906/strata}
PYTHON=${PYTHON:-/opt/strata/.venv/bin/python}
TOOLS=${TOOLS:-/opt/strata/tools}
CTX=${CTX:-32768}
# 32,500 and not the layered record's 32,700: a decode lever needs a generation window long
# enough to time (128 tokens, ~6 s), and the engine refuses prompt + max_new > the context.  The
# 0.6 % shorter prompt moves the context mix by less than the arm-to-arm spread it is measured in.
PROMPT_TOKENS=${PROMPT_TOKENS:-32500}
GEN_TOKENS=${GEN_TOKENS:-128}
if [ $(( PROMPT_TOKENS + GEN_TOKENS + 64 )) -gt "$CTX" ]; then
  echo "PROMPT_TOKENS=$PROMPT_TOKENS + GEN_TOKENS=$GEN_TOKENS does not fit CTX=$CTX" >&2; exit 2
fi
PREFILL=${PREFILL:-2048}
SPLIT=${SPLIT:-24}
RESIDENT_GIB=${RESIDENT_GIB:-20}
CACHE_MIB=${CACHE_MIB:-4096}
POOL_WORKERS=${POOL_WORKERS:-6}
LOG=$OUT/session.log
mkdir -p "$OUT"

# every arm runs the layer-3 stack of the layered record: the ported repacked-int8 prompt path
# plus the levers the engine already had. The arm's own string adds one lever to it.
STACK="STRATA_PF_RP=1 STRATA_DENSE_MMQ=1 STRATA_DENSE_RP=1 STRATA_MMQ_KQ=1 \
STRATA_HIP_PROMPT_F16=1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1"

# interleaved, so a drift in the box cannot read as the flag: base, flag, base, flag
ARMS=${ARMS:-"base1|
ovl1|STRATA_SPLIT_OVERLAP=1
base2|
ovl2|STRATA_SPLIT_OVERLAP=1"}

export HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1 GPU_MAX_HW_QUEUES=8
export AB_GGUF="$GGUF" AB_TOKENS="$PROMPT_TOKENS" AB_TOOLS="$TOOLS"

# one prompt for the whole session: the seeded word list of bench/gfx906/rp-ab.sh, so the tokens
# are the ones every earlier arm of this box was measured on
if [ ! -s "$OUT/prompt-$PROMPT_TOKENS.txt" ]; then
  "$PYTHON" - <<'EOF' > "$OUT/prompt-$PROMPT_TOKENS.txt"
import os, random, sys
sys.path.insert(0, os.environ['AB_TOOLS'])
import strata_tokenizer as st
random.seed(3)
words = ['lighthouse','quartz','ferret','meridian','cobalt','harvest','velvet','onyx','fjord','canyon','ember','tundra']
text = ' '.join(random.choice(words) for _ in range(65536))
tok = st.Tokenizer.from_gguf(os.environ['AB_GGUF'])
ids = tok.encode(text, parse_special=True)[:int(os.environ['AB_TOKENS'])]
print(','.join(map(str, ids)))
EOF
fi
PROMPT=$(cat "$OUT/prompt-$PROMPT_TOKENS.txt")
NTOK=$(( $(tr -cd ',' < "$OUT/prompt-$PROMPT_TOKENS.txt" | wc -c) + 1 ))


# shellcheck disable=SC2086
arm_run() {   # label, env-string
  local label=$1 envs=$2 dir=$OUT/$label
  rm -rf "$dir"; mkdir -p "$dir"
  printf 'label=%s ntokens=%s gen=%s prefill=%s split=%s env=[%s]\n' \
    "$label" "$NTOK" "$GEN_TOKENS" "$PREFILL" "$SPLIT" "$envs" | tee "$dir/info.txt"
  local fifo=$dir/stdin.fifo
  rm -f "$fifo"; mkfifo "$fifo"
  # the exact argv and env, so an arm can be replayed from the record
  printf 'env %s %s %s --serve --pack %s --native %s --ple-gguf %s --mtp %s --spec %s
  --expert-profile %s --expert-cache auto --prefill %s --spec 4 --spec-min-p 0.5 --max-context %s --kv int8
  --pool-workers %s --pcie-frac 0 --vram-reserve-mib 0 --conversation-cache-mib %s --resident-budget-gib %s
  --layer-split %s\n' "$STACK" "$envs" "$ENGINE" "$PACK_DIR" "$GGUF" "$PLE" "$MTP" "$SPEC" "$PROFILE" \
    "$PREFILL" "$CTX" "$POOL_WORKERS" "$CACHE_MIB" "$RESIDENT_GIB" "$SPLIT" > "$dir/cmd.txt"
  env $STACK $envs "$ENGINE" --serve \
    --pack "$PACK_DIR" --native "$GGUF" --ple-gguf "$PLE" \
    ${MTP:+--mtp "$MTP" --suffix-draft 0} \
    --expert-profile "$PROFILE" --expert-cache auto \
    --prefill "$PREFILL" --spec 4 --spec-min-p 0.5 \
    --max-context "$CTX" --kv int8 \
    --pool-workers "$POOL_WORKERS" --pcie-frac 0 --vram-reserve-mib 0 \
    --conversation-cache-mib "$CACHE_MIB" --resident-budget-gib "$RESIDENT_GIB" \
    --layer-split "$SPLIT" < "$fifo" > "$dir/stdout.log" 2> "$dir/stderr.log" &
  local ep=$!
  exec 3>"$fifo"
  local ok=0
  for _ in $(seq 1 150); do   # up to 12.5 min to load (the engine needs real minutes)
    grep -qm1 '^READY' "$dir/stdout.log" && { ok=1; break; }
    kill -0 $ep 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" != 1 ]; then
    echo "NO-READY" | tee -a "$dir/info.txt"; tail -20 "$dir/stderr.log" | tee -a "$dir/info.txt"
    kill $ep 2>/dev/null; exec 3>&-; wait $ep 2>/dev/null; return 1
  fi
  if ! grep -qa "native pack: $PACK_DIR experts" "$dir/stderr.log"; then
    echo "WRONG-PACK: the engine did not resolve $PACK_DIR as its native pack" | tee -a "$dir/info.txt"
  fi
  echo "GEN $GEN_TOKENS $PROMPT" >&3
  ok=0
  for _ in $(seq 1 240); do   # up to 20 min for prompt + decode
    grep -qm1 '^DONE' "$dir/stdout.log" && { ok=1; break; }
    grep -qm1 '^ERR ' "$dir/stdout.log" && { ok=2; break; }
    kill -0 $ep 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" = 2 ]; then
    echo "ERR-FROM-ENGINE" | tee -a "$dir/info.txt"; grep -a '^ERR ' "$dir/stdout.log" | tee -a "$dir/info.txt"
    exec 3>&-; wait $ep 2>/dev/null; return 1
  fi
  [ "$ok" = 1 ] || echo "NO-DONE" | tee -a "$dir/info.txt"
  exec 3>&-
  wait $ep; echo "EXIT rc=$?" | tee -a "$dir/info.txt"
  grep -a 'prompt .* tok/s' "$dir/stderr.log" | tee "$dir/summary.txt"
  if grep -qa 'overlapped: STRATA_SPLIT_OVERLAP=1' "$dir/stderr.log"; then
    echo "hand-off: OVERLAPPED (the flag took)" | tee -a "$dir/summary.txt"
  elif grep -qa 'STRATA_SPLIT_OVERLAP is off beside' "$dir/stderr.log"; then
    echo "hand-off: NOT-APPLIED (refused beside --pipeline-windows)" | tee -a "$dir/summary.txt"
  else
    echo "hand-off: serial order (off)" | tee -a "$dir/summary.txt"
  fi
}

: > "$LOG"
while IFS='|' read -r label envs; do
  label=$(printf '%s' "$label" | tr -d ' ')
  envs=$(printf '%s' "$envs" | sed 's/^ *//; s/ *$//')
  [ -n "$label" ] || continue
  say "=== arm $label at $CTX [${envs:-the layer-3 stack only}] start ===" | tee -a "$LOG"
  arm_run "$label" "$envs" >>"$LOG" 2>&1 || { say "=== arm $label FAILED - session stopped ===" | tee -a "$LOG"; exit 1; }
  say "=== arm $label done ===" | tee -a "$LOG"
done <<< "$ARMS"

say "session done" | tee -a "$LOG"
for d in "$OUT"/*/; do [ -s "$d/summary.txt" ] && sed "s|^|$(basename "$d"): |" "$d/summary.txt"; done

# an optional component-level script (fused_gr_bench, a parity test, ...) run once the arms are
# done and the GPUs are free of an engine: its output is $OUT/extra.log and goes into the record
if [ -n "${EXTRA_INSIDE:-}" ]; then
  say "=== extra: $EXTRA_INSIDE ===" | tee -a "$LOG"
  if OUT=$OUT bash "$EXTRA_INSIDE" > "$OUT/extra.log" 2>&1; then
    say "=== extra done ($OUT/extra.log) ===" | tee -a "$LOG"
    tail -30 "$OUT/extra.log" | tee -a "$LOG"
  else
    say "=== extra FAILED (rc=$?, see $OUT/extra.log) ===" | tee -a "$LOG"
  fi
fi
