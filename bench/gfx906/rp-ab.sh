#!/bin/sh
# One RP A/B engine run: a fixed random-word prompt of PROMPT_TOKENS tokens, one engine start,
# the RP flags chosen by the mode. Run INSIDE the runtime image (it carries /opt/strata, the venv
# tokenizer, the expert profile and /data), with the build at /work and the data at /data:
#
#   docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
#     -v "$PWD/data":/data -v "$PWD/rp-build":/work strata-gfx906:local \
#     /work/bench/gfx906/rp-ab.sh rp4k RP
#
#   rp-ab.sh <label> [BASELINE|RP|RP_CHECK]
#
#   BASELINE   no RP flags                       RP        all four RP flags on
#   RP_CHECK   the four flags + STRATA_PF_RP_CHECK=1 (bitwise parity against the reference path)
#
# Arms of the 2026-11-02 record: `base64k BASELINE` and `rp64k RP` (PREFILL=auto), `base4k`/`rp4k`
# (PREFILL=4096), `base4kmm`/`rp4kmm` (PREFILL=4096 EXTRA=--mmap-experts), `check RP_CHECK`.
#
# Results go to $OUT/<label>/ (info.txt, stdout.log, stderr.log, summary.txt). The pp number is
# the "prompt N tokens = 0 reused + N read in M ms (X tok/s)" line of stderr.log; the parity line
# is "rp check: 4000 products compared, 0 differ" (one per card).
LABEL="$1"; MODE="${2:-RP}"
[ -n "$LABEL" ] || { echo "usage: rp-ab.sh <label> [BASELINE|RP|RP_CHECK]" >&2; exit 2; }

OUT=${OUT:-/data/rp}
PACK=${PACK:-/data/packs/iq3_s}
# PACK= arrives from the image's entrypoint, which exports the deployment's default pack name
# (docker-gfx906/entrypoint.sh: PACK=iq2_xs) before it runs the command it was given; a session that
# picked that up would benchmark the previous quant and fail on a CPU that has no AVX-512.  An
# inherited value that is not an absolute path is ignored (2026-11-05, split-overlap record).
case "$PACK" in /*) ;; *) PACK=/data/packs/iq3_s ;; esac
GGUF=${GGUF:-/data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
PLE=${PLE:-/data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf}
MTP=${MTP:-/data/mtp/rt}
PROFILE=${PROFILE:-/opt/strata/data/expert-profile.bin}
ENGINE=${ENGINE:-/work/build-906/strata}
PYTHON=${PYTHON:-/opt/strata/.venv/bin/python}
TOOLS=${TOOLS:-/opt/strata/tools}
CTX=${CTX:-65536}
PROMPT_TOKENS=${PROMPT_TOKENS:-65500}
GEN_TOKENS=${GEN_TOKENS:-8}
PREFILL=${PREFILL:-auto}
EXTRA=${EXTRA:-}                      # extra engine args, e.g. EXTRA=--mmap-experts
SPLIT=${SPLIT:-24}
RESIDENT_GIB=${RESIDENT_GIB:-20}
CACHE_MIB=${CACHE_MIB:-4096}
POOL_WORKERS=${POOL_WORKERS:-6}

DIR=$OUT/$LABEL
rm -rf "$DIR"; mkdir -p "$DIR"
FIFO=${TMPDIR:-/tmp}/rp-ab.$$.fifo
rm -f "$FIFO"; mkfifo "$FIFO"

if [ "$MODE" = "RP" ] || [ "$MODE" = "RP_CHECK" ]; then
  export STRATA_PF_RP=1 STRATA_DENSE_MMQ=1 STRATA_DENSE_RP=1 STRATA_MMQ_KQ=1
fi
if [ "$MODE" = "RP_CHECK" ]; then
  export STRATA_PF_RP_CHECK=1
fi

export GPU_MAX_HW_QUEUES=8 HSA_ENABLE_SDMA=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export AB_GGUF="$GGUF" AB_TOKENS="$PROMPT_TOKENS" AB_TOOLS="$TOOLS"

# the fixed prompt: seeded random words, encoded with the pack's tokenizer, truncated to PROMPT_TOKENS
"$PYTHON" - <<'EOF' > "$DIR/prompt.txt"
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
NTOK=$(tr -cd ',' < "$DIR/prompt.txt" | wc -c)
echo "label=$LABEL mode=$MODE ntokens=$((NTOK+1))" | tee "$DIR/info.txt"

# shellcheck disable=SC2086
"$ENGINE" --serve \
  --pack "$PACK" --native "$GGUF" --ple-gguf "$PLE" \
  --mtp "$MTP" --suffix-draft 0 \
  --expert-profile "$PROFILE" --expert-cache auto \
  --prefill "$PREFILL" $EXTRA --spec 4 --spec-min-p 0.5 \
  --max-context "$CTX" --kv int8 \
  --pool-workers "$POOL_WORKERS" --pcie-frac 0 --vram-reserve-mib 0 \
  --conversation-cache-mib "$CACHE_MIB" --resident-budget-gib "$RESIDENT_GIB" \
  --layer-split "$SPLIT" < "$FIFO" > "$DIR/stdout.log" 2> "$DIR/stderr.log" &
EPID=$!
exec 3>"$FIFO"
# NOTE: grep on a plain file exits at EOF, so wait by polling (the `grep -m1` form only blocks on
# a pipe) - the engine needs real minutes to load.
ok=0
for _ in $(seq 1 150); do   # up to 12.5 min
  grep -qm1 '^READY' "$DIR/stdout.log" && { ok=1; break; }
  kill -0 $EPID 2>/dev/null || break
  sleep 5
done
[ "$ok" = "1" ] || { echo "NO-READY (engine died or too slow)" | tee -a "$DIR/info.txt"; kill $EPID 2>/dev/null; tail -25 "$DIR/stderr.log" | tee -a "$DIR/info.txt"; exit 1; }
echo "GEN $GEN_TOKENS $(cat "$DIR/prompt.txt")" >&3
ok=0
for _ in $(seq 1 360); do   # up to 30 min
  grep -qm1 '^DONE' "$DIR/stdout.log" && { ok=1; break; }
  sleep 5
done
[ "$ok" = "1" ] || echo "NO-DONE" | tee -a "$DIR/info.txt"
exec 3>&-
wait $EPID
echo "EXIT rc=$?" | tee -a "$DIR/info.txt"

echo "=== result ===" | tee "$DIR/summary.txt"
grep -a 'prompt .* tok/s' "$DIR/stderr.log" | tee -a "$DIR/summary.txt"
grep -a 'rp check' "$DIR/stderr.log" | head -8 | tee -a "$DIR/summary.txt"
