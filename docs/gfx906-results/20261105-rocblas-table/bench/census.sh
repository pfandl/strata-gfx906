#!/usr/bin/env bash
# The call census of one prompt chunk: every rocBLAS call the deployment's FP16-out GEMMs make, so a
# tuning table can be priced on the shapes the engine actually issues instead of on phase names.
#
#   docs/gfx906-results/20261105-rocblas-table/bench/census.sh
#   python3 docs/gfx906-results/20261105-rocblas-table/bench/parse-calls.py \
#     /home/me/Documents/strata/data/rocblas-census/rocblas.trace.csv \
#     --probe /home/me/Documents/strata/data/rocblas-census/tune-census.log
#
# One engine start through bench/gfx906/decode-ab.sh with rocBLAS's own logs on (ROCBLAS_LAYER=7 =
# trace + bench + profile): the library writes one CSV line per call, shape and strides included,
# so the inventory is the library's and not our instrumentation's.  decode-ab.sh stops the
# deployment for the arm and restarts it after, which is why this cannot run beside the server -
# the model wants both cards and all of the VRAM.
#
# PROMPT_TOKENS=4000 is two chunks of the deployed 2048 (the second one is 1951 tokens: 4000 - 2049,
# which is why the trace shows two T values and the inventory is reported per T); MTP= (empty) drops
# the drafter, whose own GEMMs are not part of the chunk; GEN_TOKENS=4 keeps the decode window short
# (its calls are T <= 4 and parse-calls.py reports them separately); SPEC=1 makes the arm's decode
# windows one row each.
#
# The trace of the record's run is bench/rocblas.trace.csv; the probe that prices it is
# bench/tune-census.log (`tune_rocblas` on the same shapes at T = 2048).
set -u -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)          # strata-gfx906/
OUT=${OUT:-/data/rocblas-census}                # inside the container; ./data/rocblas-census on the host

exec env \
  OUT="$OUT" \
  CTX="${CTX:-8192}" \
  PROMPT_TOKENS="${PROMPT_TOKENS:-4000}" \
  GEN_TOKENS="${GEN_TOKENS:-4}" \
  PREFILL="${PREFILL:-2048}" \
  MTP= SPEC=1 \
  ARMS="census1|STRATA_PREFILL_TIMING=1 ROCBLAS_LAYER=7 \
ROCBLAS_LOG_TRACE_PATH=$OUT/rocblas.trace.csv \
ROCBLAS_LOG_BENCH_PATH=$OUT/rocblas.bench.csv \
ROCBLAS_LOG_PROFILE_PATH=$OUT/rocblas.prof.yaml" \
  "$REPO/bench/gfx906/decode-ab.sh"
