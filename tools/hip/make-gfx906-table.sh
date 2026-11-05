#!/usr/bin/env bash
# The gfx906 table itself: the 10 shapes one prompt chunk actually calls (a call census, not the tool's
# 14-shape default list - docs/gfx906-results/20261105-rocblas-table/) at the token buckets the gfx1030
# table uses, so the nearest-bucket lookup behaves the same on both cards.
#
#   docker run --rm -v .../strata-gfx906:/src -v .../rp-build:/work strata-gfx906-builder:latest \
#     bash /src/tools/hip/make-gfx906-table.sh
#
# Writes tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt (the header carries the arch and the rocBLAS
# version string; any other pair makes Gemm refuse the file).  ~40 minutes: 80 shape/bucket pairs, each
# timing the default kernel and the 164 solutions rocBLAS lists for the shape.
set -eu
BIN=${BIN:-/work/build-906/tune_rocblas}
OUT=${OUT:-/src/tools/hip/gfx906-rocblas-5.5.0.cd957402309.txt}
SHAPES="320,10240,320 10240,320,10240 4,10240,4 48,2560,96 512,2560,512 2560,640,2560 1,2560,1 \
10240,2560,10240 2560,2560,2560 128,2560,128"
TOKENS=${TOKENS:-256,512,768,1024,1536,2048,4096,8192}
cases=""
for t in ${TOKENS//,/ }; do
  for s in $SHAPES; do cases="$cases --case $t,$s"; done
done
# shellcheck disable=SC2086
exec "$BIN" $cases --min-gain 1.02 --tuning-out "$OUT"
