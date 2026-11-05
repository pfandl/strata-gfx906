#!/usr/bin/env bash
# Component-level check of STRATA_GR_SPLIT=1 (the gfx906 K-split hyper-connection down read): the
# two binaries the build carries for exactly this kernel, each run with the flag off and on.
#
# Run inside the runtime image with the build at /work (decode-ab.sh does it through EXTRA_INSIDE,
# once the arms are done and no engine holds the GPUs):
#
#   OUT=/data/grsplit bash /work-bench/gr-component.sh
#
#   fused_gr_bench [iters] [T min] [T max] [cold sets]  the multi-token read, "old" vs "fast", per T,
#                                                       with a hash of each side and the per-stage
#                                                       stamps (norm / down / up) of a cold read
#   gr_parity                                           the contract test: the window read against T
#                                                       single-token reads.  The split's fixed-order
#                                                       finish sums its K slices in another order, so
#                                                       a difference here is the documented cost of
#                                                       the flag, not a bug (fused_gr.cu:2303-2307)
set -u
OUT=${OUT:-/data/grsplit}
BIN=${BIN:-/work/build-906}
ITERS=${ITERS:-50}
D=$OUT/component
rm -rf "$D"; mkdir -p "$D"

for v in 0 1; do
  name=off; [ "$v" = 1 ] && name=on
  for i in 1 2 3; do
    STRATA_GR_SPLIT=$v "$BIN/fused_gr_bench" "$ITERS" 1 5 1 > "$D/bench-$name-$i.log" 2>&1
    echo "bench-$name-$i rc=$?"
  done
  STRATA_GR_SPLIT=$v "$BIN/gr_parity" > "$D/parity-$name.log" 2>&1
  echo "parity-$name rc=$?"
done

echo "--- fused_gr_bench, T 1..5, old vs fast (us), flag off ---"
grep -ah '^T \\|^COLD' "$D"/bench-off-*.log | tail -8
echo "--- fused_gr_bench, flag on ---"
grep -ah '^T \\|^COLD' "$D"/bench-on-*.log | tail -8
echo "--- parity tail, off ---"
tail -4 "$D/parity-off.log"
echo "--- parity tail, on ---"
tail -4 "$D/parity-on.log"
