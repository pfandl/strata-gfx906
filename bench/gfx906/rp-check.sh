#!/bin/sh
# Fast host-side compile check of the gfx906 engine with the RP port, no image build: the build
# dir persists, so a re-run is incremental. The flags are the ones the workspace build and the
# runtime image's engine stage use.
#
#   bench/gfx906/rp-check.sh                          # SRC = this checkout, WORK = ~/.cache/strata-gfx906-rp-check
#   SRC=/path/to/checkout WORK=/tmp/rp-build bench/gfx906/rp-check.sh
#
# IMAGE must be the builder image (ROCm + cmake + ninja + clang): build it from
# docker/Dockerfile.builder and tag it strata-gfx906-builder, or set IMAGE to your own tag.
set -e
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SRC=${SRC:-$(CDPATH= cd -- "$HERE/../.." && pwd)}
WORK=${WORK:-$HOME/.cache/strata-gfx906-rp-check}
IMAGE=${IMAGE:-strata-gfx906-builder}
mkdir -p "$WORK"
docker run --rm --memory=60g --memory-swap=60g \
  -v "$SRC":/src:ro \
  -v "$WORK":/work \
  "$IMAGE" bash -c '
    set -e
    mkdir -p /work/build-906
    cd /src
    cmake -S . -B /work/build-906 -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_HIP_GFX906=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx906 \
      -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DSTRATA_BUILD_TESTS=OFF
    cmake --build /work/build-906 --target strata rp_expert_bench dense_mmq_bench -j 32
    echo BUILD-OK
  '
