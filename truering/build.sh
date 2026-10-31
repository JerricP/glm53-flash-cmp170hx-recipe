#!/bin/bash
# Build the true-ring engine image (docs/true-ring.md) from the image start.sh pins.
#
#   ./truering/build.sh                    # -> glm53-flash-exp:truering
#   TAG=myname:tag ./truering/build.sh
#
# Then add to .env and ./start.sh restart:
#   IMAGE=glm53-flash-exp:truering
#   VLLM_HOST_ALLREDUCE=0
#   VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-$(sed -n 's/^RELEASE_IMAGE=\(.*\)$/\1/p' "$HERE/../start.sh")}
TAG=${TAG:-glm53-flash-exp:truering}
FI=/opt/venv/lib/python3.12/site-packages/flashinfer
FILES="comm/pcie_ipc_policy.py comm/pcie_ipc_tuning.py data/csrc/pcie_ipc_all_reduce.cu
data/include/flashinfer/comm/pcie_ipc_all_reduce.cuh jit/comm.py"

[ -n "$BASE" ] || { echo "build.sh: no RELEASE_IMAGE in start.sh; set BASE=" >&2; exit 1; }
docker image inspect "$BASE" >/dev/null 2>&1 || docker pull "$BASE"

work=$(mktemp -d)
trap 'rm -rf "$work"; [ -n "${cid:-}" ] && docker rm -f "$cid" >/dev/null' EXIT
cid=$(docker create "$BASE")
for f in $FILES; do
    mkdir -p "$work/fi/$(dirname "$f")"
    docker cp -q "$cid:$FI/$f" "$work/fi/$f"
done
# The patch is made against FlashInfer 0.7.0.post1 as shipped in the recipe 1.8.0 image; refuse anything else.
patch -d "$work/fi" -p1 --forward --batch < "$HERE/flashinfer-true-ring.patch"
cp "$HERE/Dockerfile" "$work/"
docker build -t "$TAG" --build-arg BASE="$BASE" "$work"
echo "built $TAG from $BASE"
