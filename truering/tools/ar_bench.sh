#!/bin/bash
# All-reduce microbench for [M, 4096] bf16 through the engine's own TP-group dispatch (ar_bench.py), one backend per
# run, in a throwaway container. Stop the server first (./start.sh stop).
#
#   truering/tools/ar_bench.sh hybrid              # HOSTSHM, the recipe's choice with VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0
#   truering/tools/ar_bench.sh custom              # flags-in-data custom all-reduce (full P2P)
#   truering/tools/ar_bench.sh nccl
#   truering/tools/ar_bench.sh fipcie              # stock FlashInfer PCIe IPC (pinned image)
#   truering/tools/ar_bench.sh fi-ring             # FlashInfer PCIe IPC + true ring (glm53-flash-exp:truering)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PINNED=$(sed -n 's/^RELEASE_IMAGE=\(.*\)$/\1/p' "$HERE/../../start.sh")
IMG=$PINNED
case "${1:-}" in
  hybrid) E=(VLLM_HOST_ALLREDUCE=1 VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0) ;;
  custom) E=(VLLM_HOST_ALLREDUCE=1 VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1 VLLM_CUSTOM_ALLREDUCE_FLAGS=1) ;;
  nccl) E=(VLLM_HOST_ALLREDUCE=0 AR_NO_CUSTOM=1) ;;
  fipcie) E=(VLLM_HOST_ALLREDUCE=0 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0) ;;
  fi-ring) E=(VLLM_HOST_ALLREDUCE=0 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0)
           IMG=${RING_IMAGE:-glm53-flash-exp:truering} ;;
  *) echo "usage: $0 hybrid|custom|nccl|fipcie|fi-ring" >&2; exit 1 ;;
esac
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
args=()
for kv in "${E[@]}" NCCL_P2P_LEVEL=SYS AR_LABEL="$1" HOME=/out TMPDIR=/out VLLM_CACHE_ROOT=/out/vllm \
          PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False; do args+=(-e "$kv"); done
docker run --rm --user "$(id -u):$(id -g)" --gpus all --shm-size 16g -v "$HERE:/work:ro" -v "$out:/out" "${args[@]}" \
  --entrypoint torchrun "$IMG" --nproc-per-node 4 /work/ar_bench.py 2>&1 | grep -E "^$1 "
