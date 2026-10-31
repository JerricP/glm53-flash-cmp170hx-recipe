# The recipe's pinned engine image plus GNmac-lab's cut-through ring all-reduce, registered as the FlashInfer PCIe
# IPC variant TRUE_RING. Build it with ./build.sh, which extracts the five original FlashInfer files from BASE,
# applies flashinfer-true-ring.patch to them and puts them in fi/; nothing else in the image changes.
ARG BASE=ghcr.io/morrowmake/vllm-cmp170hx@sha256:18a2f26207568128e1d792ce923e86a74e40def1e43e492db83132cab8cdbe14
FROM ${BASE}
COPY fi/ /opt/venv/lib/python3.12/site-packages/flashinfer/
