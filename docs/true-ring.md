# True-ring all-reduce (fork addition)

At TP4 every decode step runs about 107 small all-reduces (one per attention and MoE block of the 45-layer target
and the 5-layer DFlash2 drafter), each `[M, 4096]` bf16 with M up to 8 rows (64 KiB). With peer-to-peer verified
but the custom all-reduce off (`VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0`), the release sends them through its host-staged
`HOSTSHM` all-reduce. This fork adds an image in which they run on FlashInfer's PCIe IPC all-reduce with
[GNmac-lab](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx)'s cut-through neighbour ring
(`patches/patch_true_ring.py` there, MIT): every rank writes only to rank + 1, there are no barriers, and each
16-byte pack carries its own arrival signal, so it moves to the next hop as soon as it lands.

## Use it

Needs: TP4, and the recipe's peer-to-peer content check passing (`P2P=auto`, the default).

```bash
./truering/build.sh            # builds glm53-flash-exp:truering from the image start.sh pins (a few seconds)
```

Add to `.env` and restart:

```bash
IMAGE=glm53-flash-exp:truering
VLLM_HOST_ALLREDUCE=0
VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
```

```bash
./start.sh restart
docker logs glm53-flash 2>&1 | grep "all-reduce backends" | head -1   # tp:0 should list ['FLASHINFER_PCIE_IPC', 'PYNCCL']
```

The first start JIT-compiles the module (it is named `pcie_ipc_comm_truering`, so it never shares a cache directory
with the stock one) and FlashInfer's startup tune measures every variant per CUDA-graph size and keeps the fastest; it
picks the ring (`TRUE_RING`) for every size from 2 rows up. Prefill-sized messages (above 64 rows) go to NCCL, as
they do in the release. **To revert**, delete the three lines and `./start.sh restart`.

What the image changes: five FlashInfer 0.7.0.post1 files, patched by
[`truering/flashinfer-true-ring.patch`](../truering/flashinfer-true-ring.patch) = GNmac-lab's `patch_true_ring.py`,
unchanged, plus the JIT module rename. `build.sh` extracts the originals from the pinned image, so the patch fails
loudly if the pin moves to a FlashInfer it was not made for. Nothing else in the image changes.

## Measurements

One machine: EPYC 7302 (Rome), 96 GB DDR4-3200, 4x CMP 170HX, PCIe Gen2 x16, each GPU on a different root complex.
Recipe 1.8.0, TP4, DFlash2 (release adaptive depth), `MAX_SEQS=8`, `MM_CAP=1`, `P2P=auto`,
`VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0`, `NCCL_P2P_LEVEL=SYS`. On this board the flags-in-data custom all-reduce (full
P2P) measured 5-15 % slower end to end than `HOSTSHM`.

**Where the decode step goes** (torch profiler, one user, 400-token decode, rank 0): verify step 18.4 ms; all-reduce
26 % (`HOSTSHM`, 46-52 us per call in place), bf16 dense projections 27 %, MoE 21 %, attention 12 %, norms/mHC 9 %,
idle 1.4 %. The drafter is 15 % of the step.

**All-reduce in isolation**, `[M, 4096]` bf16, through the engine's own TP-group dispatch
([`truering/tools/ar_bench.sh`](../truering/tools/ar_bench.sh), server stopped; 200 calls in one CUDA graph,
slowest rank, best of 7), us per call:

| M (bytes) | HOSTSHM | custom (flags) | NCCL | FI PCIe IPC stock | FI PCIe IPC + ring |
|---|---:|---:|---:|---:|---:|
| 1 (8 KiB) | 17.6 | 12.9 | 20.4 | 12.4 | 12.6 |
| 4 (32 KiB) | 31.2 | 28.4 | 29.7 | 34.8 | 15.7 |
| 6 (48 KiB) | 40.4 | 42.7 | 37.3 | 45.4 | 16.9 |
| 8 (64 KiB) | 42.1 | 59.8 | 64.6 | 50.9 | **19.6** |
| 16 (128 KiB) | 63.8 | 131.2 | 76.3 | 70.9 | 34.6 |
| 32 (256 KiB) | 121.1 | 254.6 | 110.5 | 108.2 | 66.5 |

**End to end**, two boots of the ring image against the release configuration (second of two bench runs each):

| | release (HOSTSHM) | ring, boot 1 | ring, boot 2 |
|---|---:|---:|---:|
| decode, structured / code / prose (tok/s) | 398 / 299 / 205 (same day: 383 / 319 / 198) | 442 / 334 / 234 | 430 / 340 / 223 |
| decode step | 17.2-19.0 ms | 15.8-16.6 ms | 16.1-16.6 ms |
| aggregate, 1 / 4 / 8 users (tok/s) | 199-202 / 452-454 / 594-628 | 212-214 / 495-501 / 680-702 | |
| 32k cold prefill (tok/s) | 3,082-3,111 | 3,095-3,103 | 3,098 |
| KV pool (tokens) | 1,280,311 | 1,272,971 | 1,272,971 |

Single-user decode noise on this machine is about ±5 % (structured, prose) and ±15 % (code). Cold reads at 50k
and 200k tokens and cached first-token times did not change. Stock FlashInfer PCIe IPC without the ring was slower
than the release (372 / 253 / 192 tok/s).

## Correctness

- The recipe's boot check and `./start.sh smoke` pass.
- Greedy, temperature 0, 20 fixed prompts, cold prefix cache ([`truering/tools/greedy_check.py`](../truering/tools/greedy_check.py)):
  the ring is deterministic, 20/20 identical run to run and across reboots.
- It is **not bitwise identical** to the release: 1/20 prompts match. The ring sums in bf16 in ring order, while
  `HOSTSHM` sums in fp32 in a fixed rank order (in the microbench ~67 % of elements are the correctly rounded sum,
  against 100 %). Every divergence is at a token where the release's own top-1/top-2 logprob margin is at most
  1.0 nats (median 0.25); only 9.7 % of tokens have a margin that small (median margin 8.5).
- For scale: the release itself, with one other request in the batch, gives 3/20 identical prompts and divergence
  margins up to 0.875 nats, often on the same tokens.
- No KL-divergence figures: the recipe's `kld_probe` (cited in docs/results.md) is not included in the repo or image.

## Limits

- Measured on one board and topology only; PP4 and long-run stability are not tested.
- The ring's spin loops have no timeout (the release's `HOSTSHM` bounds its spins), so a dead peer process would hang
  the step instead of raising an error. Stock FlashInfer PCIe IPC behaves the same way.
- World size 4 and 16-bit payloads only.
