# Re-applying this machine's tuning to a new Morrowmake release

One record of what was tuned on this machine on 2026-10-31 (recipe 1.8.0, TP4), so it can be re-applied to later
GLM releases and, where it fits, to the DeepSeek recipe.

Sources: `~/tuning-log.md`, `~/engine-log.md`, `~/engine-work/`, the current `.env`, `start.sh`, `git log` of this
repo (fork commits `6618585` and `59003b2` on top of upstream `9d3b0be`). Hardware facts marked *(machine)* were read
read-only from the running system on 2026-10-31. Facts marked *(brief)* come from the task brief and were not
checked. "Not recorded" means none of these sources has it.

---

## 1. Machine

| Item | Value |
|---|---|
| CPU | AMD EPYC 7302 16-Core, 1 socket, 1 NUMA node *(machine)* |
| Board | HUANANZHI H12D-8D *(machine)* |
| BIOS | version 3.0, dated 09/16/2026 *(machine)* |
| RAM | 6 x 16 GB DDR4-3200 *(brief)*; OS sees 91 GiB *(machine)*. Channel population: not recorded |
| GPUs | 4x NVIDIA CMP 170HX, 65,536 MiB each *(machine)* |
| PCIe | every card Gen2 x16, current = max *(machine)*: bus 01:00.0, 41:00.0, 81:00.0, C4:00.0, each on a different root complex *(brief)* |
| Driver / CUDA | driver 615.71.09, CUDA UMD 13.4 *(machine)*; image: CUDA 13.0 torch build, nvcc 13.3 |
| OS / kernel | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic *(machine)* |
| Recipe | Morrowmake glm53-flash-cmp170hx-recipe 1.8.0 (upstream `971d715`, fork `59003b2`) |
| Engine | vLLM fork commit `819146023facb31c04f1f8512e52464835a10d40` (vLLM 0.32.0rc1.dev215, torch 2.13.0+cu130, FlashInfer 0.7.0.post1) |
| Image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:18a2f26207568128e1d792ce923e86a74e40def1e43e492db83132cab8cdbe14` |
| Power cap | 140 W per card (default 250 W), set at boot by `cmp-powerlimit.service` (`nvidia-smi -pm 1`, `nvidia-smi -pl 140`) *(machine)* |
| Fans / thermal | controlled by the separate `cmp170-manager` container. Its settings: not recorded |

**Deliberately NOT changed:** power limits, clocks, fans, BIOS, driver, kernel parameters, `cmp-powerlimit.service`,
`cmp170-manager`. All numbers below are at the 140 W cap. Peak temperatures across both sessions: core 73 C, HBM 78 C
(limits 80 C / 88 C; no alerts). The cap stays at 140 W; section 9 has a user-run comparison with 160 W.

---

## 2. Result summary

"Hybrid baseline" = the release with the hybrid P2P settings, before any tuning. "Final" = everything in section 3.
bench.py figures are the second of two runs (median of 5 inside each run).

| Metric | Hybrid baseline | Final (ring image) | Change |
|---|---:|---:|---:|
| Decode structured (tok/s) | 398.3 | 442.3 / 430.4 (two boots) | +8 to +11% |
| Decode code (tok/s) | 298.9 | 333.8 / 340.2 | +12 to +14%, **inside the ±15% code noise band** |
| Decode prose (tok/s) | 205.3 | 233.7 / 222.6 | +8 to +14% |
| Prefill, 32k cold (tok/s) | 3108 / 3111 | 3095 / 3103, 3098 | unchanged |
| 200k cold read / cached TTFT | 2867 tok/s / 0.97 s | 2863-2864 tok/s / 0.89-0.92 s | unchanged |
| Aggregate 1 / 4 / 8 users (tok/s) | 199-202 / 452-454 / 594-628 * | 212-214 / 495-501 / 680-702 | ~+6 / +10 / +12% |
| Decode step (one user, prose) | 17.2-18.4 ms * | 15.8-16.6 ms | about −1.7 ms |
| KV pool (tokens) | 1,203,765 | 1,272,971 | +69,206 (MM_CAP=1 +76,546; ring −7,340) |
| `./start.sh restart` to healthy | 353.3 s | 282-287 s (282.1 s on the ring image; 343.4 s on the first ring start, which JIT-builds the module) | about −19% |

\* Measured with pycache and `MM_CAP=1` already applied (phase-2 start). Same-day hybrid for comparison: decode
383.0 / 318.9 / 198.4, step 18.2-19.0 ms.

Noise on this machine (from the brief, confirmed by repeat runs): structured and prose ±5%, code ±15%.

---

## 3. What is applied now

### .env (current, complete)

```bash
# --- already set before tuning (unchanged) ---
HOST=0.0.0.0                                  # listens on every interface; no API_KEY set (see ~/tuning-log.md)
EXTRA_ARGS=--enable-prompt-tokens-details     # pre-existing
PORT=8000
SERVED_MODEL_NAME=GLM5.3-Flash
MAX_SEQS=8                                    # must match cmp170-manager GLOBAL_CAP
P2P=auto                                      # "hybrid": peer access verified by the content check...
VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0              # ...but no flags-in-data custom all-reduce
NCCL_P2P_LEVEL=SYS                            # NCCL may use peer-to-peer

# --- added in the tuning session ---
MM_CAP=1                                      # caps image/video inputs at 1280 tokens each; KV +76,546. Speed not benchmarked

# --- added in phase 2 (the "four ring lines": this comment + 3) ---
# phase2: FlashInfer PCIe IPC + GNmac-lab true ring (derived image), see ~/engine-log.md
IMAGE=glm53-flash-exp:truering                # the derived image (below)
VLLM_HOST_ALLREDUCE=0                         # HOSTSHM off; required, it refuses to coexist with FI PCIe IPC
VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1      # decode all-reduce on FlashInfer PCIe IPC; startup tune picks TRUE_RING
```

### start.sh (upstream PR #19 by ElectroHobo, open as of 2026-10-31)

In `container_env()` (line 612 in 1.8.0), replace

```bash
    echo "PYTHONDONTWRITEBYTECODE=1"
```

with

```bash
    echo "PYTHONPYCACHEPREFIX=/cache/pycache"
```

Effect: Python bytecode is kept under `cache/pycache` (173 MB). The first start fills it; later restarts are 282-287 s
instead of 353 s. Benches unchanged. Fork commit `6618585`.

### The derived image

`glm53-flash-exp:truering` (local id `sha256:e16d9fcdafd0…`) = the pinned image + five patched FlashInfer files.
Nothing else changes.

```bash
cd ~/Documents/glm53-flash-cmp170hx-recipe
./truering/build.sh                 # FROM the RELEASE_IMAGE pinned in start.sh -> glm53-flash-exp:truering
```

What `build.sh` does (fork commit `59003b2`; original build: `~/engine-work/img-truering/`):

1. `docker create` the pinned image and copy out five files from
   `/opt/venv/lib/python3.12/site-packages/flashinfer/`: `comm/pcie_ipc_policy.py`, `comm/pcie_ipc_tuning.py`,
   `data/csrc/pcie_ipc_all_reduce.cu`, `data/include/flashinfer/comm/pcie_ipc_all_reduce.cuh`, `jit/comm.py`.
2. `patch -p1` them with `truering/flashinfer-true-ring.patch` = GNmac-lab `patches/patch_true_ring.py` (unchanged:
   adds `IpcVariant.TRUE_RING = 4`, the kernel, and `PCIE_IPC_TUNE_VERSION = 2`), plus one line of our own: the JIT
   module is renamed `pcie_ipc_comm` -> `pcie_ipc_comm_truering`.
3. `docker build` a two-line Dockerfile: `FROM <pinned digest>` + `COPY fi/ …/flashinfer/`.

`install_fipcie.py` from GNmac-lab is **not** needed: FlashInfer 0.7.0.post1 in the image already has PCIe IPC.

---

## 4. Re-apply on a new recipe release

Work on a copy of `.env` first: `cp .env .env.before-<version>`.

### a. Record the new release's own baseline first

1. Remove the four ring lines from `.env` (comment, `IMAGE=`, `VLLM_HOST_ALLREDUCE=0`,
   `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`). Keep the P2P hybrid lines and `MM_CAP=1`.
2. `./start.sh update` (or `git pull origin main` from Morrowmake, then `./start.sh restart`).
3. Check: `docker logs glm53-flash 2>&1 | grep -F '[p2p]'` shows `custom_all_reduce=0`, and the tp:0
   `all-reduce backends` line shows `['HOSTSHM', 'PYNCCL']`.
4. Warm up: one throwaway request; `docker logs --since 5m glm53-flash 2>&1 | grep -c "JIT compilation"` must reach 0
   (boot-check compiles age out after 5 minutes).
5. Run `~/bench.py` three times and `~/bench_long.py --model GLM5.3-Flash` twice. Note the KV pool from
   `./start.sh status` and the restart time.
6. Record the greedy reference: `python3 truering/tools/greedy_check.py run base-<version>.json`, twice. They must be
   20/20 identical, or the reference is not usable.

### b. PR #19

```bash
grep -n 'PYTHONPYCACHEPREFIX\|PYTHONDONTWRITEBYTECODE' start.sh
gh pr view 19 -R Morrowmake/glm53-flash-cmp170hx-recipe --json state,mergedAt
```

If `start.sh` still says `PYTHONDONTWRITEBYTECODE=1`, re-apply the one-line change from section 3. Time two restarts
(the first fills the cache) against one before the change.

### c. FlashInfer and the ring patch

1. Check the FlashInfer version in the new image:
   `docker run --rm --entrypoint python <new image> -c "import flashinfer; print(flashinfer.__version__)"`
   (needs no GPU). The patch was made for 0.7.0.post1.
2. Test the patch against the new image without building anything:

   ```bash
   W=$(mktemp -d); cid=$(docker create <new pinned digest>)
   FI=/opt/venv/lib/python3.12/site-packages/flashinfer
   for f in comm/pcie_ipc_policy.py comm/pcie_ipc_tuning.py data/csrc/pcie_ipc_all_reduce.cu \
            data/include/flashinfer/comm/pcie_ipc_all_reduce.cuh jit/comm.py; do
     mkdir -p $W/$(dirname $f); docker cp -q $cid:$FI/$f $W/$f; done; docker rm -f $cid
   patch -d $W -p1 --dry-run --forward --batch < truering/flashinfer-true-ring.patch
   ```

   Every file must say `checking file …` with no `FAILED` or `Reversed`. If a hunk fails, compare the new files with
   GNmac-lab's current `patch_true_ring.py` before porting by hand.
3. Check the vLLM side still exists:
   `/opt/vllm-src/vllm/distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py`, and
   `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` in `/opt/vllm-src/vllm/envs.py`.
4. Rebuild: `./truering/build.sh`. It reads `RELEASE_IMAGE` from the new `start.sh`, so the image is built FROM the
   new pinned digest.
5. Keep the compiled module separate from stock. `start.sh` forces `FLASHINFER_WORKSPACE_BASE=/cache`, so a Dockerfile
   `ENV` cannot move the cache. The module rename in the patch (`pcie_ipc_comm_truering`) is what keeps the builds
   apart. Check it survived: `grep pcie_ipc_comm_truering` in the image's `flashinfer/jit/comm.py`.

### d. Renamed or retired switches

1. Retired names are listed in the image at `/opt/vllm-src/vllm/sm80/envs.py` (in 1.8.0 the path is
   `vllm/sm80/envs.py`, not `vllm_sm80/envs.py`), tested by `tests/test_retired_env_vars.py`, and in
   `docs/engine-switches.md` under "Renamed and retired settings".
2. A retired name only logs a warning. After the start:

   ```bash
   docker logs glm53-flash 2>&1 | grep -i "is retired and ignored"
   ```

   This must print nothing for our switches: `VLLM_HOST_ALLREDUCE`, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`,
   `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P`, `NCCL_P2P_LEVEL`, `MM_CAP`.
3. Also confirm the backend actually changed: the tp:0 line must show `['FLASHINFER_PCIE_IPC', 'PYNCCL']` and the log
   must contain `Initialized FlashInfer PCIe IPC all-reduce for TP4`. Check that the startup tune picked the ring:
   in `cache/.cache/vllm/flashinfer_autotune_cache/…/pcie_ipc_allreduce_tp_0-1-2-3.json` the entries for 2+ rows
   should have variant `4` as their first number.

### e. Verify, then decide

Put the four ring lines back, `./start.sh restart`, warm up as in (a).

1. Boot check: the start must print `[boot-check] PASS`.
2. `./start.sh smoke` must print `smoke: ok`.
3. Greedy:

   ```bash
   python3 truering/tools/greedy_check.py run ring-<version>.json
   python3 truering/tools/greedy_check.py run ring-<version>-2.json
   python3 truering/tools/greedy_check.py compare ring-<version>.json ring-<version>-2.json   # must be 20/20 identical
   python3 truering/tools/greedy_check.py compare base-<version>.json ring-<version>.json      # expect divergences
   ```

   Accept only if every divergence is at a small baseline margin. In 1.8.0: all at ≤ 1.0 nats (median 0.25), where only
   9.7% of tokens have a margin that small. For scale, the stock hybrid with one concurrent user reached 0.875 nats.
   Read the text at any divergence above ~1 nat.
4. `~/bench.py` three times (plus `~/bench_long.py` twice).
5. **Keep the ring only if structured and prose beat the new stock baseline from (a) by more than 5% on two separate
   boots.** Code alone cannot decide it (±15%).

### f. Revert

1. Delete the four ring lines from `.env`.
2. Run `./start.sh restart`. It returns to the pinned image and HOSTSHM.
3. Check that the tp:0 line shows `['HOSTSHM', 'PYNCCL']` again.

To revert PR #19, restore `echo "PYTHONDONTWRITEBYTECODE=1"`. To revert MM_CAP, delete `MM_CAP=1`. The derived
image can stay; it is unused without `IMAGE=`. Ask before deleting images or caches.

---

## 5. Tried and did NOT help

| What | Result | Source |
|---|---|---|
| `MAX_BATCHED=4615` (4,608-token chunks) | prefill 3095 / 3101 vs 3108 / 3111 (flat); KV −22,037 | tuning-log |
| `MAX_BATCHED=5767` (5,760-token chunks) | prefill 3092 / 3083 (−0.7%), 200k cold −0.8%; KV −40,631. 6919 not tried | tuning-log |
| Prefill all-reduce overlap/split | does not exist in 1.8.0 (`VLLM_GLM5_PREFILL_OVERLAP*` retired) | tuning-log |
| Full P2P (flags-in-data custom all-reduce, `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1`) vs hybrid | 5-15% slower end to end *(brief; raw runs not recorded)*. Microbench [8, 4096]: 59.8 µs vs HOSTSHM 42.1 µs | engine-log |
| Stock FlashInfer PCIe IPC (no ring) | decode 372.0 / 253.2 / 191.9 vs 398.3 / 298.9 / 205.3; step 18.9-20.0 ms; KV −7,340 | engine-log |
| Custom all-reduce with flags off (`VLLM_CUSTOM_ALLREDUCE_FLAGS=0`) | microbench only: 281.4 µs at [8, 4096] (6.7x HOSTSHM). Not run end to end | engine-log |
| NCCL only | microbench only: 64.6 µs at [8, 4096] | engine-log |
| 1stage all-reduce variant (`VLLM_CUSTOM_ALLREDUCE_ALGO`) | not recorded | — |
| OS tests: busy core, no idle states, performance governor, ASPM off | not recorded | — |
| RAM at 2933 vs 3200 | not recorded | — |
| Task 3 A: fuse more small bf16 projections into grouped launches | estimate 0.3-0.8 ms/step (2-5%). Skipped: `_thin_gemm_kernel` is already per-shape tuned (beats cuBLAS by 5-30% per Morrowmake's table) and its largest shape runs at ~1.37 TB/s in situ, ~85% of the measured 1.6 TB/s. User declined | engine-log |
| Task 3 B: int8 weights for the bf16 projections | estimate 1.5-2.5 ms/step (10-15%). Skipped: changes model weights, needs a quality evaluation and new kernels, multi-day. User declined | engine-log |

Not "didn't help" but untested for speed: `MM_CAP=1` (kept; tests skipped by user). `MAX_SEQS` 6 vs 8 and reasoning
effort: skipped by user, nothing measured.

---

## 6. Known caveats

- **Not bitwise equal to the hybrid.** The ring sums in bf16 in ring order; HOSTSHM sums in fp32 in fixed rank order
  (microbench: ~67% of elements equal the correctly rounded sum, vs 100%). Greedy: 1/20 prompts identical to the
  hybrid. The ring is deterministic itself: 20/20 run to run and across reboots.
- **No timeout in the ring's wait loops.** HOSTSHM bounds its spins; the ring (and stock FI PCIe IPC) spin on data. A
  crashed rank would hang the others instead of raising. Recovery: `./start.sh restart`; if the GPUs stay busy or the
  restart hangs, reboot the machine. Never `docker kill`. This path was not exercised.
- **No KL-divergence number.** `docs/results.md` cites `kld_probe`, but it is not included in the repo or the image.
- Measured on this one board and topology, TP4 only. PP4 and long-run stability: not tested.
- The KV pool drops by 7,340 tokens with the ring (FlashInfer workspace).

---

## 7. Does it apply to DeepSeek? (only the ring was tested; see 7a)

Target: `~/Documents/dsv41-flash-cmp170hx-recipe` (at `7f9023c`, image
`ghcr.io/morrowmake/vllm-cmp170hx@sha256:f67d9be47ffbd4a8b84c77fe76bdb3f842187d9b2786fff573e390ae5361bfb0`). Apart
from the true ring (tested, slower, section 7a), the rows below are read-only file inspections from 2026-10-31.

| Item | Should transfer? | What to check first |
|---|---|---|
| PR #19 (pycache) | Likely. Its `start.sh` line 505 still has `echo "PYTHONDONTWRITEBYTECODE=1"` | Same one-line change; time two restarts |
| Hybrid P2P (`P2P=auto`, `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0`, `NCCL_P2P_LEVEL=SYS`) | Probably. The DS recipe has the same switches; `VLLM_HOST_ALLREDUCE` defaults to 1 | Measure hybrid vs full P2P on DS; do not assume GLM's result |
| `MM_CAP=1` | No. The DS `.env.advanced.example` has no `MM_CAP` | — |
| `MAX_BATCHED` findings | No. Model-specific (DS default `2048`) | Re-measure if wanted |
| True ring | **No: tested, 5-7% slower decode on DS** (section 7a). It builds and runs correctly there | Do not apply to DS without first finding out why (profile one DS decode step) |
| Task 3 ideas | Not applicable as written (GLM-specific kernels) | — |

### 7a. DeepSeek true-ring test (2026-10-31): reverted

Same method as GLM phase 2. Baseline = the DS `.env` as it was (HOSTSHM). Ring = the same four lines with
`IMAGE=dsv41-flash-exp:truering`. `~/bench_ds.py` (thinking off) three times each.

| Metric | Baseline (HOSTSHM) | Ring | Change |
|---|---:|---:|---:|
| Decode structured (tok/s), 3 runs | 247.2 / 248.4 / 245.6 | 231.6 / 233.3 / 232.2 | **−6%** |
| Decode code | 232.4 / 234.5 / 232.5 | 220.6 / 221.4 / 221.7 | −5% |
| Decode prose | 129.8 / 130.9 / 128.0 | 113.5 / 123.6 / 118.9 | **−7%** |
| Prefill 32k (tok/s) | 3,676-3,718 | 3,810-3,861 | +3.5% |
| Decode step | 23.4-23.9 ms | 25.0-25.4 ms | +1.5 ms |
| Aggregate 1 / 4 / 8 users (tok/s) | 121 / 257 / 337 | 113 / 254 / 342 | −7% / −1% / +1% |
| KV pool (tokens) | 1,240,955 | 1,220,630 | −20,325 |
| Start to healthy | 544 s | 642 s (first start, includes the JIT build) | |

DS run-to-run spread was small (structured and code ~1%, prose ~±4%), so the loss is outside the noise.

What worked:
- Step 1 passed. Both images have FlashInfer 0.7.0.post1, the five PCIe IPC files and vLLM's wrapper are
  byte-identical, and the patch applies.
- The image was built with `BASE=<DS digest> TAG=dsv41-flash-exp:truering ./truering/build.sh` (`sha256:350a8834497b…`).
  It uses the same module rename, and its JIT cache sits in the DS checkout's `cache/`.
- Boot check PASS; smoke ok; tp:0 `['FLASHINFER_PCIE_IPC', 'PYNCCL']`; startup tune picked TRUE_RING for every
  capture size 2-48; no retired-name warnings.
- Greedy, thinking off (`GC_MODEL=DS4.1-Flash GC_EXTRA='{"chat_template_kwargs": {"thinking": false}}'`): the ring
  is deterministic (20/20). Against the baseline, 3/20 prompts are identical. All 17 divergences are at a baseline
  margin of 0.014-0.913 nats (median 0.197); 9.0% of baseline tokens have a margin ≤ 1.0.

Decision: structured and prose are slower, so it was reverted. The DS `.env` is identical to `.env.before-ring` again.
The image `dsv41-flash-exp:truering` is kept but unused.

**Why it is slower (confirmed 2026-10-31, all-reduce call counter, not a profiler).** DS sends the attention
`wo_b` output all-reduce in **fp32** `[M, 5120]`: `VLLM_DEEPSEEK_V41_WO_B_BF16_ALLREDUCE` (on by default) switches it
to bf16 only when vLLM's custom all-reduce takes it, and that path is disabled on this board. HOSTSHM accepts fp32;
FlashInfer PCIe IPC accepts bf16 only. With HOSTSHM off, half of the decode all-reduces (twice the bytes) go to NCCL,
plus a small bf16 `[M, 320]` call. The bf16 MoE output does go to the ring.

Possible fixes, **parked until the DS recipe matures upstream** (user decision):

| Option | Change | Estimated decode gain |
|---|---|---:|
| 1 | Keep HOSTSHM for fp32 and send bf16 `[M, 5120]` to the ring (relax the construction guard, add a dispatch rule) | ~+5% |
| 2 | Send `wo_b` in bf16 whenever FlashInfer takes it, so everything uses the ring with HOSTSHM off | ~+12-13% at best |

Both estimates come from a microbench at hidden 5120 (`AR_HID=5120 AR_DTYPE=… truering/tools/ar_bench.sh`).
Option 2 trades the exact fp32 sum for a bf16 ring sum. On a new DS release, first check whether `wo_b_all_reduce` (in
`sm80_attention.py`) still sends fp32 when the custom all-reduce is off. Details and the counter patch are in
`~/engine-log.md` and `~/engine-work/ar_count_hook.patch`.

**Do not run the full torch profiler on DS.** With the default Engram row cache (7 GiB per card), DS leaves only ~9 GiB
of RAM available, and a 4-rank profiler run froze the machine. For tests, use `VLLM_ENGRAM_ROW_CACHE_GIB=3`
(~52 GiB stays available).

DS-specific notes for later:
- `./start.sh status` in the DS recipe reports "models: missing or incomplete". This is a false alarm: the hybrid's
  files are symlinks to container paths, so the host-side check fails. The start itself logs "hybrid: built,
  skipping" and downloads nothing.
- `greedy_check.py` in `~/engine-work/` now takes `GC_MODEL` and `GC_EXTRA`. The fork copy in `truering/tools/` does
  not yet.

---

## 8. Credits and licences

| Part | Who | Licence | Where checked |
|---|---|---|---|
| `patch_true_ring.py` (cut-through ring) | GNmac-lab | MIT | `~/Documents/gnmac-ds41/LICENSE` |
| Recipe (scripts, docs) | Morrowmake | **MIT** (the brief said Apache-2.0; the repo's `LICENSE` is MIT) | `LICENSE` in this repo |
| Engine (vLLM fork in the image) | Morrowmake / vLLM contributors | Apache-2.0 | `/opt/vllm-src/LICENSE` in the image |
| FlashInfer (the five patched files) | FlashInfer contributors | Apache-2.0 | `flashinfer_python` METADATA in the image |
| PR #19 "Cache Python bytecode under cache/ instead of writing none (start ~77 s faster)" | ElectroHobo | — (upstream PR, open) | `gh pr view 19` |
| Microbench `ring_ar.cu` (adapted to hidden 4096) | GNmac-lab | MIT | `~/ring_ar.cu`, `~/engine-work/ring_ar_4096.patch` |

Detailed logs: `~/tuning-log.md` (phase 1), `~/engine-log.md` (phase 2), `docs/true-ring.md` (fork doc).

---

## 9. Power cap: 140 W vs 160 W (user-run, 2026-10-31)

GLM on the ring config (section 3), TP4, `MAX_SEQS=8`. The user changed the limit at runtime with
`sudo nvidia-smi -pl <W>`. `cmp-powerlimit.service` was not touched, so a reboot restores 140 W.

**Concurrency** (`conc.py`: 400-token decodes, three rounds per user count; same session, 160 W then 140 W),
total tok/s:

| Users | 160 W (runs) | 140 W (runs) | 160 W vs 140 W |
|---:|---|---|---:|
| 1 | 226 (216, 226, 229) | 212 (212, 196, 230) | +6.6% |
| 2 | 356 (356, 365, 346) | 340 (340, 328, 355) | +4.7% |
| 4 | 515 (525, 515, 504) | 511 (518, 511, 493) | +0.8% |
| 8 | 726 (726, 710, 758) | 725 (725, 748, 707) | +0.1% |

**Single user** (`bench.py`, second of two runs). 160 W was measured on 2026-10-31. The 140 W numbers are the two ring
boots from phase 2, an earlier session:

| Metric | 140 W | 160 W | Change |
|---|---|---:|---:|
| Decode structured (tok/s) | 442.3 / 430.4 | 471.7 | +7 to +10% |
| Decode code | 333.8 / 340.2 | 346.4 | within the ±15% noise |
| Decode prose | 233.7 / 222.6 | 247.5 | +6 to +11% |
| 32k cold prefill (tok/s) | 3,095-3,103 | 3,162 | +2% |

**Reading:**
- 160 W makes 1-2-user decode roughly 5-10% faster and prefill about 2% faster.
- At 4-8 users it changes nothing, because batched decode is not limited by the power cap on this machine.
- Not adopted: the machine stays at 140 W.

**Caveats:**
- The single-user comparison spans two sessions.
- No power-draw or temperature log was running during these runs, so it is not known how often the cards reached the cap, or how much hotter 160 W runs.
- The 140 W single-user `conc.py` rounds spread from 196 to 230 tok/s.

**To confirm it:**
- Run `~/engine-work/templog.sh` (logs power draw and temperatures).
- Alternate 140 → 160 → 140 → 160 W in one session, with `bench.py` ×2 and `conc.py 1,2,8` at each step.
- Keep the 80 C core / 88 C HBM stop rule.
