<h1 align="center">GLM-5.3-Flash on 4× NVIDIA CMP 170HX</h1>

<p align="center">
  <strong>by <a href="https://x.com/Morrowmake">Morrowmake</a></strong>
  <br><br>
  <a href="https://x.com/Morrowmake"><img alt="Follow on X" src="https://img.shields.io/badge/Follow-%40Morrowmake-000000?style=flat&logo=x&logoColor=white"></a>
  &nbsp;
  <a href="https://github.com/Morrowmake/vllm-cmp170hx/tree/819146023facb31c04f1f8512e52464835a10d40"><img alt="engine" src="https://img.shields.io/badge/engine-vLLM%20fork%20%40%20819146023facb31c04f1f8512e52464835a10d40-4b32c3?style=flat"></a>
  &nbsp;
  <img alt="release" src="https://img.shields.io/badge/release-1.8.0-2ea44f?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/recipe-MIT-blue?style=flat">
</p>

> **This is a fork** of [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe)
> 1.8.0 by [JerricP](https://github.com/JerricP). Everything below is Morrowmake's recipe, unchanged, plus two
> optional changes:
>
> - **True-ring all-reduce** ([docs/true-ring.md](docs/true-ring.md)): the TP4 decode all-reduce on FlashInfer's PCIe
>   IPC all-reduce with [GNmac-lab](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx)'s cut-through ring,
>   in an image built from the pinned one by [`truering/build.sh`](truering/build.sh). On one EPYC Rome board with one
>   GPU per root complex: decode all-reduce 42 → 20 us, single-user decode +8-14 %, 4/8 users +10/+12 %, prefill
>   unchanged. Output stays deterministic but is not bitwise identical to the release (bf16 ring sum). Off unless you
>   build the image and set three lines in `.env`.
> - **Faster restarts**: Python bytecode is kept under `cache/pycache` (same change as upstream PR #19).
>
> The figures in the rest of this README are Morrowmake's, not this fork's.

**A 320B-parameter MoE with a 262,144-token context, served on four CMP 170HX
cards in two layouts. Release 1.8.0 tensor-parallel results with verified peer-to-peer at 74 SMs:
482.6 tok/s for one user and 943.7 tok/s across eight,
3,161 tok/s cold prefill and a 1,203,765-token KV pool. The weights
are W4A16 and nothing else is cut: the KV cache is full precision, there is no
FP8 anywhere, and nothing is offloaded to CPU or disk.**

This repository installs and runs **GLM-5.3-Flash** on **four NVIDIA CMP 170HX
cards** behind an OpenAI-compatible API, with tool calls, reasoning, images and
video, and a **DFlash2** speculative drafter. Upstream vLLM's sparse-attention
path needs a Hopper GPU; the CMP 170HX is Ampere (sm_80). Our
[vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere) adds
the Ampere kernels that make the model run at all, then spends the rest of its
patches on making it fast and making it repeatable. It runs only as a container
image, so the engine and its Python environment stay out of your system.
P2P is enabled automatically after a startup [content check](docs/how-to-use.md#pcie-peer-to-peer) passes; otherwise the server runs with P2P disabled.

Also on these cards: [DeepSeek-V4.1-Flash on four CMP 170HX](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe).

## Quick start

One command from a fresh clone, after the [one-time host setup](#one-time-host-setup):

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe
./start.sh          # pull the engine image, fetch the model, start, wait for /health
```

Then try it:

```bash
./start.sh smoke    # one chat request and one tool call
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-flash",
       "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
       "max_tokens": 4096}'
```

> **GLM-5.3 always reasons before it answers.** The reasoning comes back in the
> message's `reasoning` field and the answer in `content`. **Set `max_tokens`
> for the reasoning plus the answer:** if the limit is reached while the model
> is still reasoning, `content` comes back empty with `finish_reason: "length"`.
> `"chat_template_kwargs": {"enable_thinking": false}` has no effect; use
> `reasoning_effort` (`low`, `high`, `max`) to shorten the reasoning
> ([Talk to it](docs/how-to-use.md#talk-to-it)).

`./start.sh` skips anything already done, so running it twice is safe; an
interrupted download resumes. The image ships no warm-cache seed: it starts
compile on first boot (cold), so the first start is slower; later starts reuse
the compiled kernels in `./cache`.
Any OpenAI-compatible client works with base URL `http://127.0.0.1:8000/v1`
and model `glm-5.3-flash`. The API listens on this machine only and has no
API key unless you [change that](docs/how-to-use.md#serving-other-machines).
Python, streaming and tool-call clients are in [examples/](examples/README.md).

**Prefer Docker Compose?** The same container, also one command from a fresh
clone:

```bash
docker compose up              # or in the background: docker compose up -d
```

It downloads the checkpoints, then starts the server as the owner of the
checkout; `docker compose ps` shows it healthy once `/health` answers
([Docker Compose](docs/how-to-use.md#docker-compose)). Compose restarts a
crashed server up to 3 times; `./start.sh` does not (run `./start.sh restart`).
Use one or the other.

Both download the checkpoints at the exact Hugging Face revisions this release
was validated with.

> **Which setups this release is for.** Two layouts, one switch
> (`LAYOUT=tp4` or `LAYOUT=pp4`):
>
> - **Tensor-parallel 4** (the default) is optimised for **PCIe x16 links** —
>   cards with the x16 capacitor modification. It gives each request the
>   fastest answer. On cards limited to x4 links it is bus-bound and much
>   slower.
> - **Pipeline-parallel 4** passes only activations between the cards, so it
>   needs far less link bandwidth; it is the layout **for cards limited to x4
>   links**, and for many parallel users and long prompts. Performance on x4
>   links has not yet been measured.
>
> `./start.sh` checks your link width and suggests `LAYOUT=pp4` if a card is
> narrower than x16.

## One-time host setup

- **4× NVIDIA CMP 170HX, each exposing 64 GiB**; **PCIe Gen2 x16 links** for
  tensor-parallel 4 (pipeline-parallel 4 is built for narrower links)
- Linux with NVIDIA driver **580 or newer**
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  usable by your user: `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists all four cards
- optional: a P2P-capable driver build for verified peer-to-peer
  ([PCIe peer-to-peer](docs/how-to-use.md#pcie-peer-to-peer)); without it the
  server runs with P2P disabled
- for `./start.sh`: `git`, `curl`, `flock`, `setsid`, Python 3 for the boot
  check, and `jq` for the smoke test
- about 180 GiB (193 GB) of disk for the two checkpoints, plus the engine image
  and the kernel compile caches ([details](docs/how-to-use.md#what-you-need))

## Headline numbers

Measured 2026-10-16 on 4× CMP 170HX, 74 SMs per card, mainline cmpunlocker
with P2P, recipe defaults, 262,144-token context, 180 W per card and PCIe x16 links.
Decode uses the MiaAI-Lab protocol: 400 tokens, temperature 0, requests sent with
thinking off (GLM-5.3 reasons regardless; reasoning tokens count); one user is
the median of five runs, and two, four and eight users the median aggregate
rate of three runs, each layout on one boot.

| | Tensor-parallel 4<br>+ [peer-to-peer](docs/how-to-use.md#pcie-peer-to-peer) (default after verification) | Pipeline-parallel 4<br>+ [peer-to-peer](docs/how-to-use.md#pcie-peer-to-peer) |
|---|---:|---:|
| **1 user** · structured | **482.6** | 232.4 |
| **1 user** · code | **442.1** | 209.9 |
| **1 user** · prose | **232.1** | 100.1 |
| **2 users, total** · structured | 606.5 | 343.8 |
| **2 users, total** · code | 460.3 | 273.4 |
| **2 users, total** · prose | 314.7 | 195.5 |
| **4 users, total** · structured | 717.7 | 409.2 |
| **4 users, total** · code | 618.7 | 382.8 |
| **4 users, total** · prose | 501.4 | 314.2 |
| **8 users, total** · structured | **943.7** | 664.1 |
| **8 users, total** · code | **835.0** | 608.8 |
| **8 users, total** · prose | **667.2** | 469.2 |
| **Prompt reading** (cold prefill) | 3,161 | **7,419** |
| **KV pool** (tokens) | 1,203,765 | **2,088,960** |

All rates are up-to figures in tok/s. One user is streaming speed per request; two, four and
eight users are the combined rate of that many simultaneous requests.
Structured output and code are drafted well, so they decode fastest; prose
least. DFlash2 drafts up to 7 tokens ahead for one user and 3 under load.
One-user prose varies most between runs. Both layouts were measured on PCIe x16
links. Quality on the final release build, scored on the answer after the
reasoning: TP4 HumanEval 151/164 and GSM8K 1,286/1,319.
Method and exact figures: [Results](docs/results.md).

## Settings

Settings live in `.env`; see [Settings](docs/how-to-use.md#settings)
for boot checks and default reasoning effort.

## Choosing a layout

Tensor-parallel 4 (`tp4`, the default) gives one or two interactive users the
fastest answer per request and needs x16 links. Pipeline-parallel 4 (`pp4`)
suits many parallel users, long prompts and x4 links, with higher cold prefill
throughput and a larger KV pool ([Results](docs/results.md)). Switch with `LAYOUT=pp4 ./start.sh restart` (or set it
in `.env`); details in [Choosing a layout](docs/results.md#choosing-a-layout).

## Documentation

- [Results](docs/results.md): throughput, quality, decode and prefill in detail
- [How to use this repo](docs/how-to-use.md): install, settings, peer-to-peer,
  kill switches, day to day, updates, Docker Compose, containers
- [Optional compiled Marlin](docs/compiled-marlin.md): defaults, switches, building it
- [How it works](docs/how-it-works.md): what makes it fast and correct, and what runs
- [Engine switches](docs/engine-switches.md): the engine's internal switches, for troubleshooting
- [Status and known limits](docs/known-limits.md)
- [Model licences](docs/model-licences.md): complete target-model MIT notice,
  drafter restrictions and method citations
- [Examples](examples/README.md): curl, Python, streaming, tool calls, layout `.env` files
- [Container image](docker/README.md) and [CHANGELOG.md](CHANGELOG.md)
- Found a problem? [Open an issue](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/issues/new/choose);
  the bug template asks for the version, layout, peer-to-peer setting and log lines we need

## Update

- `./start.sh update` pulls the new release and engine image and restarts; your `.env` is preserved.
  A checkout that ran the engine natively under an earlier release moves to the
  container on the same `.env` and checkpoints ([details](docs/how-to-use.md#day-to-day)).
- With Compose: `git pull`, then `docker compose up -d`.
- Older releases are git tags; their images may be retired.

**Updating to 1.8.0 from 1.7.x:**

- **Size `max_tokens` for the reasoning plus the answer.** GLM-5.3 always
  reasons; this engine returns the reasoning in `reasoning` and the answer in
  `content`. A reply that hits the limit while still reasoning has an empty
  `content`. `enable_thinking: false` has no effect.
- In `.env`, delete any `VLLM_BRANCH=ampere-glm53` line (that branch is
  retired), any `IMAGE` or `VLLM_COMMIT` line (to follow this release's image),
  and an `EXTRA_ARGS` line carrying `max_pixels` and `max_image_tokens` if you
  added it for `MM_CAP=1` on 1.7.2.
- Engine switches have new names: rename any old `VLLM_GLM5_*` (or other
  renamed) line in `.env` or `docker/container.env`; old names are ignored
  with a warning at startup. Delete a `SPEC_N` line: with DFlash2 the server
  refuses it. The [CHANGELOG](CHANGELOG.md) lists every renamed and retired
  setting.
- Native installs need Docker and the NVIDIA Container Toolkit first; see the
  [CHANGELOG](CHANGELOG.md) for the `.env` lines to remove.

---

## Licences

**This recipe** — the scripts and the documentation — is MIT, © 2026 Morrowmake.
See [LICENSE](LICENSE).

**The vLLM fork** is [Apache-2.0](https://github.com/Morrowmake/vllm-cmp170hx/blob/819146023facb31c04f1f8512e52464835a10d40/LICENSE).

The downloaded models have separate licences:

- **Target:** [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP/blob/main/README.md)
  declares `license: mit` (MIT), inherited from the base model
  [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE).
  Its card says: "Follow the base model's usage terms."
- **Default drafter:** [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/main/README.md)
  declares `license: cc-by-nc-nd-4.0` (CC BY-NC-ND 4.0). Its card says
  "for research and evaluation. For commercial licensing, contact contact@inco.ai."
  The [licence](https://creativecommons.org/licenses/by-nc-nd/4.0/) requires
  attribution, prohibits commercial use, and prohibits distributing modified material.

DFlash2 is the only supported speculative mode; every figure on this page uses
it. Its default checkpoint does not grant commercial-use permission: commercial
use requires a separate licence from incoai ([Known limits](docs/known-limits.md)).
The [model-licence reference](docs/model-licences.md) preserves the target's
complete inherited MIT notice and describes the separate checkpoint terms.

**The benchmark prompts** come from MiaAI-Lab's repository (AGPL-3.0) and
their sparkDash prompt constants (MIT), used as data with attribution. No code from their repositories is included here.

## Credits

If this recipe helps you, please credit Morrowmake and link this repository.
Citation metadata: [`CITATION.cff`](CITATION.cff).

- **[MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)**
  for publishing the benchmark prompts used in the decode tables.
- **[incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)** for the
  DFlash2 drafter checkpoint and [DFlash 2](https://inco.ai/blog/dflash2/).
  The original [DFlash](https://github.com/z-lab/dflash) method is by
  Jian Chen, Yesheng Liang and Zhijian Liu (ICML 2026).
- **[canada-quant](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP)**
  for the W4A16 quantisation.
- **[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash)** for GLM-5.3-Flash,
  and the **[vLLM](https://github.com/vllm-project/vllm)** project for the
  engine these patches sit on top of.

## Source

The fork contains our maintained changes, inherited upstream code and the
adaptations credited below —
[Morrowmake/vllm-cmp170hx @ `ampere`](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere).

## Acknowledgements

- Mainline [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) and its maintainer's P2P work, plus our minimal TRAP31 patch, form the basis of the P2P driver build. Ready to install: [Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker).
- Upstream [vLLM](https://github.com/vllm-project/vllm) provides the
  [RecoverSSM implementation](https://github.com/vllm-project/vllm/commit/70afdedc1081d28c3eaae53bece8292298484c86)
  and fixes we build on.
- [MiaAI-Lab's GLM-5.3-Flash recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/6278ecb01034cea9ef6de0f851d09fccafe3e835)
  provided ideas for tool-call masking, the startup boot check, cached
  prompt-boundary reuse and scheduler back-off, Mamba in-flight reservation,
  the image layer-count guard and the default reasoning-effort setting.
  These are idea credits, not a claim that we imported its AGPL implementation.
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport/commit/9925c45ab4c0740dfc8e77b4715f665e8b205447)
  and lazymio provided the shared-expert stream-reordering implementation
  adapted in the engine (Apache-2.0).
- [JJ48](https://github.com/JJ48/glm53-flash-170hx-serving) published the
  acceptance/step-cost approach that informed acceptance-adaptive draft depth.
- [TensorFold](https://github.com/ashhart/TensorFold/tree/17c73e1) informed
  confidence-gated draft skipping; that optional engine path is disabled by
  this release, but remains in its source.
- [kindlingai's GX10 recipe](https://github.com/kindlingai/glm-5.3-flash-gx10)
  provided ideas for GLM KDA state recovery, unused-draft work and
  exact BF16 mHC weight storage. No code from that unlicensed recipe was used.
- The engine builds on [Marlin](https://github.com/IST-DASLab/marlin)
  (Elias Frantar and subsequent Neural Magic contributors),
  [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
  (Songlin Yang, Yu Zhang and contributors), and DeepSeek's
  [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) reference implementation.
  Original source notices and component licences apply separately.
