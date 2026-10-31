#!/usr/bin/env python3
"""TP4 decode all-reduce microbench through the engine's own TP-group dispatch (torchrun --nproc-per-node 4).

The backend is chosen exactly as the server chooses it, by env (set by the caller):
  hybrid   VLLM_HOST_ALLREDUCE=1 VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0                 -> HOSTSHM
  custom   VLLM_HOST_ALLREDUCE=1 VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1 [VLLM_CUSTOM_ALLREDUCE_FLAGS=0|1] -> CUSTOM
  fipcie   VLLM_HOST_ALLREDUCE=0 VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1          -> FLASHINFER_PCIE_IPC
  nccl     VLLM_HOST_ALLREDUCE=0 and AR_NO_CUSTOM=1                                 -> PYNCCL
Per shape [M, 4096] bf16: ITERS all-reduces captured back to back in one CUDA graph, replayed REPS times;
time per call = slowest rank's replay / ITERS, best of REPS. Correctness: result vs an fp64 sum of the
four ranks' inputs (max abs error, and the share of elements that equal the exactly rounded bf16 sum).
"""
import json, os, sys, time
import torch, torch.distributed as dist

HID = 4096
MS = [int(m) for m in os.environ.get("AR_MS", "1,2,4,6,8,16,32").split(",")]
ITERS, REPS = int(os.environ.get("AR_ITERS", "200")), 7


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    if os.environ.get("AR_NO_CUSTOM") == "1":
        ps.set_custom_all_reduce(False)
    with set_current_vllm_config(VllmConfig()):
        ps.init_distributed_environment(world, rank, "env://", rank, "nccl")
        ps.initialize_model_parallel(tensor_model_parallel_size=world)
        tp = ps.get_tp_group()
        comm = tp.device_communicator
        fi = getattr(comm, "fi_pcie_ipc_ar_comm", None)
        if fi is not None and not fi.disabled:
            from pathlib import Path
            fi.setup(hidden_dim=HID, dtype=torch.bfloat16, capture_sizes=sorted(set(MS + [64])),
                     tune_cache=Path(os.environ.get("AR_FI_CACHE", "/out/fi_tune.json")))
        dev = torch.device("cuda", rank)
        live = [n for n, a in (("FLASHINFER_PCIE_IPC", fi), ("HOSTSHM", getattr(comm, "hostshm_comm", None)),
                               ("CUSTOM", comm.ca_comm), ("PYNCCL", comm.pynccl_comm))
                if a is not None and not getattr(a, "disabled", False)]
        if rank == 0:
            print(f"{os.environ.get('AR_LABEL', '?'):10s} live backends in dispatch order: {live}"
                  f" (ca class {type(comm.ca_comm).__name__ if comm.ca_comm else None})", flush=True)
        rows = []
        for M in MS:
            g = torch.Generator(device="cpu").manual_seed(1000 + M)
            allx = [torch.randn(M, HID, generator=g).to(torch.bfloat16) for _ in range(world)]
            x = allx[rank].to(dev)
            exact = sum(a.double() for a in allx)
            ref_bf16 = exact.to(torch.bfloat16)
            # eager correctness
            y = tp.all_reduce(x.clone()).float().cpu()
            err = (y.double() - exact).abs().max().item()
            exact_share = (y.to(torch.bfloat16) == ref_bf16).float().mean().item()
            # graph timing
            graph = torch.cuda.CUDAGraph()
            with ps.graph_capture(dev) as ctx:
                xs = x.clone()
                for _ in range(3):
                    tp.all_reduce(xs)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, stream=ctx.stream):
                    for _ in range(ITERS):
                        out = tp.all_reduce(xs)
            torch.cuda.synchronize()
            best = 1e9
            for _ in range(REPS):
                dist.barrier(group=tp.cpu_group)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); graph.replay(); e1.record(); e1.synchronize()
                t = torch.tensor([e0.elapsed_time(e1)])
                dist.all_reduce(t, op=dist.ReduceOp.MAX, group=tp.cpu_group)
                best = min(best, t.item())
            # result of the captured calls must match the eager result bitwise
            gy = out.float().cpu()
            rows.append({"M": M, "bytes": M * HID * 2, "us": 1000 * best / ITERS, "max_abs_err": err,
                         "exact_bf16_share": exact_share, "graph_eq_eager": bool(torch.equal(gy, y))})
            del graph
        if rank == 0:
            sel = comm.__dict__.get("hostshm_comm") is not None
            print("RESULT", json.dumps({"label": os.environ.get("AR_LABEL", "?"), "rows": rows}))
            for r in rows:
                print(f"{os.environ.get('AR_LABEL','?'):10s} M={r['M']:3d} {r['bytes']:7d} B  {r['us']:7.2f} us  "
                      f"maxerr={r['max_abs_err']:.4f} exact={100*r['exact_bf16_share']:.1f}%  graph==eager {r['graph_eq_eager']}")
    dist.barrier()


if __name__ == "__main__":
    main()
