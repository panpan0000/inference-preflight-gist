# Preflight Checklist: Distributed vLLM Inference on Kubernetes (LWS)

Draft v0.1 — [to verify] = not yet run in our POC, check before relying on it.

Checks run in init containers before the serving container starts.
Each check emits JSON to /preflight/result.json and exits with a class code:

```
3 = transient  -> retry
4 = node / hw  -> reschedule
5 = config     -> stop
```

Severity is two-level and exit-0 only depends on `error` items:

```
error  -> blocks the pod (nonzero exit)
warn   -> recorded in result.json, never blocks (exit 0)
```

## 1. Node level
| Check | Command | sev | class |
|---|---|---|---|
| GPU persistence mode | `nvidia-smi -pm 1` (revert after test) | warn | 4 |
| CPU governor = performance | `echo performance > /sys/.../scaling_governor` | warn | 4 |
| PCIe ACS off (where allowed) | `lspci -vvv \| grep ACS` | warn | 5 |
| GPU burn / stability | `gpu_burn 60` or DCGM diag level 2 | error | 4 |
| vGPU quirk (MaskGPU) | `/usr/local/maskgpu/*.so.preload` must be empty; TVMFFIGetTypeInfo crash workaround | error | 5 |

## 2. GPU / CUDA
- Host driver vs image CUDA compatibility: `nvidia-smi` vs `torch.version.cuda`  | error | 5
- DCGM health level 2 on all visible GPUs                                   | error | 4
- Blackwell [to verify]: MLNX stack is EOL -> DOCA stack; `nvlsm` package
  required (NVLink dependency); `nv-fabric` svc + `peer-mem` are deprecated.

## 3. RDMA (the silent killer)
| Check | Fail class |
|---|---|
| MTU must match end-to-end (4218 or 9000): host NIC -> CNI (SR-IOV/Macvlan) -> switch. A one-byte mismatch silently drops jumbo frames | 4 (error) |
| RoCE GID selection correct on the VF in use | 5 (error) |
| `nvidia-peermem` loaded on host (GPU-direct RDMA), else NIC cannot touch GPU memory | 4 (error) |
| Pairwise `ib_write_bw` leader vs each worker, gdr mode `--use_cuda=<dev>` | 4 (error) |
| Mode switch RoCE/IB via `mlxconfig` (see `setNicRdmaMode.sh` in spiderpool); IB mode: `gdrdrv` + `ib_ipoip` | 5 (error) |

## 4. Transport / collectives
- `nccl-tests` all_reduce, shapes matching wide-EP traffic (8M..1G)  [to verify: cross-node needs MPI] | 3 in error->retry
- UCX transport matrix:
  `CUDA-IPC` (same node / NVL72) -> RDMA `rc` (cross-node) -> `TCP` (last resort).
  Silent TCP fallback is painful: not just bandwidth — GPU staging + kernel +
  descriptor overhead compounds with layer-wise KV blocks.
  Default `warn`; `error` when a platform flag like `rdma.required=true` is set. | 3
- DeepEP [to verify]: NVSHMEM + IBGDA enabled.

## 5. Serving-specific probes
| Check | sev | class |
|---|---|---|
| Model weights / shared storage / local cache rw probes | error (required) / warn (optional cache) | 5 |
| `LWS_LEADER_ADDRESS` resolve + port probe (bounded retry) | error after budget | 3 |
| Minimal rendezvous smoke before vLLM starts | error | 3 |

## Prior art (adopt, don't reinvent)
NVIDIA NVSentinel preflight = admission webhook injecting DCGM diagnostics
(level 2 default, `DCGM_DIAG_STATUS_RETRY_MAX_ATTEMPTS=10` x 10s) + NCCL
loopback / gang all-reduce. Its defaults are field-tested and adopted here
with attribution:
- `BW_THRESHOLD_GBPS`: 150 (NVLink) / 15 (PCIe) loopback; 100 multi-node
- `IPC_LOCK` capability required for RDMA memory registration
- Failure model: bounded retry, then **non-blocking** (HealthEvent with
  `RecommendedAction=NONE`, exit 0)

Our gate differs deliberately: we **block**, but classify every failure into
retry / reschedule / stop, and our `warn` level ≈ their non-fatal HealthEvent.
