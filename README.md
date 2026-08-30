# inference-preflight-gist

Preflight for distributed vLLM inference on Kubernetes with LeaderWorkerSet.

The runnable demo is [`examples/lws-vllm-nccl-preflight.yaml`](examples/lws-vllm-nccl-preflight.yaml).
It runs the vLLM collective benchmark in an init container before the
workload container starts.

## Contents

- [lws-inference-preflight-checklist.md](lws-inference-preflight-checklist.md) — the layered
  preflight checks, severity (error/warn) and failure-class (retry/reschedule/stop) model.
- [lws-cross-node-preflight.md](lws-cross-node-preflight.md) — how to run cross-node
  NCCL and ib_write_bw checks inside LWS init containers.
- [examples/lws-vllm-nccl-preflight.yaml](examples/lws-vllm-nccl-preflight.yaml) —
  minimal two-Pod, one-GPU-per-Pod LWS demo.
- [examples/lws-nccl-tests-preflight.yaml](examples/lws-nccl-tests-preflight.yaml) —
  `mpirun` from LWS leader init-container to its worker init-container.
- [examples/dcgmi-remote-hostengine.yaml](examples/dcgmi-remote-hostengine.yaml) —
  connect `dcgmi` to the same node's standalone hostengine.
- [examples/privileged-dcgm-diag.yaml](examples/privileged-dcgm-diag.yaml) —
  isolated privileged DCGM level-1 diagnostic.
- [examples/lws-multi-init-results.yaml](examples/lws-multi-init-results.yaml) —
  multiple init images and a final per-Pod result gate.
- [log/](log/) — captured Pod placement, LWS status, and relevant NCCL output.

## Run the demo

The example assumes the LWS CRD/controller is installed and that the image is
available on every candidate node:

```bash
kubectl -n peter apply -f examples/lws-vllm-nccl-preflight.yaml
kubectl -n peter get pods -l app=vllm-nccl-preflight -o wide
kubectl -n peter logs -f vllm-nccl-preflight-0 -c nccl-preflight
```

The `peter` namespace and private image in the example are from the captured
run; change them for another cluster.

## What this proves

The captured run placed the two Pods on `gpu-10-125-2-62` and
`gpu-10-125-2-51`, and LWS reported `AllGroupsReady`. NCCL formed a two-rank,
two-node communicator over RoCE/IB and the standard vLLM collective path
completed for 128, 512, and 2048 tokens. See [`log/`](log/) for the evidence.

This is a vLLM-native collective benchmark, not NVIDIA `nccl-tests`
`all_reduce_perf`; it reports operation latency rather than `algbw`/`busbw`.

## Run the MPI preflight

Create the SSH Secret once, then apply the LWS example:

```bash
ssh-keygen -q -t ed25519 -N '' -f /tmp/nccl-ssh
cp /tmp/nccl-ssh.pub /tmp/authorized_keys
kubectl -n peter create secret generic nccl-ssh --from-file=id_ed25519=/tmp/nccl-ssh --from-file=authorized_keys=/tmp/authorized_keys
kubectl -n peter apply -f examples/lws-nccl-tests-preflight.yaml
```

The worker init-container runs `sshd`; the leader waits for it, uses `mpirun`
to launch two ranks, then returns the same exit code to the worker. `Results:`
with `#wrong 0` is the success signal. Replace `ghcr.io` with your mirror if
needed.

## Why the YAML contains two small runtime patches

The available `vllm-openai:v0.17.0` image contains an older benchmark that uses
global `RANK` as the CUDA ordinal. Because each Pod receives one GPU, the demo
patches it to `cuda:0` before invoking `torchrun`. The same version includes
FlashInfer backends that are not usable for this cross-node RTX 5090 run, so
the demo disables those backends and measures the standard NCCL path.

The init container also mounts a 10 GiB memory-backed `/dev/shm`; without it,
NCCL can fail with `No space left on device` while creating its shared-memory
segments.

## DCGM access

GPU Operator normally runs DCGM Exporter with an embedded hostengine. Port
`9400` is Prometheus HTTP, not a `dcgmi` endpoint. The captured exporter has no
`DCGM_REMOTE_HOSTENGINE_INFO`, so `dcgmi` cannot connect to that Pod.

For metrics only:

```bash
kubectl -n gpu-operator port-forward pod/nvidia-dcgm-exporter-vrqt4 9400:9400
curl -s http://127.0.0.1:9400/metrics | head
```

For `dcgmi`, enable GPU Operator's standalone DCGM (`dcgm.enabled=true`). It
exposes `nv-hostengine` on each node at port `5555`; the example uses
`status.hostIP` to select the Pod's own node. The remote client is unprivileged;
only the isolated example that starts its own hostengine uses `privileged: true`.
Replace public registries with cluster mirrors where required.

## Result scope

An `emptyDir` is shared only by containers in one Pod. It can combine several
init-container results before a final gate, but it cannot aggregate every Pod
in an LWS group. Cross-Pod summaries need a leader-side collector or external
storage.

## Context

- LWS KEP #813 (preflight for distributed inference): https://github.com/kubernetes-sigs/lws/pull/813
- Ecosystem prior art — NVIDIA NVSentinel preflight: https://github.com/NVIDIA/NVSentinel/blob/main/docs/configuration/preflight.md
- NVIDIA DCGM: https://docs.nvidia.com/datacenter/dcgm/latest/
- NVIDIA nccl-tests: https://github.com/NVIDIA/nccl-tests

## Status

The LWS demo was run end-to-end on 2026-08-24. The broader checklist and
`ib_write_bw`/GPU-direct-RDMA sections remain design notes and are still marked
`[to verify]` where applicable.
