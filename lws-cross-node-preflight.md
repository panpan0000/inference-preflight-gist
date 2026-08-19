# Cross-node NCCL / ib_write_bw inside LWS init containers

Draft v0.1 — [to verify] = not yet run; check before relying on it.

Why this is hard: init containers run before the main container and before the
pod is Ready, so nothing you normally rely on is guaranteed yet.

## Pod spec essentials (1 GPU per pod, the wide-EP shape)
```yaml
hostIPC: true                  # same-node UCX CUDA-IPC must see peer GPU memory
initContainers:
  - name: preflight
    resources: { limits: { nvidia.com/gpu: "1" } }   # GPU is NOT inherited; init needs its own request
    securityContext:
      capabilities: { add: ["IPC_LOCK"] }            # required for RDMA memory registration
# RDMA VF arrives via the pod's multus SR-IOV network (pod-level netns -> visible to init containers)
```

## 1. Ranks from LWS env (no launcher needed)
```
RANK=$LWS_WORKER_INDEX     # leader == 0   [to verify: exact semantics in our LWS version]
WORLD=$LWS_GROUP_SIZE
```

## 2. Peer discovery without a DNS guarantee  (KEP #813, gap 1)
The leader's *service* DNS may not resolve during init (endpoint programming,
DNS propagation). Pattern: bounded retry on `LWS_LEADER_ADDRESS`, pod-DNS fallback
(pod A records exist from pod creation, before Ready):
```bash
until getent hosts "$LWS_LEADER_ADDRESS" ||
      getent hosts "${LEADER_POD}.${POD_NAMESPACE}.pod.cluster.local"; do
  sleep 2; n=$((n+1)); [ "$n" -ge 120 ] && exit 3   # transient -> retry
done
```

## 3. Rendezvous (collectives need all ranks) — verified paths
- PRIMARY (launcher-free): `torch.distributed` with `tcp://` rendezvous. Rank 0
  (leader) hosts it; no mpirun, ssh, or shared FS:
```python
init_process_group(backend="nccl",
                   init_method="tcp://<LEADER_ADDR>:29500",
                   rank=int(os.environ["LWS_WORKER_INDEX"]),
                   world_size=int(os.environ["LWS_GROUP_SIZE"]))
# then one all_reduce of a 64MB tensor; report time + NCCL version + env
```
- NOT recommended cross-node: `nccl-tests`. Upstream README: "NCCL tests rely
  on MPI to work on multiple processes, hence multiple nodes" -> requires
  `MPI=1` build + mpirun with a hostfile of pod FQDNs. Heavy in init containers.
- Deadline-wrap every stage: `timeout(1)` around rendezvous + the collective.

## 4. ib_write_bw pairing (GPU-direct, not TCP fallback)
```
leader:  ib_write_bw -d <rdma_dev>                        # server
workers: ib_write_bw -d <rdma_dev> --use_cuda=0 $LEADER_IP
```
[to verify: `--use_cuda` needs a perftest build with CUDA support]

## 5. Results & exit codes
```
JSON + exit code contract: 3=transient->retry, 4=node/hw->reschedule, 5=config->stop.
Today: enforced by script + LWS RecreateGroupOnPodRestart.
Gaps: recreate loops are unbounded; no terminal failed state; restart counter
resets; --previous logs vanish. -> KEP #813 retry-budget / terminal-status
proposal (opt-in, backward compatible).
```

## Contrast: NVSentinel gang checks
NVSentinel's `preflight-nccl-allreduce` needs a gang-aware scheduler and the
K8s 1.35/1.36 alpha Workload / PodGroup APIs. LWS group semantics
(`LWS_LEADER_ADDRESS` + group size) are another path to the same rendezvous —
this is the peer-discovery discussion behind KEP #813.
