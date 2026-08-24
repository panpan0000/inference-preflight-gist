# Captured evidence

These files are a compact record of the successful 2026-08-24 run in the
`peter` namespace:

- [`pod-status.txt`](pod-status.txt) — Pod placement and LWS `AllGroupsReady`.
- [`pod-nccl-preflight.log`](pod-nccl-preflight.log) — relevant NCCL and
  standard all-reduce output.

The evidence is intentionally an excerpt, not a claim that every NCCL
warning is harmless. In particular, the run reported GPU Direct RDMA as
disabled; the result proves cross-node NCCL initialization and the tested
collective path, not GPU-direct-RDMA performance.
