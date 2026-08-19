# infernece-preflight-gist

Preflight for distributed vLLM inference on Kubernetes with LeaderWorkerSet.

Draft v0.1 — work in progress, several items marked `[to verify]`.

## Contents

- [lws-inference-preflight-checklist.md](lws-inference-preflight-checklist.md) — the layered
  preflight checks, severity (error/warn) and failure-class (retry/reschedule/stop) model.
- [lws-cross-node-preflight.md](lws-cross-node-preflight.md) — how to run cross-node
  NCCL and ib_write_bw checks inside LWS init containers.

## Context

- LWS KEP #813 (preflight for distributed inference): https://github.com/kubernetes-sigs/lws/pull/813
- Ecosystem prior art — NVIDIA NVSentinel preflight: https://github.com/NVIDIA/NVSentinel/blob/main/docs/configuration/preflight.md

## Status

POC-tested in part; commands not yet run end-to-end are annotated with `[to verify]`.
