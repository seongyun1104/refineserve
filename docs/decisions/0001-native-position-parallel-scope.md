# ADR 0001: Native position-parallel MoE runtime scope

- Status: accepted
- Date: 2026-08-04

## Decision

This project studies native position-parallel generation and refinement workloads in
an expert-parallel MoE runtime. Workloads produce ready positions, the runtime executes
them through attention and MoE layers, and the workload directly commits completed
work into request progress.

## Research question

> Can native position-parallel refinement improve multi-GPU MoE Expert Parallel
> wall-clock efficiency, and can critical-path-aware scheduling control the resulting
> expert scattering, communication, and load imbalance?

Diffusion is one workload generator rather than the platform identity. Other native
decode paths can be added behind the same work-item interface.

## Stable architecture boundary

```text
native decode workload
  -> ready work items
  -> admission and batch formation
  -> attention and layer-local routing
  -> EP dispatch / expert compute / combine
  -> direct workload finalization
  -> metrics and calibration
```

```python
class DecodeWorkload:
    def ready_work_items(self) -> list[WorkItem]: ...
    def finalize(self, completed_items: list[WorkItem]) -> None: ...


@dataclass
class WorkItem:
    request_id: int
    position_id: int
    iteration: int
    is_finalization_eligible: bool
    route_signature: list[int] | None
```

Current implementations:

```text
AutoregressiveWorkload
BlockRefinementWorkload
```

Planned implementations:

```text
MaskedParallelWorkload
DiffusionWorkload
```

## Runtime optimization boundary

The primary track holds routing behavior fixed and optimizes request composition and
execution timing. Router-changing methods can later provide input traces, but they are
not part of the M1 runtime scheduler.

For a candidate batch, M1 estimates:

```text
estimated_batch_time
  = sum over layers [
      max over EP ranks (
          dispatch_time
        + unique_expert_weight_time
        + grouped_expert_compute_time
        + combine_time
      )
    ]
  + queue/deadline penalty
  + KV-rank load penalty
  + request-progress fragmentation penalty
  - locality reuse benefit
```

Required scheduler baselines:

```text
FIFO
locality_only
load_balance_only
critical_path_only
locality_plus_load
joint
routing_oracle
runtime_oracle
```

Request-progress fragmentation and underfilled-batch metrics are required because a
myopic layer-cost minimizer can improve one batch while worsening the remaining batch
sequence. A small-workload exact global makespan oracle measures this gap.

## Growth path

```text
deterministic simulator
  -> multi-seed/scenario evaluation
  -> trace schema and replay calibration
  -> single-GPU toy MoE
  -> four-GPU NCCL Expert Parallel prototype
  -> native block-refinement model
  -> full diffusion evaluation
```
