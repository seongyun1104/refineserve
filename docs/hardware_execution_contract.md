# Hardware Execution Contract

## 1. Project identity

This project explores decode optimization mechanisms beyond speculative decoding.

The primary research axis is:

```text
Native diffusion / block-refinement position parallelism
×
MoE Expert Parallel execution
×
Critical-path-aware scheduling
```

The project does not use draft-and-verify speculative decoding as its core execution model.

## 2. Primary research question

> Can native position-parallel refinement improve the wall-clock efficiency of
> multi-GPU MoE Expert Parallel execution, and can critical-path-aware scheduling
> control the resulting expert scattering, communication overhead, and GPU imbalance?

The controlled hardware variable is:

```text
K = number of simultaneously active refinement positions
```

`K` is not a speculative token count. Each active position is a direct unit of native
refinement work. There is no target verifier, accepted prefix, rejection, or rollback.

`K` is also not a synonym for the model's decoding block width. Native-model
experiments must keep these four values distinct:

```text
block_width                   = range in which positions may be selected
active_position_count (K)     = positions that actually execute the MoE path
finalized_positions_per_step  = useful model progress in that iteration
order_policy                  = model/sampler rule for choosing positions
```

The controlled prototype sweeps `active_position_count`. It does not claim to sweep
arbitrary-order freedom. Its synthetic finalized-work counter must not be used as a
substitute for native-model quality or progress.

## 3. Non-negotiable invariants

Every primary experiment must preserve all of the following.

1. The workload exposes `K > 1` native active positions.
2. Active positions directly enter the MoE execution path.
3. The measured path contains real multi-GPU EP dispatch, expert compute, and combine.
4. Results are compared against `K=1`.
5. Scheduler overhead is included in wall-clock measurements.
6. The layer critical path is determined by the slowest participating rank.
7. Useful finalized work and total executed work are reported separately.
8. Results do not rely on draft acceptance or speculative verification.
9. The runtime scheduler does not change model-selected positions, denoising order,
   remasking, or block-finalization semantics.

Any experiment missing one of these requirements is supporting calibration, not primary evidence.

## 4. Explicitly out of scope

Do not add the following to the current hardware milestone:

- DFlash
- EAGLE
- Medusa
- draft model execution
- target-model verification
- accept/reject logic
- speculative decoding
- speculative width control
- MTP
- tree verification
- acceptance-rate optimization
- full diffusion model training
- text quality evaluation
- large AR MoE model benchmarking as the main experiment
- framework integration work that does not validate the primary hypothesis

Do not reinterpret active-position width `K` as speculative-token width.

## 5. Role of ordinary AR MoE models

Ordinary AR MoE models may only be used for supporting measurements such as router
trace collection, layer-wise routing entropy, expert skew, routing persistence, expert
kernel calibration, message-size distribution, and NCCL calibration. These measurements
may parameterize or validate the simulator. They must not be presented as direct
validation of native position-parallel refinement.

The evidence hierarchy is:

```text
Primary evidence
1. Native K-position EP hardware prototype
2. Native block-refinement model

Supporting evidence
3. Measured routing traces from an AR MoE model
4. Expert kernel and NCCL microbenchmarks
5. Synthetic simulation
```

## 6. Current H100×4 primary experiment

Run a controllable native position-parallel MoE EP prototype on four H100 GPUs.

### Default model shape

```text
Transformer/MoE blocks: 8
Experts: 16
Top-k routing: 2
Expert Parallel size: 4
Tensor Parallel size: 1
Pipeline Parallel size: 1
Hidden size: 2048
Activation dtype: BF16
```

### Active-position sweep

```text
K = 1, 2, 4, 8, 16, 32, 64
```

The existing block-refinement schedule may also be evaluated:

```text
32 → 24 → 16 → 12 → 8 → 4 → 2 → 1
```

Each active position must flow through:

```text
Router
→ EP dispatch
→ rank-local expert compute
→ EP combine
```

No speculative verifier may be inserted.

### Native-model validation order

The controlled K-sweep above remains the primary causal experiment. After its
matrix and profiler artifacts are complete, native-model validation uses this
order:

```text
1. inclusionAI/LLaDA-MoE-7B-A1B-Instruct: EP adapter correctness smoke test
2. inclusionAI/LLaDA2.0-mini: primary native block-diffusion MoE validation
3. DiffusionGemma-26B-A4B-it: secondary cross-model validation
```

The immediate gated roadmap is stricter than the model list:

```text
M2.1  native LLaDA2 route opportunity, no EP timing claim
M2.2  native-shape 256-expert/top-8 true-EP timing replay
M3    full LLaDA2.0-mini true-EP adapter only if both gates pass
M4    native FIFO versus coordinated scheduler timing
M5    width/policy ablation with quality constraints
```

For `LLaDA2.0-mini`, the standard checkpoint and generation semantics are held
fixed. The initial primary setting is `block_length=32`, `steps=32`, BF16,
EP=4, TP=1, and PP=1. Router scoring, remasking, model weights, and block
finalization are not modified in the first comparison. The stock dInfer
multi-GPU tensor-parallel MoE path is supporting bring-up only; primary
evidence requires rank-owned experts and real all-to-all dispatch and combine.

The model's block length is not silently equated with the controlled
prototype's K. The controlled K-sweep establishes the position-width causal
effect; the model experiment tests whether that phenomenon survives a native
block-diffusion workload.

The model/sampler owns position selection and finalization. The runtime scheduler may
change request batch composition and execution placement only. It may not choose easier
positions for locality, defer high-entropy positions, or change the model's order
policy. This boundary follows the distinction between parallel decoding and
arbitrary-order generation established by
[The Flexibility Trap / JustGRPO](https://arxiv.org/abs/2601.15165v4).

Quality is outside the current controlled H100 milestone, but it becomes a required
constraint for native-model width ablations. A native-model result must report useful
progress and task quality alongside systems throughput; the fastest width is not the
preferred width if it changes model semantics or violates the declared quality budget.
JustGRPO itself is related work and a possible future post-training track, not a model
or training dependency of the current EP runtime milestone.

Native-model width ablations therefore record at minimum:

```text
block_width
active_position_count
finalized_positions_per_step
order_policy
task accuracy / Pass@1 / Pass@k where applicable
entropy of finalized positions
high-entropy position deferral
wall-clock latency
useful finalized tokens per second
```

These quality metrics constrain later native-model conclusions; they do not retroactively
turn the controlled synthetic K-sweep into a text-quality experiment.

## 7. Required baselines

Run at least the following scheduler arms:

```text
1. FIFO
2. Locality-only
3. Load-balance-only
4. Critical-path-aware
5. Joint scheduler
6. Oracle routing upper bound, when available
```

Use identical route bundles and arrival workloads across scheduler arms. Do not
regenerate routes independently for each scheduler.

## 8. Required routing conditions

```text
Uniform
Mild expert skew
Strong expert skew
Hot expert
Request-correlated routing
Temporally stable routing
Temporally unstable routing
Measured trace replay
```

Measured route replay is supporting calibration until it is combined with a native
active-position workload.

## 9. Required measurements

Record wall-clock values from synchronized CUDA or profiler measurements.

### End-to-end

- per-layer makespan
- eight-layer makespan
- P50, P95, and P99 latency
- useful finalized positions per second
- total executed positions per second
- work amplification
- scheduler selection latency
- scheduler fraction of total runtime

### Expert execution

- tokens per expert
- active unique experts
- active experts per rank
- expert invocation count
- expert batch-size histogram
- rank-local expert compute time
- grouped-kernel utilization proxy
- expert load imbalance

### Communication

- dispatch time
- combine time
- bytes per rank pair
- non-empty peer count
- average and maximum peer message size
- communication fraction
- NCCL synchronization time
- rank idle or straggler time

### Correctness

- dispatched token count
- combined token count
- route consistency
- no dropped or duplicated positions
- deterministic replay under a fixed seed

## 10. Measurement rules

- Warmup iterations must be excluded.
- Raw samples must be retained.
- Median and P95 must both be reported.
- GPU synchronization rules must be identical across arms.
- Correctness checks and host-visible metric extraction must be outside the isolated
  CUDA data-path interval. End-to-end control-plane time is recorded separately and is
  not hidden.
- Scheduler selection time must not be hidden.
- Calibration values must not be extrapolated beyond their measured range.
- Monotonic fitted curves must be stored alongside raw measurements.
- Online schedulers may only use information available before execution.
- Actual future routes or future runtime values are oracle-only inputs.
- P95/P99 remain descriptive unless the per-cell sample count is large enough for a
  preregistered tail analysis; P50 and paired differences are primary for small
  diagnostic reruns.
- Clock-lock success, SM/memory clocks, temperature, and power must be retained for
  percent-level comparisons.

## 11. H100 rental-time priority order

```text
P0. Save environment and topology metadata
P1. Validate four-rank NCCL and all-to-all correctness
P2. Measure rank-local expert kernels over K=1…64
P3. Measure dispatch and combine over K=1…64
P4. Run the complete eight-layer EP path
P5. Compare FIFO and critical-path-aware scheduling
P6. Run at least three seeds
P7. Run routing-skew ablations
P8. Collect profiler traces
P9. Collect ordinary AR MoE routing traces only if time remains
P10. Framework adapter experiments only after everything above
```

If rental time becomes limited, stop after P8. Do not replace the core experiment with
a large-model demonstration.

## 12. Success and failure interpretation

The project does not assume that larger `K` is always faster. Positive and negative
results both answer the research question. Preserve expert scattering, communication,
imbalance, and scheduler-overhead boundaries rather than tuning them away.

## 13. Stop condition for scope drift

Stop the current task immediately when implementation begins optimizing draft quality,
acceptance rate, accepted prefix length, target verification, speculative K, MTP
acceptance, rejection recovery, or text-generation quality. Return to the native
active-position EP pipeline.

## 14. Canonical project statement

> This project investigates decode optimization beyond speculative decoding. Its first
> research track studies whether native position-parallel refinement can be converted
> into efficient multi-GPU MoE Expert Parallel execution through critical-path-aware
> scheduling.
