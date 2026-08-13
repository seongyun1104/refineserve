# Conditional upstream strategy (inactive)

Status: **CLOSED — NATIVE EVIDENCE GATE DID NOT PASS**

Last checked: 2026-08-13

## Final disposition

M2.1 found less than 0.5% best-found request-composition headroom in every measured
LLaDA2.0-mini cell. RefineServe is therefore closed without an SGLang RFC, vLLM port,
or framework adapter. The material below is retained only as the historical strategy
that would have applied after a positive native result. It is not an active roadmap.

See the [project closure](project_closure.md).

## Historical conditional decision

If RefineServe passes its native route, native-shape timing, correctness, and
end-to-end scheduler gates, the preferred upstream order is:

```text
1. SGLang RFC and incremental opt-in changes
2. vLLM cross-runtime port and DiffusionGemma validation
```

This is a conditional integration strategy, not a current implementation milestone.
M2.1, M2.2, M3, and M4 remain framework-independent evidence gates. No upstream RFC
or framework adapter work begins before M4 produces a reproducible native-model result.

## Why SGLang is the first candidate

SGLang currently contains a block-diffusion framework, LLaDA2 support, dynamic
batching work, a dLLM scheduling implementation, DeepEP-backed MoE execution, and
EPLB support. This places request admission, native denoising state, and MoE execution
in one runtime, which is the integration boundary RefineServe needs.

The relevant control remains request composition:

```text
model-selected native denoising work
  -> ready request pool
  -> coordinated request composition
  -> MoE EP dispatch, expert compute, and combine
```

The runtime policy may alter request grouping and execution placement. It may not alter
the router, selected positions, denoising order, remasking policy, or finalization
semantics.

The SGLang roadmap issue used as evidence is currently closed with an inactive label.
It documents implemented and proposed work but is not treated as a commitment to
future interfaces. The current source tree and maintainer feedback must be checked
again before an RFC is filed.

## Current SGLang topology limitation

The current LLaDA2 DeepEP implementation sets:

```text
ep_size = tp_size
```

and contains a TODO for `tp < ep`. Therefore the desired topology:

```text
TP=1
EP=4
replicated attention
rank-owned experts
```

is not assumed to work in the current LLaDA2 path. This limitation is kept separate
from the scheduler contribution. RefineServe must first prove the mechanism in its own
true-EP adapter. An upstream RFC should ask maintainers whether topology plumbing and
scheduler integration should be separate efforts.

## Why vLLM is second

vLLM's EP deployment model directly supports the desired topology. With EP enabled,
`EP_SIZE = TP_SIZE * DP_SIZE`; at `TP=1`, attention is replicated across DP ranks while
experts are distributed across all EP ranks.

The integration risk is the scheduler surface. The current DiffusionGemma integration
deliberately reuses the existing engine scheduler and model runner with minimal core
changes. A separate dLLM RFC proposes custom scheduler and worker plugins, but vLLM's
configuration warns that the custom scheduler interface is not public and compatibility
may not be maintained.

For this reason, a vLLM port is more defensible after RefineServe has already shown the
same mechanism in a native LLaDA2 runtime. DiffusionGemma then serves as cross-model and
cross-runtime validation, not as a replacement for the LLaDA2 result.

Internal framework reuse of speculative-accounting fields does not change the
RefineServe research scope. RefineServe does not add draft generation, target
verification, acceptance optimization, or speculative-width control.

## Upstream evidence prerequisites

An SGLang RFC requires all of the following:

```text
M2.1  measured native LLaDA2 route opportunity is characterized
M2.2  native-shape recoverable time exceeds measurement resolution and policy cost
M3    true-EP LLaDA2 adapter passes layer-wise and generation correctness gates
M4    coordinated scheduling improves end-to-end native execution after overhead
>=3   independent trace segments or workload realizations retain the result
```

A mapping-dependent or workload-dependent result is acceptable, but the RFC must state
the valid regime and retain negative cells. A route-space gain alone is insufficient.

## Proposed SGLang contribution sequence

### RFC

Suggested title:

```text
[RFC][DLLM][MoE] Critical-path-aware batch composition for native
block-diffusion Expert Parallel inference
```

The RFC should present the measured problem, the model-semantics boundary, estimator
inputs available online, scheduler overhead, and fallback behavior. It should not ask
to merge the RefineServe repository as a subsystem.

### PR 1: instrumentation

Add opt-in, behavior-preserving metrics:

```text
denoising step and block identity
active and finalized position counts
per-layer expert/rank load
critical-rank load and time
dispatch/combine breakdown where available
```

Default scheduling and model output remain unchanged.

### PR 2: policy interface

Add the smallest maintainer-approved extension point for dLLM batch composition.
FCFS remains the default. The interface must receive only pre-execution observable
state and must preserve the model-selected work set.

### PR 3: RefineServe policy

Add an opt-in coordinated critical-path policy with:

```text
previous-route estimator
bounded selection time
FIFO fallback
all-rank composition broadcast
scheduler overhead metrics
```

The coordinator must produce one composition shared by all EP ranks. Rank-local
independent scheduling is not an acceptable implementation.

## vLLM port criteria

The vLLM port begins only after the SGLang implementation or equivalent native LLaDA2
prototype establishes the mechanism. Before coding, re-evaluate:

```text
custom scheduler interface stability
DiffusionGemma model-state contract
EP and all-to-all backend compatibility
request-level composition control surface
metrics semantics for native diffusion runs
```

The target result is mechanism generalization across model and runtime, not parity of
internal implementation.

## Re-evaluation condition

`SGLang first` is not permanent. Reverse or revise the order if, at upstream time:

- SGLang still cannot express `TP=1, EP=4` for LLaDA2 without a broad refactor;
- the required dLLM batch-composition hook is rejected or unavailable;
- vLLM exposes a stable scheduler contract and native model with the required control;
- the measured mechanism is tied to a substrate better represented by vLLM.

The final upstream choice is made from current source and maintainer feedback after M4,
not from the 2026-08-13 repository snapshot alone.

## Verified references

- [SGLang dLLM roadmap issue](https://github.com/sgl-project/sglang/issues/14199)
- [SGLang LLaDA2 implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llada2.py)
- [SGLang DeepEP and EPLB server arguments](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/server_arguments.mdx)
- [vLLM expert-parallel deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)
- [vLLM DiffusionGemma integration](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-06-10-diffusion-gemma.md)
- [vLLM dLLM plugin RFC](https://github.com/vllm-project/vllm/issues/36155)
- [vLLM scheduler configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)
