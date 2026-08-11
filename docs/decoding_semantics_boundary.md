# Decoding Semantics Boundary

The runtime optimizes execution of positions selected by the native model or sampler.
It does not select, defer, remask, or finalize positions itself.

## Separate variables

```text
block_width
  The range over which the model or sampler may choose positions.

active_position_count
  The number of selected positions that execute the MoE path in this iteration.

finalized_positions_per_step
  Useful model progress committed by this iteration.

order_policy
  The model/sampler position-selection rule, such as left-to-right, confidence-based,
  entropy-bounded, or model-defined.
```

The controlled hardware prototype varies only `active_position_count`. Its `K` sweep
does not establish a quality effect of arbitrary-order block width.

## Ownership

```text
Model / sampler
  Owns active-position selection, denoising order, remasking, and finalization.

Runtime scheduler
  Owns request batch composition, rank placement, and execution ordering of already
  selected work.
```

The runtime scheduler may not change model-owned `RefinementState`. The simulator
checks this invariant around every online selection call.

## Related-work interpretation

[The Flexibility Trap / JustGRPO](https://arxiv.org/abs/2601.15165v4) separates
parallel decoding from arbitrary-order generation and reports that confidence-driven
order freedom can reduce reasoning-path coverage in the evaluated reasoning and code
settings. The official [ICML 2026 awards
announcement](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/)
highlights that JustGRPO uses fixed left-to-right RL rollouts while retaining parallel
decoding at inference.

For this project, that result is a scope and measurement guardrail:

- It does not invalidate position-parallel EP execution.
- It prohibits interpreting systems K as arbitrary-order block width.
- It requires quality and useful-progress gates in later native-model width ablations.
- It does not add GRPO training to the current hardware milestone.

Canonical rule:

> We optimize the execution of model-selected parallel positions without increasing
> arbitrary-order freedom or altering the model's denoising semantics.
