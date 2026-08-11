# LLaDA Native MoE EP Adapter Design

## Role and order

```text
LLaDA-MoE-7B-A1B-Instruct
  Adapter correctness and weight-layout smoke test

LLaDA2.0-mini
  Primary native block-diffusion MoE validation

DiffusionGemma-26B-A4B-it
  Secondary cross-model validation with a separate architecture adapter
```

The controlled K-sweep remains the causal hardware experiment. The adapter does not
change model routing, denoising, remasking, position selection, or finalization.

## Confirmed checkpoint shapes

### LLaDA2.0-mini

Source: [official checkpoint config](https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/main/config.json)
and [model card](https://huggingface.co/inclusionAI/LLaDA2.0-mini).

```text
hidden_size:                2048
num_hidden_layers:          20
first_k_dense_replace:      1
num_experts:                256
num_experts_per_tok:        8
num_shared_experts:         1
moe_intermediate_size:      512
router_dtype:               FP32
score_function:             sigmoid
norm_topk_prob:             true
routed_scaling_factor:      2.5
n_group / topk_group:       8 / 4
checkpoint dtype:           BF16
recommended block / steps: 32 / 32
```

With EP=4, each rank owns 64 routed experts. The first dense layer is not sent through
the EP adapter.

### LLaDA-MoE-7B-A1B-Instruct

Source: [official checkpoint config](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct/blob/main/config.json).

```text
hidden_size:             2048
num_hidden_layers:       16
num_experts:             64
num_experts_per_tok:     8
expert_intermediate_size: 1024
checkpoint dtype:        BF16
```

With EP=4, each rank owns 16 experts.

## Why stock dInfer is not primary evidence

The stock `LLaDA2MoeSparseMoeBlock` describes and implements a tensor-parallel MoE:
each expert is sharded across all ranks, the fused MoE runs with `reduce_results=True`,
and outputs are reduced. It does not implement rank ownership followed by token
all-to-all dispatch and combine.

Source: [dInfer LLaDA2 MoE implementation](https://github.com/inclusionAI/dInfer/blob/master/python/dinfer/model/modeling_llada2_moe.py).

## EP=4 ownership

```text
rank 0: global experts   0..63
rank 1: global experts  64..127
rank 2: global experts 128..191
rank 3: global experts 192..255
```

For the 7B smoke model the same rule uses 16 experts per rank.

Attention, embeddings, normalization, the dense first layer, and the shared-expert
branch remain TP=1 and DP-local. Routed expert weights alone are sharded by ownership.

## Per-layer execution

```text
Local hidden states for model-selected positions
  -> unchanged FP32 model router
  -> global expert IDs and original top-8 router weights
  -> destination rank = global_expert_id // experts_per_rank
  -> stable all-to-all(hidden, local expert ID)
  -> rank-local grouped expert computation
  -> reverse all-to-all
  -> restore origin slot from the local stable inverse permutation
  -> apply router weights after the nonlinear expert computation
  -> FP32 weighted top-8 sum (weights already include routed scaling)
  -> add locally computed shared-expert branch
  -> continue the unmodified transformer layer
```

The adapter must preserve sigmoid routing, group-limited top-k, expert bias,
normalization, and scaling factor. Uniform top-k averaging is valid only in the toy
prototype and must not appear in the model adapter.

Router weights must never be multiplied into hidden states before expert computation:
SwiGLU is nonlinear, so `FFN(weight * hidden) != weight * FFN(hidden)`. The standard
LLaDA2 gate normalizes the selected sigmoid scores and then multiplies the router
weights by `routed_scaling_factor=2.5`. The EP adapter preserves those weights without
renormalizing them. After reverse all-to-all, it applies them to expert outputs and
performs the routed top-8 accumulation in FP32. The replicated shared-expert output is
then added exactly once. This matches the stock semantic order:

```text
routed = sum_i(gate_weight_i * expert_i(hidden))
output = routed + shared_expert(hidden)
```

The expert-capacity policy must be dropless for the correctness baseline. Request batch
composition is not allowed to change token dropping, routing IDs, or model output.

## Assignment baseline and destination-coalesced path

The first correctness path is assignment-granular: one hidden row is sent per selected
expert. Router weights and origin slots remain on the source rank; stable assignment
order restores returned expert outputs before the source performs the FP32 weighted
sum. This is simple and semantically direct, but it is not the native performance
ceiling.

With 256 experts, top-8, and four equal ownership ranges, uniform expert selection has
an expected 3.61 distinct destination ranks per token. Sending one hidden row for each
of eight expert assignments therefore contains an expected 54.8% assignment-to-
destination duplication. Actual trace collision is measured rather than assumed.

The native performance track consequently includes a destination-coalesced path from
the start:

```text
source groups a token's selected experts by destination rank
  -> send hidden once per (token, destination)
  -> send destination-local expert IDs and router weights as metadata
  -> destination expands the hidden row to its selected local experts
  -> local expert compute
  -> FP32 destination-local weighted partial sum
  -> return one partial hidden per (token, destination)
  -> source FP32 sum across destination partials
```

This optimized path is not allowed to replace the assignment baseline until Gates A/B
show route, weight, shared-expert, layer-output, and decision equivalence. Both paths
retain raw timings. A systems result that only uses the assignment-granular path is
explicitly labeled a correctness baseline, not a competitive EP implementation.

## Weight loading

1. Resolve checkpoint tensor names without materializing duplicate full-model copies.
2. Load non-routed weights on every DP/EP rank because TP=1.
3. Slice routed expert weights by the fixed ownership interval.
4. Keep shared-expert weights replicated.
5. Record checkpoint revision, config hash, tensor-name mapping, and expert interval in
   metadata.
6. Fail closed on missing, duplicated, or out-of-range expert tensors.

## Correctness gates

Free generation is not the first correctness test. A fixed teacher-forced input and
the stock single-GPU path provide layer-wise reference tensors so the first divergent
operation can be identified.

### Gate 0: reference capture

- Fixed checkpoint revision, prompt, mask schedule, hidden input, and seed.
- Capture router logits, selected expert IDs/weights, MoE output, and layer output from
  the stock TP=1 path.
- Record token-level near-tie margins between the final selected and first unselected
  expert, and between the selected and first unselected group in group-limited routing.

### Gate A: adapter unit layer

- Same hidden input, router output, and weights for reference and EP paths.
- Dispatched assignment count equals `tokens * top_k`.
- Reverse combine restores every origin slot exactly once.
- Global and local expert IDs round-trip exactly.
- Top-8 router weights round-trip without renormalization.
- Shared-expert output is added once, not once per routed assignment.
- Router logits are FP32-identical only when reference and EP paths use the same token
  subset and therefore the same GEMM M dimension. With a different M dimension, report
  relative error and require it to be at most `1e-6`; a different kernel tiling is not
  misclassified as a routing-logic bug.
- Top-8 expert IDs match exactly outside a preregistered near-tie band. Any mismatch
  inside that band is retained with its expert and group boundary margins and is not
  silently discarded.
- Normalized and scaled router-weight relative error is at most `1e-5`.

### Gate B: full layer reference

- Compare single-GPU and EP=4 hidden outputs after every MoE layer.
- Report max absolute, mean absolute, relative RMSE, and decision agreement under BF16.
- Initial diagnostic thresholds are relative RMSE <= `2e-3` for one MoE output and
  <= `5e-3` for accumulated hidden state; maximum relative error is reported rather
  than hidden by the aggregate.
- Diagnose the first divergent layer before running generation.
- Teacher-forced final-logit top-1 agreement must be at least 99.9% of positions.

### Gate C: native generation smoke

```text
one request
block_length=32
steps=32
temperature=0
EP=4, TP=1, PP=1
```

- Fixed seed and prompt.
- Same finalized tokens as the single-GPU reference, or a documented numerically stable
  tolerance policy if BF16 tie-breaking changes a token.
- No token drop, duplication, deadlock, or phase divergence across ranks.
- Record `block_width`, `active_position_count`, `finalized_positions_per_step`, and
  `order_policy` separately.
- Assert `drop_count == 0` before comparing outputs; stable nonzero dropping does not
  satisfy the dropless baseline.
- Repeat with at least eight deterministic request batch compositions. With the same
  per-request input, seed, and mask schedule, top-8 IDs, remasking decisions, and
  finalized tokens must match exactly outside declared near ties.
- Record each token's maximum output deviation across compositions. It must remain below
  the first percentile of the measured decision-margin distribution; this links the
  floating-point tolerance to whether a model decision can change.

Float tolerances are diagnostic. Exact expert-ID agreement and generation decisions
outside the declared near-tie band are the hard semantic gates. A mismatch may only be
classified as numerically ambiguous after its token and group margins are reported; it
is never silently accepted.

The 7B smoke validates all-to-all plumbing, ownership, and slot restoration. It does
not validate LLaDA2.0-mini's shared expert, group-limited routing, scaling factor,
FP32 sigmoid router, dense-first-layer bypass, or 256-expert load distribution.

## DiffusionGemma boundary

DiffusionGemma is not assumed to reuse this decoder-only adapter. Its architecture,
canvas width, sampler, position accounting, and framework path require a separate
design and correctness review. It remains secondary until the LLaDA2.0-mini native
model gate is complete.

## Initial scheduler policy

FIFO is the initial native-model baseline. The current rank-local critical-path policy
is not carried forward as the default because it slowed the controlled H100 matrix.
A coordinated offline replay may be evaluated after Gate C, followed by an online
policy only if replay demonstrates enough savings to cover measured selection and
coordination overhead.

## Native-model metrics

- model-defined block width and order policy
- active positions entering each MoE layer
- finalized positions per denoising step
- layer-wise route IDs and weights
- active experts and tokens per expert/rank
- full rank-pair split vectors
- dispatch, expert, combine, and rank-idle time
- task accuracy, Pass@1/Pass@k where applicable
- entropy and deferral of finalized positions
- useful finalized tokens/s and end-to-end latency
