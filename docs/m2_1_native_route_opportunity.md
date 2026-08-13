# M2.1 native route opportunity gate

Status: complete; measured 2026-08-13

Result: [M2.1 native LLaDA2 route-opportunity result](m2_1_20260813_results.md).
The formal result is placement/prediction dependent, but best-found objective reduction
is below 0.5% in every measured cell. Paid M2.2 scheduler timing is not authorized by
the current evidence.

## Question

M2.1 asks only whether stock `inclusionAI/LLaDA2.0-mini` denoising routes contain
request-level batch-composition opportunity. It does not ask whether a scheduler is
faster on hardware.

The milestone is route-only supporting calibration. It cannot authorize a native-model
speedup claim or replace measured EP dispatch, expert compute, and combine.

## Native position semantics

The collector keeps these quantities separate:

```text
block_width
routed_positions
masked_positions_before_step
finalized_positions_this_step
position_role
```

Stock generation routes the clean prefix, prior finalized blocks, and the entire current
block on every unfinished denoising step. Token selection applies only to the current
block. Therefore routed positions, masked positions, and newly finalized positions are
not interchangeable and are never collapsed into one `K`.

Position roles are captured immediately before the routed forward:

```text
0  prefix
1  current_block_finalized
2  current_block_masked
```

## Initial trace set

```text
model                   inclusionAI/LLaDA2.0-mini
checkpoint revision     immutable commit
workloads                reasoning, code, general
requests per segment     32
segments per workload    seeds 17, 29, 41
block width              32
denoising steps          32
generation length        128
temperature              0
threshold                0.95
```

Prompts are stored as hashes with workload and segment labels. The collector does not
store prompt text in the public trace bundle.

## Required analysis

The route analysis records:

1. pairwise spatial Jaccard similarity between position expert sets;
2. per-position route persistence between adjacent denoising steps;
3. request-signature persistence within a request;
4. request-signature similarity between requests;
5. projected rank load under contiguous and round-robin expert ownership;
6. FIFO, current-route best-found, and previous-step-plan objectives.

It also produces `native_denoising_progress.png` with denoising iteration on the x-axis
and masked positions, unique experts, maximum projected rank load, route persistence,
and best-found batching headroom as separate aligned panels. Incomplete late-step pools
remain in correlation and projection results; only fixed-pool scheduler comparisons
require all 32 requests.

The planner objective is fixed:

```text
sum over batch and layer of max destination-rank assignment load
```

The current-route plan remains labeled `best-found`, not an exact global oracle. The
previous-step plan is constructed only from the preceding observed route and evaluated
against the current route.

## Decision

The existing 80% estimator gate is reused:

```text
best-found headroom is zero
  -> NO_COMPOSITION_HEADROOM

headroom is positive and previous-route p25 captures >= 80% of best-found gain
  -> PREVIOUS_ROUTE_PASS

headroom is positive but previous-route capture is below 80%
  -> PREDICTION_GAP
```

The decision is reported independently for contiguous and round-robin placement. A
placement-dependent outcome is not collapsed into a universal pass.

Even `ROUTE_SPACE_PASS` authorizes only M2.2 native-shape accessibility measurement.
It does not authorize M3, which requires M2.2 to show recoverable timing above
measurement resolution and scheduler overhead.

## Commands

```bash
python hardware/collect_llada2_router_trace.py \
  --model inclusionAI/LLaDA2.0-mini \
  --revision <immutable-revision> \
  --workloads reasoning code general \
  --seeds 17 29 41 \
  --generation-length 128 \
  --output results/m2_1/llada2-router

python hardware/analyze_native_route_opportunity.py \
  results/m2_1/llada2-router \
  --output results/m2_1/native-opportunity
```

The optional controlled initial-width trace is enabled separately with
`--include-initial-width-ablation`; it is not part of the default paid M2.1 run.
