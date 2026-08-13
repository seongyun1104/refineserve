# M2 trace and calibration contract

Status: implemented foundation; calibration continuation closed after M2.1
Date: 2026-08-04

## Purpose

M2 replaces synthetic routing and timing assumptions with measured native
position-parallel traces. A calibrated result is valid only when the raw trace,
provenance, units, validation report, fitted model, and replay output remain linked.

## Bundle layout

```text
trace_bundle/
  metadata.json
  routes.csv
  route_priors.csv
  expert_kernel_samples.csv      # optional until kernel calibration
  network_samples.csv            # optional until collective calibration
```

`metadata.json` is the bundle authority:

```json
{
  "schema_version": 2,
  "trace_kind": "native_position_parallel",
  "created_at_utc": "2026-08-04T00:00:00Z",
  "source": "synthetic_fixture_or_hardware_run_id",
  "model_identifier": "model-name",
  "model_revision": "immutable-revision",
  "random_seed": 17,
  "model": {
    "num_layers": 1,
    "num_experts": 8,
    "top_k": 2,
    "num_gpus": 4,
    "hidden_size": 2048,
    "bytes_per_element": 2
  },
  "expert_to_rank_mapping": [
    [0, 1, 2, 3, 0, 1, 2, 3]
  ],
  "measurement_environment": {
    "gpu_model": "GPU model",
    "gpu_count": 4,
    "topology": "topology identifier",
    "node_scope": "single_node",
    "cuda_version": "version",
    "nccl_version": "version",
    "pytorch_version": "version",
    "kernel_backend": "Triton/CUTLASS/other",
    "dtype": "bf16",
    "intermediate_size": 8192,
    "concurrent_streams": 1,
    "warmup_count": 20,
    "measurement_iterations": 100
  },
  "units": {"latency": "ms", "size": "bytes"}
}
```

The source must identify a reproducible run or fixture. Device, software, topology,
execution mode, warmup, and repetition metadata are required in schema v2; placeholder
values keep synthetic fixtures explicit but cannot support `CALIBRATED_RESULT` status.

## Route tables

`routes.csv` contains exactly one row per top-k slot:

```text
request_id,iteration,layer_id,position_id,route_slot,expert_id,routing_weight,batch_size,active_position_count,context_length
```

The composite key through `route_slot` is unique. Every work item/layer group must
contain exactly `top_k` slots numbered `0..top_k-1`, and its expert IDs must be unique.
IDs are zero-based and must fit the dimensions in metadata. Batch size, active-position
count, context length, and routing weights are captured at observation time rather than
reconstructed later.

`expert_to_rank_mapping` contains one row per layer and one rank ID per expert. Replay
uses this mapping in layer execution, dispatch accounting, and scheduler estimation;
it never substitutes the synthetic round-robin mapping.

`route_priors.csv` contains the information available before a request's first native
iteration:

```text
request_id,layer_id,route_slot,expert_id
```

It follows the same top-k rules. Keeping priors separate prevents the first-iteration
scheduler from silently reading the current route.

Replay is strict: a missing route or prior is an error. There is no synthetic fallback,
because fallback would mix evidence classes inside one run.

### Native LLaDA2 dense trajectory extension

M2.1 uses compressed dense arrays because a stock denoising forward routes the entire
clean-prefix/current-window tensor at every sparse layer. Its observation manifest adds:

```text
workload_class
block_id
denoise_step
block_width
model_forward_positions
masked_positions_before_step
masked_positions_after_step
finalized_positions_this_step
```

`position_roles_dense.npz` labels each routed position immediately before the forward
as `prefix`, `current_block_finalized`, or `current_block_masked`.
`route_weights_dense.npz` stores selected weights after sigmoid, selected-score
normalization, and routed scaling. IDs, weights, and roles share the observation
`array_key` and are covered by the bundle checksum.

The dense extension does not redefine `active_position_count`. Analyses must report
model-forward routed positions, active masks, and newly finalized positions separately.
See [M2.1 native route opportunity gate](m2_1_native_route_opportunity.md).

## Timing samples

`expert_kernel_samples.csv`:

```text
sample_id,gpu_id,expert_id,token_count,latency_ms,warmup,repetition
```

`network_samples.csv`:

```text
sample_id,collective,active_ranks,message_count,transferred_bytes,latency_ms,warmup,repetition
```

For runtime injection, `collective` is `ep_dispatch_combine` and each sample is a
rank-local endpoint duration for the combined dispatch/combine path. Curves are grouped
by `(collective, active_ranks, message_count)` and interpolate only payload bytes.

Raw rows are immutable inputs. Derived medians, confidence intervals, monotone fitted
curves, interpolation policy, and out-of-range behavior belong in a separate generated
calibration artifact. M2 must report fit error and may not silently extrapolate beyond
the sampled range.

Range failures are written to `rejections.csv` with the input type, observed range,
calibrated range, and maximum overflow. `experiment.json` records the rejected
experiment count. Call-level miss counts and ratios remain required before M2 closure;
the current strict replay stops at the first miss in each rejected cell.

Current implementation status:

- route/prior bundle validation and strict replay: implemented;
- raw expert/network sample validation and bundle checksum: implemented;
- warmup filtering, percentile summary, monotone bounded curves: implemented;
- expert-kernel curve injection into both layer execution and scheduler estimation:
  implemented;
- rank-local network-curve injection into layer execution and scheduler estimation:
  implemented;
- partial or full 2x2 route/cost comparison runner: implemented;
- native LLaDA2 route collection and route-opportunity analysis: complete;
- native-shape kernel/network calibration and execution of the full 2x2 matrix: not
  pursued after the M2.1 negative magnitude result.

Two calibration-model limits remain explicit. The implemented expert curve is
one-dimensional in token count and must not mix measurements with different active
expert counts or grouped-execution modes. The implemented network family keys on rank
count and message count, but does not yet model non-empty peers and maximum peer bytes
as independent inputs. Hardware collection must preserve those raw fields so M2 can
upgrade to:

```text
expert_kernel_ms = f(tokens_per_expert, active_expert_count)
network_ms = f(total_bytes, non_empty_peers, max_peer_bytes, rank_count)
```

Until those inputs are represented or held constant by the measurement protocol, a
replay remains partially calibrated rather than a final `CALIBRATED_RESULT`.

## Validation gates

```text
schema and units recognized                              required
model dimensions equal replay config                    required
route/prior composite keys unique                       required
every route group has exactly top_k unique experts      required
all route IDs in range                                  required
all latency and size samples non-negative               required
every replay lookup covered                             required
raw bundle checksum stored in replay metadata           required
```

## Comparison matrix

M2 keeps source changes separable:

```text
synthetic routes + synthetic costs
measured routes  + synthetic costs
synthetic routes + measured costs
measured routes  + measured costs
```

FIFO and the selected adaptive online policy run in every valid cell. This separates
route-distribution effects from kernel/network calibration effects and prevents a
single aggregate result from hiding the source of a sign change.

## Online information boundary

Measured inputs do not grant future knowledge. The selected online policy may use:

```text
previous observed route
current ready queue and waiting time
known layer-local expert placement
calibrated expected kernel/network cost
```

Actual next routes belong only to the route oracle. Realized congestion, sampled
kernel tail, final straggler rank, and future arrivals belong only to the runtime or
offline oracle. Measured-route replay must preserve this boundary.

## M2 completion gates

```text
measured routes + measured costs improve makespan vs FIFO     required
>= 3 route seeds or independent trace segments                required
scheduler wall time included in makespan                       required
P95 and P99 non-regression                                    required
kernel/network/layer-critical-path estimator error reported   required
calibration miss count and ratio reported                     required
online gap to route/runtime oracle reported                   required
shallow-path bypass prevents the M1 boundary regression       required
```

The future bypass condition is:

```text
predicted scheduling gain
  <= estimated selection cost * safety factor
=> use FIFO
```

Its thresholds must be learned from calibration traces or held-out segments; M1's
synthetic result alone is not sufficient to tune them.
