# RefineServe

> **Research prototype — no native-model speedup claim yet.**

**Decode optimization beyond speculative decoding: native position-parallel
refinement × MoE Expert Parallelism × critical-path scheduling.**

`RefineServe` is an independent research runtime for native position-parallel decode
serving. Its first track tests whether position-parallel refinement can improve
expert-parallel MoE execution without causing excessive expert scattering,
communication, or EP-rank critical-path imbalance. The first milestone is deliberately
a deterministic discrete-event simulator, not a language model implementation.

The project keeps an explicit [claim and related-work ledger](docs/fact_check.md). The
[native runtime scope decision](docs/decisions/0001-native-position-parallel-scope.md)
defines the workload/runtime boundary.

M1 demonstrates that critical-path-aware batch scheduling can reduce simulated MoE EP
makespan under communication-bound, multi-layer position-parallel workloads, while
short execution paths expose scheduler overhead as a limiting factor. M2 determines
whether that result survives measured routing and calibrated GPU/NCCL costs.

The first H100×4 controlled run has established functional EP=4 correctness but not a
calibrated K-scaling claim. Its near-flat 31 ms interval contained control,
synchronization, validation, and metric work. Before another scheduler matrix, M2 now
uses a three-mode gate (`local_copy`, minimal-payload NCCL, real-payload NCCL) and
classifies each regime as `FAIL`, `PASS-UNPOWERED`, or `PASS-POWERED`. See the
[hardware status](docs/m2_hardware_status.md) and
[follow-up measurement plan](docs/m2_followup_measurement_plan.md).

The corrected H100x4 follow-up has completed: Gate 2 is
`PASS-UNPOWERED`, Gate 2B is `PROXY_TIME_UNRESOLVED`, and Gate 3 was not run.
Clock locking was denied, so the result is retained as hardware characterization rather
than a percent-level performance claim. See the
[Gate 2/2B result](docs/m2_gate2_20260812_results.md).

Paid scheduler timing remains blocked until a native LLaDA2 denoising router trajectory
has been screened with the same summed batch/layer critical-load objective and the
native-shape timing opportunity passes its gate. Route-only screening is supporting
calibration; it is not EP timing and cannot substitute for the native EP adapter.
Scheduler authorization is evaluated against the destination-coalesced, full-iteration
path rather than the assignment-granular EP-only correctness path. See the latest
[external-review response](docs/external_review_response_v6.md), the
[accounting changelog](docs/accounting_changelog.md), and the
[Gate 2 internal double-check](docs/gate2_internal_double_check.md).

## Initial research contract

The baseline models one EP group with **8 sequential Transformer blocks**. Every block
contains attention followed by a top-2 MoE layer:

```text
request admission
  -> batch scheduler
  -> 8 x [attention -> route -> EP dispatch -> experts -> EP combine]
  -> finalize token/block
  -> metrics
```

The default logical model has 16 experts across 4 GPUs, hidden size 2048, BF16
activations, EP=4, TP=1, and PP=1. Eight blocks are large enough to expose repeated
routing and network costs while keeping parameter sweeps fast. Layer count is a cost
model parameter; the simulator does not allocate model weights.

The two implemented M0 execution modes are:

- `autoregressive`: one active position per request and model iteration.
- `diffusion`: one 32-token block is refined with active-position schedule
  `32, 24, 16, 12, 8, 4, 2, 1`, then all 32 final tokens are committed.

The M0 `diffusion` mode is synthetic native full refinement. Diffusion is an experiment
mechanism, not the platform's top-level identity. Future native workloads may
include block refinement and masked parallel generation behind a common work-item
interface.

The project uses `active_positions` rather than `K` because top-k expert routing and
parallel position count are different quantities.

## Install and run

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
refineserve \
  --config configs/baseline.yaml \
  --mode diffusion \
  --scheduler previous_route \
  --output results/example
```

Run the first four-arm experiment:

```bash
python experiments/compare_four_arms.py \
  --config configs/baseline.yaml \
  --output results/four_arms
```

The experiment compares AR/FIFO, AR/previous-route locality, diffusion/FIFO, and
diffusion/previous-route locality. Each arm emits CSV and JSON files. The aggregate
directory also contains separate matplotlib figures for throughput, P95 latency,
expert batch size, and communication fraction. `report.md` and `experiment.json`
record the initial gate decisions. To isolate locality from the default 2 ms deadline
fallback, rerun with `configs/locality_relaxed.yaml`; this is a mechanism diagnostic,
not the latency-safe production policy.

Run the M1 rank-critical-path cost profile:

```bash
python experiments/compare_four_arms.py \
  --config configs/m1_critical_path.yaml \
  --output results/m1_critical_path
```

`m1_critical_path.yaml` enables synthetic expert-weight size/HBM bandwidth and computes
layer latency from the slowest EP rank. These parameters are instrumentation defaults,
not calibrated hardware claims. Each run additionally writes `rank_layers.csv` with
per-batch, per-layer, per-GPU expert tokens, unique experts, weight bytes, communication,
layer time, and straggler status.

Run the M1 scheduler mechanism diagnostic:

```bash
python experiments/compare_m1_schedulers.py \
  --config configs/m1_scheduler_diagnostic.yaml \
  --mode diffusion \
  --output results/m1_schedulers
```

This compares FIFO, locality-only, load-balance-only, critical-path-only,
locality-plus-load, joint, routing-oracle, and runtime-oracle policies. The diagnostic
uses a 100 ms wait bound to expose scheduler freedom; it is not a production latency
setting. The scheduler's synthetic per-decision and per-candidate overhead is charged
to makespan and queue delay.

## Simulator layers

1. **Workload:** deterministic arrivals and AR or block-refinement progress.
2. **Scheduling:** FIFO, bounded-wait locality, load, layer-critical-path, and joint
   greedy batch formation.
3. **Routing:** seed-stable synthetic top-k traces with configurable skew and temporal
   stability.
4. **Layer execution:** attention cost, expert grouping, saturating expert throughput,
   and kernel launch cost.
5. **EP/network:** expert placement, dispatch/combine messages, bandwidth, fixed message
   latency, aggregation, and congestion penalty.
6. **Metrics:** finalized versus processed work, latency, expert efficiency, routing
   stability, communication, and work-conservation checks.

M1 records both the historical aggregate communication model and an optional
rank-critical-path model. The latter computes:

```text
layer time = max over EP ranks (
    attention
  + grouped expert time under a token/weight roofline
  + rank-local dispatch/combine time
)
```

This makes unique-expert footprint and the identity/frequency of the straggler GPU
observable before scheduler optimization begins.

The M1 joint score is:

```text
sum(layer EP-rank critical path)
  + deadline penalty
  + KV-rank load fragmentation penalty
  + request-progress fragmentation penalty
  - locality reuse benefit
```

Every candidate addition evaluates all eight modeled layers with either previous-step
routes or oracle current routes. The online estimator reuses per-request route
histograms and is regression-tested against a readable full layer replay. The two
information-oracle policies currently coincide under deterministic costs. A separate
exact global makespan oracle exhaustively searches small validation workloads.

Under three paired seeds, joint scheduling changed makespan by -0.20% in the
compute-bound scenario, -3.97% in the communication-bound scenario, and -0.24% in the
deadline-bound scenario. Locality-only slowed the compute and communication scenarios.
These are `SIMULATION_RESULT` observations, not hardware performance claims.

Candidate-pool restriction alone was insufficient, so the selected M1 policy uses
prewarmed previous-route profiles and a vectorized one-shot proxy. A fabric-aware
controller enables it only for the communication-bound profile and otherwise uses an
exact FIFO fallback. Across three paired seeds at the default eight layers, the policy
changes communication-bound makespan by -5.19%, with 0.312 ms mean selection P95 and
3.44% scheduler wall time relative to modeled makespan. Compute- and deadline-bound
scenarios are unchanged. The 4/8/16/32-layer sweep is non-regressive at every seed;
the four-layer communication case exceeds the total scheduler-wall budget (5.98%
versus 5%) and is recorded as the shallow-model boundary. See
[M1 status](docs/m1_status.md).

Run the selected-policy comparison and layer sensitivity sweep:

```bash
python experiments/compare_online_policy.py \
  --config configs/m1_online.yaml \
  --output results/online_policy
python experiments/sweep_layer_sensitivity.py \
  --config configs/m1_online.yaml \
  --output results/layer_sensitivity
```

Validate an M2 trace bundle and fit bounded timing curves:

```bash
PYTHONPATH=src python experiments/validate_trace_bundle.py \
  --trace traces/example \
  --config configs/m1_online.yaml
PYTHONPATH=src python experiments/fit_trace_calibration.py \
  --trace traces/example \
  --config configs/m1_online.yaml \
  --output results/calibration/example.json
```

Trace replay requires `router.source: trace` and `router.trace_path` in the run config.
Missing lookups, mismatched model dimensions, malformed top-k groups, and out-of-range
calibration queries are errors rather than silent synthetic fallbacks.
The current M2 path can inject the fitted expert-kernel curve into both modeled layer
execution and scheduler estimation. Rank-local `ep_dispatch_combine` curves can also
replace the analytic network formula in both paths. Missing message-count families or
payloads outside the measured range are errors.

Run the partial M2 source-separation matrix:

```bash
PYTHONPATH=src python experiments/compare_m2_matrix.py \
  --config configs/m1_online.yaml \
  --trace traces/example \
  --calibration results/calibration/example.json \
  --use-network-curves \
  --output results/m2_matrix
```

The four cells independently switch synthetic versus trace routes and synthetic versus
measured costs. Without `--use-network-curves`, only expert-kernel timing is measured;
with it, measured-cost cells also use rank-local network curves. The exact scope is
recorded in `experiment.json`.

## Milestone gates

The simulator justifies a calibrated prototype when a regime reproducibly shows:

- at least 2x mean tokens per expert invocation;
- at least 20% fewer expert kernel launches or network messages;
- at least 10% higher finalized-token throughput or 10% lower P95 latency;
- previous-route scheduling captures at least 80% of the oracle scheduler gain;
- results hold for at least three seeds and a named routing/network regime.

Processed positions are never treated as generated tokens. Diffusion may perform more
raw work, so the primary outputs are finalized tokens per second and end-to-end request
latency.

## Growth path

1. Deterministic synthetic position-parallel simulator (M0 complete).
2. Joint critical-path scheduler with unique-expert weight traffic, per-layer EP-rank
   timing, KV-rank load, deadlines, batch traces, and bounded online overhead (M1
   complete for the eight-layer contract).
3. M2.1 native LLaDA2 route collection and request-composition opportunity analysis
   ([gate contract](docs/m2_1_native_route_opportunity.md)).
4. M2.2 native 256-expert/top-8 true-EP timing replay, without the full model.
5. M3 LLaDA2.0-mini true-EP adapter, only if M2.1 and M2.2 pass.
6. M4 native FIFO versus coordinated critical-path scheduler measurement.
7. M5 block-width and denoising-policy ablation with systems and quality metrics.
8. Serving-engine adapters only after the mechanism is validated.

If M4 produces a reproducible native-model result, the conditional upstream order is
SGLang first and vLLM second. This does not authorize framework work before the native
evidence gates. See the [upstream strategy](docs/upstream_strategy.md).

The current milestone intentionally excludes model training, real KV migration,
offloading, TP/PP, learned route prediction, multi-node deployment, and engine
integration. These are calibration or later-system concerns, not prerequisites for
testing the central hypothesis.

The selected scheduler minimizes a cheap proxy for predicted per-layer EP-rank
critical path while treating route similarity as a secondary reuse signal. Expert
cosine similarity alone can reduce messages while worsening the slowest GPU's
completion time, especially when position parallelism already creates large expert
batches. M2 now focuses on whether measured route, kernel, and collective traces retain
the sign of this synthetic result.
