# RefineServe

> **Completed negative-result study — primary hypothesis not supported.**

**Decode optimization beyond speculative decoding: native position-parallel
refinement × MoE Expert Parallelism × critical-path scheduling.**

`RefineServe` investigated whether request-composition scheduling could reduce MoE
Expert Parallel critical paths in native position-parallel diffusion decoding.
Measurements on stock `inclusionAI/LLaDA2.0-mini` showed high temporal route
persistence but weak inter-request route differentiation. The best-found scheduling
headroom remained below 0.5% in every measured cell, below the existing 2% hardware
measurement target. The project is therefore closed without a native-model speedup or
upstream integration claim.

This is a workload-scoped negative result. It does not claim that request-level
scheduling is ineffective for every model or serving regime. It establishes that the
measured LLaDA2.0-mini denoising workloads do not justify further investment in this
project's request-composition control plane. See the
[project closure](docs/project_closure.md) and the detailed
[M2.1 result](docs/m2_1_20260813_results.md).

The project keeps an explicit [claim and related-work ledger](docs/fact_check.md). The
[native runtime scope decision](docs/decisions/0001-native-position-parallel-scope.md)
defines the workload/runtime boundary.

M1 showed that critical-path-aware batch scheduling can reduce simulated MoE EP
makespan under communication-bound, multi-layer position-parallel workloads. M2 then
showed that the controlled H100 toy path was unpowered for a percent-level scheduler
claim, and M2.1 found negligible request-composition headroom in native LLaDA2 routes.

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

The native LLaDA2 denoising route screen is complete. Route-only screening is
supporting evidence rather than EP timing, but its sub-0.5% best-found magnitude does
not justify M2.2 paid timing, a full true-EP adapter, or serving-engine integration.
The earlier measurement contracts and review responses remain in the repository as an
audit trail.

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

## Final milestone status

| Milestone | Status | Disposition |
|---|---|---|
| M0 synthetic feasibility | Complete | Functional position-parallel simulation established. |
| M1 critical-path scheduler | Complete | Simulated benefit and scheduler-overhead boundary measured. |
| M2 H100 characterization | Complete | EP=4 correctness passed; Gate 2 was `PASS-UNPOWERED` and Gate 2B was `PROXY_TIME_UNRESOLVED`. |
| M2.1 native route opportunity | Complete — negative | Best-found request-composition headroom was below 0.5% in every measured LLaDA2 cell. |
| M2.2 native-shape timing replay | Deferred indefinitely | Existing evidence does not justify another paid scheduler-timing run. |
| M3 full LLaDA2 true-EP adapter | Deferred indefinitely | Its prerequisite gates did not pass. |
| M4/M5 native scheduling and policy ablations | Not pursued | They would not test a supported continuation of the primary hypothesis. |
| SGLang/vLLM upstream work | Not pursued | No RFC or implementation claim is warranted. |

Expert placement, active-width control, and position-level scheduling may remain useful
research questions, but they are different control variables. They are not presented
as continuations or successes of RefineServe and require separately scoped projects.

The completed study excluded model training, real KV migration, offloading, TP/PP,
learned route prediction, multi-node deployment, and engine integration. None of these
was needed to resolve the primary request-composition hypothesis.

The selected scheduler minimized a cheap proxy for predicted per-layer EP-rank
critical path while treating route similarity as a secondary reuse signal. The native
trace established that route prediction was feasible, but the request-composition
freedom available to exploit that prediction was too small to support continued work.
