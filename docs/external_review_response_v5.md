# External Review Response v5

검토 결론을 수용합니다.

```text
Gate 2 timing/accessibility                GO
Gate 2B objective -> measured-time check  GO, preregistered
Gate 3 scheduler timing                   NO-GO
active GPU rental                         none
```

이번 수정의 핵심은 **synthetic load proxy로 셀을 모두 제거해 놓고 그 proxy와 실제
시간의 관계를 영구히 검증하지 못하는 순환을 끊는 것**입니다.

## 1. Constructed proxy validation cell

다음 도구를 추가했습니다.

```text
hardware/benchmark_proxy_validation_ep4.py
hardware/analyze_proxy_validation_ep4.py
```

동일한 assignment 수, expert shape, layer 수, collective 수를 유지하면서 composition만
바꾸는 두 plan을 사전 구성했습니다.

```text
FIFO objective      = baseline
balanced objective  = exactly 33.333% lower
```

모든 destination은 두 plan에서 non-empty이고, 두 batch 전체의 destination별 총
assignment 수는 같습니다. 측정은 실제 hidden dispatch, expert-ID dispatch, local
expert compute, hidden combine을 통과합니다. 결과는 다음 세 상태로만 해석합니다.

```text
PROXY_TIME_ALIGNED
  load-space 감소와 measured time 감소의 방향이 일치

PROXY_TIME_DISCONFIRMED
  load proxy를 scheduler opportunity gate로 사용 금지

PROXY_TIME_UNRESOLVED
  load-only zero-cell 결과를 timing opportunity 부재로 승격 금지
```

따라서 synthetic/native trace에서 통과 셀이 0개여도 proxy 결과 없이
`no timing opportunity`를 주장하지 않습니다.

## 2. Toy shape와 native mini shape 분리

검토자의 정성적 결론을 수용합니다. controlled toy는 top-2/intermediate 8192이고,
LLaDA2-mini sparse path는 top-8/intermediate 512이므로 native shape가 훨씬 더
communication-accessible할 수 있습니다. 따라서 toy 0-cell 결과는 native model로
전이하지 않습니다.

저장소의 명시적 회계 가정(500 TFLOPS/rank, 400 GB/s, collective fixed 8 us,
self-traffic 25% 제외)을 적용한 잠정값은 다음과 같습니다.

| shape | K | provisional accessible fraction | 4%에 필요한 realized reduction |
|---|---:|---:|---:|
| toy | 16 | 13.65% | 29.30% |
| toy | 64 | 19.85% | 20.15% |
| LLaDA2-mini | 16 | 50.18% | 7.97% |
| LLaDA2-mini | 32 | 62.05% | 6.45% |

이는 roofline sensitivity이며 측정값이 아닙니다. native trace 판정에는 toy Gate 2
accessible fraction을 곱하지 않습니다. native adapter가 실행되면 동일 shape의
실측값으로 교체합니다.

## 3. Batch-count 범위

검토자의 방향은 맞습니다. batch 수가 늘수록 FIFO composition의 물리적 여지가
커질 수 있으므로 batch=2 negative를 일반화하지 않습니다.

`hardware/audit_batch_count_sensitivity.py`는 이제 batch 2/4/8/16에 대해 다음을
분리 기록합니다.

```text
composition-invariant lower bound
FIFO objective
best-found objective
physical opportunity fraction
realized objective reduction fraction
objective achievability
```

planner는 controlled 8-request/2-batch에서 exact source choices를 쓰고, 3개 이상
batch 또는 더 큰 request pool에서는 cardinality를 보존하는 deterministic swap
coordinate descent를 사용합니다. 어느 경우에도 `best-found`일 뿐 최적해라고
부르지 않습니다. 결과의 범위는 measured batch count와 candidate-pool size로
제한합니다.

현재 8 requests/rank, 16 restarts 감사에서 all-condition median은 다음과 같습니다.

| batch count | physical opportunity | best-found realized reduction | achievability |
|---:|---:|---:|---:|
| 2 | 1.41% | 1.27% | 94.87% |
| 4 | 4.68% | 3.67% | 66.41% |
| 8 | 9.32% | 4.60% | 49.18% |

batch 수가 늘면 물리적 여지는 증가하지만 현재 local search가 회수하는 비율은
감소합니다. 따라서 `more batches create more opportunity`와 `the current planner
captures that opportunity`를 분리합니다.

16 requests/rank, 8 restarts의 확장 감사에서도 같은 방향입니다.

```text
batch count                 2      4      8      16
physical opportunity     0.92%  2.33%  6.51%  11.54%
best-found reduction     0.89%  1.91%  4.25%   4.93%
achievability            99.1%  81.4%  61.0%  43.9%
```

즉 검토자가 요구한 4/8/16-batch 축은 실제 planner objective로 계산됐고,
batch=2 negative는 pool 전체에 일반화할 수 없다는 결론이 유지됩니다.

## 4. Physical absence와 planner miss 구분

synthetic 및 measured screening CSV에 다음 원시 열을 추가했습니다.

```text
objective_lower_bound
fifo_objective
best_found_objective
objective_opportunity
objective_opportunity_fraction
objective_physically_zero
```

따라서 결과 문구도 사전 분리합니다.

```text
FIFO objective == lower bound
  physical composition opportunity is zero under this objective

FIFO objective > lower bound and best-found == FIFO
  the current planner found no reduction; physical absence is not claimed
```

## 5. Native LLaDA2-mini route trace

trace 대상은 `LLaDA2.0-mini`입니다. 7B 모델은 EP 배관 smoke 용도일 뿐 screening
대표 모델로 쓰지 않습니다.

collector는 이제 initial block만 관찰하지 않습니다.

```text
initial width ablation: K = 1,16,32,64
native denoising: block width 32, up to 32 stock-semantics steps
captured: request ID, denoising step, all 19 sparse-layer top-8 IDs,
          compute width, masked positions before/after, finalized progress
```

행 단위 수천만-row CSV 대신 `routes_dense.npz`와
`route_observations.csv` manifest를 사용합니다. screening은 denoising step별로
imbalance, invariant lower bound, best-found reduction, same-rank collision을
계산합니다. 32-request fixed-pool 조건을 만족하지 않는 후기 step은 조용히 버리지
않고 별도 coverage CSV에 남기며, 32-request 결과로 승격하지 않습니다.

## 6. Same-destination coalescing

검토자의 지적을 수용했습니다. assignment-granular path는 correctness baseline일
뿐 경쟁력 있는 native EP baseline이 아닙니다. native adapter 설계에 다음 optimized
path를 명시했습니다.

```text
one hidden per (token, destination rank)
-> destination-local expert expansion
-> FP32 weighted partial sum per destination
-> one partial result per (token, destination rank)
```

uniform 선택 가정에서 assignment-to-destination 중복 예상치는 toy 약 10%,
LLaDA2-mini 약 54.8%입니다. optimized path는 stock semantics와 G0-G6 correctness를
통과하기 전에는 timing baseline으로 사용하지 않습니다.

## 7. NCCL provenance

`torch.distributed.all_to_all_single`이 grouped point-to-point로 내려갈 수 있다는
지적을 반영했습니다. `NCCL_ALGO`/`NCCL_PROTO`는 존재하면 기록만 하며 실행 경로를
통제했다는 증거로 쓰지 않습니다. `NCCL_DEBUG=INFO` 로그와 profiler trace가 실제
경로의 authoritative provenance입니다.

## 8. 현재 실행 결정

```text
지금 실행 가능:
  CPU screening/audit
  Gate 2 및 Gate 2B의 사전등록 설계 검증

다음 유료 실행에서 허용:
  Gate 2 three-mode timing
  constructed proxy validation

아직 금지:
  Gate 3 scheduler matrix
```

실제 LLaDA2-mini router trace와 exact-native-shape accessibility가 확보되기 전에는
native scheduling speedup을 주장하지 않습니다. 현재 GPU rental은 없습니다.
