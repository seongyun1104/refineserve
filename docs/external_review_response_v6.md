# External Review Response v6

Date: **2026-08-09**

검토 결과와 판정을 수용합니다.

```text
Gate 2 three-mode timing               GO
Gate 2B objective-to-time validation   GO
Gate 3 scheduler matrix                NO-GO
active GPU rental                      none
```

이번 반영에서 중요한 것은 회계 수치를 맞추는 데 그치지 않고, 어떤 분모와 통신
경로가 실제 native scheduler 판정을 통제하는지 사전등록한 것입니다.

## 1. Toy matmul count와 회계 변경 이력

toy expert와 LLaDA2-mini expert 모두 SwiGLU입니다.

```text
gate projection + up projection + down projection
= 3 matmuls
= 6 * tokens * hidden * intermediate FLOPs
```

v1 shape artifact 생성 코드도 이미 toy와 mini 양쪽에 factor 6을 사용했습니다. 이전
문서가 matmul count를 metadata에 기록하지 않아 factor 4로 독립 재계산할 여지를
남긴 것이 결함이었습니다. 즉 이번 1.5배 차이는 toy 공식의 조용한 변경이 아니라
기존 provenance 누락입니다. 이를 `docs/accounting_changelog.md`에 소급 기록했고,
모든 새 artifact metadata에 matmul count, FLOPs/MAC, self-traffic 처리, 분모를
저장합니다.

현재 controlled-toy 값은 다음과 같습니다.

| path / denominator | K=16 accessible | K=64 accessible |
|---|---:|---:|
| assignment-granular / EP-only | 13.65% | 19.85% |
| destination-coalesced / EP-only | 12.46% | 18.24% |

## 2. Native mini 판정 분모 사전등록

검토대로 하나의 mini accessibility 숫자로 판정하면 안 됩니다. 다음 4개 조합을
모두 출력하도록 바꿨습니다.

| path / denominator | K=16 | K=32 | K=64 |
|---|---:|---:|---:|
| assignment-granular / EP-only | 50.18% | 62.05% | 70.38% |
| destination-coalesced / EP-only | 31.31% | 42.53% | 51.81% |
| assignment-granular / full iteration | 61.31% | 62.26% | 63.48% |
| **destination-coalesced / full iteration** | **41.77%** | **42.75%** | **44.02%** |

역할은 다음처럼 고정합니다.

```text
assignment-granular rows
  correctness baseline sensitivity only

destination-coalesced / EP-only
  EP mechanism diagnostic

destination-coalesced / full iteration
  native scheduler authorization denominator
```

full-iteration sensitivity에는 stock `use_cache=False`에 따른 `prefix + current
block` 전체 재계산을 반영하고, 19 sparse layers, shared experts, 20 attention
layers, first dense SwiGLU layer, FP32 router projection, LM head를 포함했습니다.
norm, softmax, activation, packing 등 비-matmul 비용은 아직 제외되어 있으므로 이는
실측이 아닌 roofline sensitivity입니다.

이 사전등록에 따르면 K=64의 primary provisional accessibility는 44.02%이고, 4%
screen을 위해 필요한 realized objective reduction은 9.09%입니다. synthetic
request-correlated p25 5.305%는 assignment/EP-only 문턱 5.68%의 93%가 아니라,
primary 문턱의 약 58%입니다. 따라서 실제 trace가 여전히 결정 변수인 것은 맞지만,
현재 증거를 `native shape에서 거의 통과`로 표현하지 않습니다.

## 3. Restart-count 비교의 동일 조건 감사

이전의 43.93%와 v5의 88--97%는 restart 수만 다른 실험이 아니었습니다.

```text
43.93%:
  16 requests/rank, batch count 16, 8 restarts,
  all-condition median

88--97%:
  8 requests/rank, batch count 2, K=64, 64 restarts,
  routing-condition-specific statistic
```

따라서 두 값의 차이를 restart 8 -> 64 효과로 해석할 수 없습니다. 이를 해결하기
위해 같은 route generator, request pool, K, planner seed 계약을 고정하고 restart
`8/16/32/64/128`만 바꾸는 감사를 추가했습니다. v5 screening과 동일한
batch=2/restart=64 지점은 원 CSV와 정확히 일치합니다.

세 routing condition을 합친 결과는 다음과 같습니다.

| batch count | achievability p25/median @8 | @64 | @128 |
|---:|---:|---:|---:|
| 2 | 87.27% / 91.96% | 89.82% / 95.96% | 93.92% / 96.08% |
| 4 | 61.71% / 66.55% | 65.13% / 69.89% | 66.62% / 72.58% |
| 8 | 44.78% / 49.54% | 49.40% / 53.58% | 49.40% / 53.76% |

restart 증가는 일부 개선을 만들지만 44%에서 88%로 두 배가 되는 원인은 아닙니다.
주된 차이는 candidate-pool/batch 구조입니다. batch 수가 커질수록 physical
opportunity는 커지지만 현재 local search의 포착률은 낮아진다는 기존 결론이
유지됩니다. artifact는 restart별 best-so-far curve를 모두 보존합니다.

## 4. Gate 2B를 두 dose와 전달률로 변경

방향 일치만으로 `ALIGNED`를 판정하는 약점을 수용했습니다. 이후 내부 구현
double-check에서 balanced-route Gate 2 accessibility를 그대로 가져오는 추가
교란을 발견했습니다. constructed cell은 이제 FIFO에 대해 local/minimal/real
transport를 먼저 측정하고, full-payload 경로에서 두 objective dose를 실행합니다.

```text
FIFO local copy     objective dose 0%, no NCCL
FIFO minimal NCCL   objective dose 0%, three minimal collectives/layer
FIFO real NCCL      objective dose 0%, three full collectives/layer
low dose real NCCL  objective reduction exactly 1/12 = 8.333%
high dose real NCCL objective reduction exactly 1/3  = 33.333%
```

모든 composition arm에서 hidden, int32 expert-ID, assignment 수, collective 수,
각 source의 local batch assignment 수가 동일합니다. expert-ID payload도 두 dose에서
동일하며, 바뀌는 것은 batch별 destination alignment뿐입니다. Low-dose의 source별
plan이 다르므로 send/receive split은 전체 global plan에서 사전 계산합니다.

분석기는 constructed FIFO의 local/minimal/real 차분에서 route-shape-matched
accessibility를 계산하고 dose별로 다음 값을 bootstrap CI와 함께 냅니다. Gate 2의
balanced-route 값은 cross-check로만 남깁니다.

```text
transmission = measured latency reduction
               / (measured accessible fraction * objective reduction)
```

두 non-zero dose의 through-origin slope도 함께 계산합니다. 이후 screening은
`ALIGNED` 방향만 사용하지 않고 transmission 점추정과 CI를 환산 계수로 사용합니다.
두 dose가 선형성을 지지하지 않으면 단일 비례 환산을 금지합니다.

## 5. Native trace의 K 의미 정정

검토의 M3에서 제기된 `confidence에 따라 compute width가 변한다`는 전제는 stock
LLaDA2.0-mini 구현에는 적용되지 않습니다. 공식 생성 코드를 다시 확인한 결과:

```text
native block width = fixed 32
model input         = full clean prefix + current block
cache               = use_cache=False
confidence/remask   = remaining masked positions와 finalized progress를 변경
                      model-forward width 자체는 변경하지 않음
```

따라서 trace schema는 다음을 독립 기록합니다.

```text
block_width
active_position_count
model_forward_positions
remaining_masked_positions
finalized_positions_per_step
position_width_source
```

`position_width_source`는 controlled initial-width ablation과 native fixed-block
trajectory를 구분합니다. screening은 full forward에 실제로 들어간 모든 route를
사용하며, masked-position 수를 compute width로 재해석하지 않습니다.

## 6. 결과 artifact와 실행 결정

추가된 핵심 artifact는 다음과 같습니다.

```text
results/hardware/h100_ep4_20260805/shape-accessibility-v2/
results/hardware/h100_ep4_20260805/planner-restart-sensitivity-v3/
```

결론은 변경하지 않습니다.

```text
실행 허용:
  Gate 2 three-mode timing
  Gate 2B two-dose constructed proxy validation

실행 금지:
  Gate 3 scheduler matrix

Gate 3 재검토 필수 입력:
  stock LLaDA2.0-mini step-wise router trace
  exact native destination-coalesced/full-iteration accessibility
  Gate 2B transmission estimate and CI
```

현재 GPU rental은 시작하지 않았습니다.
