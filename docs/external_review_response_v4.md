# External Review Response v4

검토자가 지적한 synthetic generator 문제와 Gate 3A 보류 판단은 수용했습니다.
다만 제공된 15-cell CSV의 산술은 맞지만, **그 CSV가 사용한 achievability가 실제
planner 목적함수와 달랐습니다.** 이 차이를 먼저 교정하지 않으면 실제 trace를
넣어도 잘못된 gate를 반복하게 됩니다.

## 1. Gate 결정

```text
Gate 2 three-mode timing                         GO
Gate 3 paid scheduler timing                    NO-GO
Gate 3 prerequisite: native router opportunity  PENDING
```

Gate 2는 route workload와 독립적으로 timing harness, fixed/payload 분해, corrected
K-scaling을 측정하므로 실행 가치가 있습니다. Gate 3는 실제 native router trace의
composition opportunity를 확인하기 전에는 실행하지 않습니다.

## 2. Screening 목적함수 교정

실제 coordinated planner의 목적함수는 다음입니다.

```text
sum over every (batch, layer) of max destination-rank load
```

기존 CSV의 achievability는 전체 실행에서 단 하나의 global maximum receive load만
사용했습니다.

```text
(FIFO global max - best global max) / (FIFO global max - global mean)
```

두 값은 같지 않습니다. 한 `(batch, layer)`의 최댓값이 악화되더라도 나머지
`(batch, layer)`의 critical load 합이 더 크게 감소하면 planner 목적함수는
개선됩니다. 실제 artifact에는 `uniform, K=1, seed=41`처럼 global max가
`11 -> 13`으로 악화되지만 summed objective는 4.91% 개선되는 셀도 있습니다.
따라서 옛 global-max 지표로 계산한 `2/15 PASS`는 planner가 만든 composition
효과의 gate로 사용할 수 없습니다.

교정된 정의는 다음과 같습니다.

```text
objective lower bound
  = sum_layer max_destination(total assignments to destination over all batches)

objective achievability
  = (FIFO objective - best objective)
    / (FIFO objective - composition-invariant lower bound)

realized objective reduction
  = (FIFO objective - best objective) / FIFO objective

timing screen
  = measured accessible fraction(K)
    x p25(realized objective reduction(K, routing))
```

lower bound는 batch composition으로 바꿀 수 없는 destination 총량에서 계산합니다.
`objective achievability`는 planner가 이론적 여지 중 얼마를 찾았는지 진단하고,
실제 timing gate에는 전체 critical-load 중 실제로 줄인 몫인
`realized objective reduction`을 사용합니다.

## 3. 교정 후 synthetic 결과

5 seeds, 64 planner restarts에서 주요 p25는 다음과 같습니다.

| K | routing | objective achievability p25 | realized reduction p25 |
|---:|---|---:|---:|
| 1 | request_correlated | 1.000 | 9.239% |
| 16 | request_correlated | 0.951 | 6.085% |
| 64 | request_correlated | 0.978 | 5.305% |
| 16 | uniform | 0.906 | 1.319% |
| 64 | uniform | 0.883 | 1.070% |
| 16 | temporally_unstable | 0.969 | 1.410% |
| 64 | temporally_unstable | 0.885 | 1.080% |
| 1/16/64 | strong_skew | 0.000 | 0.000% |

검토자의 roofline accessible 추정치 2.5%/20.3%/31.1%를 참고값으로 곱하면
4% gate를 통과하는 synthetic cell은 없습니다. 가장 큰
`request_correlated, K=64`도 약 1.65%입니다. 최종 판정은 roofline이 아니라
Gate 2 실측 accessible fraction으로 내리지만, 현재 synthetic 정보만으로 유료
Gate 3A를 승인하지 않는다는 결론은 더 강해졌습니다.

## 4. Synthetic generator 수정

검토자의 generator 감사는 정확했습니다.

- 옛 `uniform`은 결정적 round-robin이었습니다. 이를
  `balanced_round_robin`으로 명시적으로 이름 변경했습니다.
- 현재 `uniform`은 seed를 사용하는 확률 표집입니다.
- 옛 `request_correlated`는 seed/K에 무관한 완전 상보 workload였습니다.
- 현재 구현은 request/layer별 선호 EP rank가 seed에 의존하고,
  `request_correlation_strength`를 0~1 연속값으로 받습니다.
- 실제 route 멀티셋의 composition-invariant lower bound를 계산하므로
  `strong_skew`의 0이 물리적 기회 부재인지 planner 실패인지 구분할 수 있습니다.

교정 후 planner는 기회가 있는 대부분의 셀에서 lower-bound opportunity의
약 88~100%를 찾습니다. 반면 strong skew에서는 FIFO objective가 lower bound와
같아 실제로 composition 여지가 없습니다.

## 5. 실제 LLaDA2 route prerequisite

실제 trace용 두 도구를 추가했습니다.

```text
hardware/collect_llada2_router_trace.py
hardware/screen_measured_router_trace.py
```

수집기는 stock `inclusionAI/LLaDA2.0-mini`의 immutable revision을 요구하고,
공식 block-diagonal causal mask를 재현해 한 개의 완전 masked initial block을
관찰합니다. block width 1/16/32/64, 5 prompt segments, 모든 sparse layer의 top-k
expert ID를 보존합니다.

screening은 동일한 summed critical-load objective와 composition-invariant lower
bound를 적용합니다. 기본 EP placement는 planned contiguous 256-expert/4-rank
ownership이며, 별도 mapping JSON으로 교체할 수 있습니다.

이 trace의 증거 등급은 제한적입니다.

```text
가능:
  실제 initial native block route의 imbalance/composition opportunity 판별

불가능:
  EP timing 주장
  later denoising iteration 대표성 주장
  quality/finalization 주장
  toy top-2 timing accessible fraction과의 직접 곱셈
```

LLaDA2는 256 experts/top-8/19 sparse layers이고 controlled Gate 2는
16 experts/top-2/8 layers입니다. 따라서 actual-route reduction과 toy timing
accessible fraction을 같은 shape처럼 곱하지 않습니다. 실제 trace가 scheduling
axis의 생존을 보여도 native shape의 timing-accessibility gate는 별도로 필요합니다.

## 6. 비용 경계

현재 로컬 호스트에는 CUDA accelerator가 없고 checkpoint 전체를 저장할 충분한
여유 공간도 없습니다. 따라서 실제 trace 수집은 이 환경에서 무료 CPU 작업이
아닙니다. 가능한 경로는 다음 두 가지이며 아직 어느 것도 시작하지 않았습니다.

1. 저렴한 단일-GPU 짧은 세션에서 route bundle만 먼저 수집
2. 승인된 H100x4 Gate 2 세션에서 Gate 2 완료 후 한 rank로 route 수집

두 번째는 모델 다운로드 시간 동안 4-GPU 비용을 지불하므로, 비용만 보면 첫 번째가
유리합니다. 어느 경우든 Gate 3 scheduler matrix는 actual-route screen 전에는
실행하지 않습니다.

## 7. 검증 상태

- Ruff 통과
- pytest 59 passed
- measured-route collector 정적 검증 통과
- measured-route screening fixture 통과
- 목적함수 정렬 synthetic artifact 생성 완료
- 실제 LLaDA2 route bundle 미수집
- H100 Gate 2 미실행
- 활성 GPU rental 없음
