# External Review Response v2

검토 감사합니다. 최종 판정은 **GO WITH BLOCKERS**로 수용하되, 실제 유료
실행은 `PASS-POWERED` gate 전까지 보류합니다. 현재 GPU 인스턴스는 활성화되어
있지 않습니다.

## 1. 핵심 판정 수용

가장 중요한 지적은 기존 gate가 harness cleanliness만 확인하고 scheduler가
접근 가능한 시간 몫과 검정력을 판별하지 못했다는 점입니다. 이를 다음과 같이
바꿨습니다.

```text
local_copy
nccl_minimal: 동일 compute + rank-pair당 1 element인 실제 collective
nccl_real:    실제 full-payload EP data plane
```

이제 직접 측정하는 양은 다음과 같습니다.

```text
launch/sync floor       = nccl_minimal - local_copy
accessible payload time = nccl_real - nccl_minimal
screened recoverable    = accessible fraction × (imbalance - 1) / imbalance
```

gate 결과도 `FAIL / PASS-UNPOWERED / PASS-POWERED`로 사전등록했습니다.
`PASS-UNPOWERED`이면 scheduler matrix를 실행하지 않고 native adapter correctness로
이동합니다. `PASS-POWERED`일 때도 검정력을 통과한 K만 실행합니다.

## 2. 온라인 scheduler 범위 축소

v1 selection cost와 clean-path roofline을 대조하면 온라인 선택 비용이 회수 가능한
시간을 넘을 가능성이 높다는 지적에 동의합니다. 따라서 broad matrix의 질문은
“계획 비용이 0인 best-found coordinator에 composition freedom이 존재하는가”로
좁혔습니다.

```text
Offline broad matrix
K:       1, 16, 64
Routing: uniform, mild_skew, strong_skew,
         request_correlated, temporally_unstable
Arms:    FIFO, random permutation, coordinated best-found replay
```

K=1은 speedup 판정 셀이 아니라 calibration anchor로 유지합니다. 최소 10회 반복을
두되, 최종 반복 수는 real-NCCL 기준 평균으로 정규화한 paired-difference SD로
결정합니다. 온라인 arm은 powered K=64의 최소 확인으로 제한하고, selection cost가
accessible-time bound보다 작을 때만 확장합니다.

## 3. 통계와 planner 진단

- predicted combined-load reduction의 1% vacuous threshold를 제거했습니다.
- predicted reduction을 연속 공변량으로 사용해 measured paired latency change와
  calibration curve를 적합합니다.
- accessible fraction과 measured rank-load imbalance도 함께 기록합니다.
- cell sign test는 보조 분석입니다. K/routing cell의 독립성이 보장되지 않으므로
  주 분석으로 사용하지 않습니다.
- restart별 raw cost와 best-so-far curve를 저장합니다.
- 마지막 두 restart에서 best objective가 개선되면 restart를 2배씩 늘리며,
  preregistered 최대 64회에서 멈춥니다.

## 4. phase attribution과 provenance

두 단계 기준을 반영했습니다.

```text
end-to-end harness 진행: unattributed <= 15%
phase-level 원인 주장:   unattributed <= 5%
                         mode/arm 간 차이 < 1 percentage point
```

NCCL algorithm/protocol/debug provenance를 필수화할 수 있는 fail-closed 옵션을
추가했습니다. fused-MoE source와 candidate configuration 파일의 SHA-256도
artifact에 기록합니다. clock lock 실패는 percent-level scheduler claim만 막고,
timing characterization과 adapter correctness는 exploratory evidence로 계속할 수
있도록 범위를 좁혔습니다.

## 5. correctness gate 반영

G1은 router GEMM의 M dimension이 같은 경우에만 FP32 exact match를 요구합니다.
M이 다르면 relative error `<= 1e-6`을 진단 기준으로 쓰고, 결정 gate는 expert ID와
margin으로 이동했습니다.

추가한 항목은 다음과 같습니다.

- top-8/top-9 expert 경계와 group-limited 경계의 near-tie margin 기록;
- `drop_count == 0` hard assertion;
- 최소 8개 deterministic batch composition;
- routing ID, remasking decision, finalized token의 정확 일치;
- token별 최대 출력 편차와 decision-margin 1 percentile의 비교.

Router weight는 stock semantics 확인 결과대로 nonlinear expert compute 이후에만
적용하며, normalized weight에 routed scaling 2.5를 반영한 뒤 FP32 weighted sum을
수행하고 shared expert를 한 번 더합니다.

## 6. 두 가지 구현상 정정

### Data-plane collective 수

현재 v2 data plane에는 layer execution당 실제 collective가 **3개** 있습니다.

```text
1. BF16 hidden dispatch
2. int32 expert-ID dispatch
3. BF16 hidden combine
```

split-count all-gather는 여기에 포함되지 않는 네 번째 control-plane collective이며
별도 계측합니다. 따라서 data-plane fixed 항을 2 collectives로 낮추는 LO1 제안은
현재 구현에는 적용하지 않았습니다. 대신 회계표에서 3 data + 1 control을 분리하고,
byte 항에도 hidden dispatch, expert-ID dispatch, hidden combine을 모두 포함했습니다.

### `FIFO + selection discarded` apply 비용

이 control은 critical plan을 실제로 eager 계산하고 checksum으로 소비를 증명한 뒤,
FIFO plan을 한 번 적용합니다. 실제 critical arm도 선택된 plan을 한 번 gather합니다.
control에 두 번째 artificial permutation을 추가하면 실제 arm에 없는 작업을
과금하므로 적용하지 않았습니다. 필요한 것은 plan 계산의 실행 증명이며 checksum을
raw run에 남깁니다.

## 7. communication substrate 범위

NCCL negative result를 substrate-general 결론으로 확대하지 않습니다. 회계 artifact는
data-plane fixed cost가 2x, 3x, 5x 감소할 때 accessible/recoverable fraction이 어떻게
달라지는지 함께 출력합니다. DeepEP-equivalent microbenchmark는 native correctness
뒤, substrate-general negative claim 전의 필수 항목으로 유지합니다.

## 8. 현재 준비 상태

- three-mode timing runner 구현;
- three-way gate analyzer 구현 및 합성 regression test 추가;
- adaptive planner restart와 raw best-so-far 기록 구현;
- continuous composition calibration analyzer 구현;
- data/control-plane 분리 roofline 회계 재생성;
- LLaDA correctness gate 보강;
- Ruff 통과, pytest **55 passed**.

아직 실행하지 않은 것은 실제 H100 timing gate와 그 결과에 따른 conditional matrix,
그리고 native LLaDA adapter correctness입니다. 다음 유료 실행은 full matrix가 아니라
3-mode gate부터 시작하며, `PASS-UNPOWERED` 또는 `FAIL`이면 scheduler matrix 비용을
즉시 절약하도록 고정했습니다.

재검토 시에는 다음 세 가지만 확인 부탁드립니다.

1. `screened recoverable >= 2 × MDE`가 advance budget gate로 충분히 보수적인지;
2. offline 15-cell 폭과 최소 10 paired repetitions의 배분이 음성 결과 해석에
   충분한지;
3. data-plane 3개 + control-plane 1개로 분리한 회계에서 추가로 빠진 payload 또는
   synchronization 항이 있는지.
