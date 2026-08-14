# 화면: Action List와 Forecast Validation

**대상 독자:** 이 두 페이지를 유지보수하는 엔지니어. 각 부분이 무엇을 표시하는지, 어떤
엔드포인트가 제공하는지, 모든 수치가 어디에서 계산되는지, 그리고 현재 무엇이 잘못되어 있는지를
설명합니다.

**이 문서의 위치.** `OVERVIEW_KO.md`가 예측의 목적을, `MODEL_KO.md`가 예측 생성 방식을,
`DATA_AND_PIPELINE_KO.md`가 데이터베이스까지 도달하는 과정을 설명합니다. 이 문서는 마지막
구간, 데이터베이스에서 픽셀까지를 다룹니다.

**하나의 아키텍처 규칙.** Next.js 앱은 예측이나 계획에 관해 **아무것도** 계산하지 않습니다. 두
화면의 모든 수치는 Python, 즉 `Time_Series_Forecasting/src/planning/`에서 계산되어 얇은 라우트
핸들러를 통해 전달됩니다. 이를 유지하십시오. 발주 수식의 구현이 정확히 하나만 존재하고,
브라우저 없이 테스트할 수 있다는 뜻입니다.

---

## 1. 요청이 이동하는 경로

```
브라우저
  └─ fetch(apiPath("/api/planning/…"))          src/lib/api-path.ts:12
       └─ Next 라우트 핸들러                      src/app/api/planning/**/route.ts
            └─ proxyPlanning()                   src/lib/planning-api.ts:19
                 │  AI_SERVICE_URL, 기본값 http://localhost:8000
                 │  헤더 x-forecast-token: FORECAST_API_TOKEN
                 └─ FastAPI                      api/main.py
                      └─ src/planning/calc.py    ← 모든 숫자가 여기에서 계산됨
```

**라우트 핸들러가 한 줄인 것은 의도입니다.** 검증을 하지 않습니다. FastAPI의
`Query(ge=…, le=…)`가 범위의 단일 출처이고, 클라이언트는 `types.ts:171-185`에서 같은 범위로
클램프합니다. 중간에 검증을 추가하면 범위가 어긋날 수 있는 세 번째 장소가 생깁니다.

### 프록시의 오류 분류

`planning-api.ts:63-159`. 페이지가 각각을 다르게 렌더링하므로 알아둘 가치가 있습니다.

| 조건 | 결과 | 비고 |
|---|---|---|
| 연결 실패 | `ensureForecastServer()` 후 1회 재시도 | **로컬** 서버는 자동 시작되고 **원격**은 750ms 대기. 이후 `{kind:"unreachable"}` 503 |
| 상위 404, `/planning/sku/` 경로가 아닌 경우 | `{kind:"outdated"}` | "예측 서버가 구버전을 실행 중입니다" |
| 상위 500 이상 | `forecastHealth()` 호출, `ready === false`면 누락 파일을 명시한 `{kind:"no_data"}` 503 | |
| 그 외 | 상태 코드를 그대로 전달 | 의도적. SKU 엔드포인트의 404가 의미를 유지하도록 |

자동 시작 동작이 `DATA_AND_PIPELINE_KO.md` 8절의 진단 함정의 원인입니다. 잘못 설정된
`AI_SERVICE_URL`이 정상처럼 보이는 것은 로컬의 무언가가 응답하기 때문입니다.

### 엔드포인트

| Next 라우트 | 상위 | 타임아웃 |
|---|---|---|
| `GET /api/planning/action-list` | `/planning/action-list` | 20초 |
| `GET /api/planning/sku/[sku]` | `/planning/sku/{id}` | 20초 |
| `GET /api/planning/sku/[sku]/history` | `/planning/sku/{id}/history` | 20초 |
| `GET /api/planning/not-forecast` | `/planning/not-forecast` | **40초**, SKU가 약 7배 |
| `GET /api/planning/validation` | `/planning/validation` | **60초**, 콜드 서버에서 저장된 실행 채점 시 전체 이력 로드 |
| `GET /api/planning/demand-patterns` | `/planning/demand-patterns` | 30초 |
| `GET /api/planning/demand-vs-forecast` | `/planning/demand-vs-forecast` | 30초 |
| `POST /api/planning/demand-trend` | `/planning/demand-trend` | 30초, **공용 헬퍼 미사용**. 헬퍼는 GET 전용 |
| `POST /api/planning/run-forecast` | `/planning/run-forecast` | 30초 |
| `GET /api/forecast/status/[jobId]` | `/forecast-status/{job_id}` | 5초, 역시 헬퍼 미경유 |

`/api/forecast/status`는 2026-08-13에 삭제된 14개의 `/api/forecast/*` 프록시 라우트 중 **유일한**
생존자입니다. 어떤 파이프라인이 작업을 생성했는지 전혀 모르는 범용 작업 기계이며, 그래서
살아남았습니다.

---

## 2. Action List

**목적.** 예측을 작업 목록으로 바꾸는 것. 어떤 SKU에 발주가 필요한지, 몇 개인지, 어떤 순서로
처리할지.

**라우트.** 목록은 `src/app/planning/action-list/page.tsx`, 상세는
`src/app/planning/action-list/[sku]/page.tsx`. SKU가 쿼리 문자열이 아니라 경로에 있는 것은 행을
공유하고 가운데 클릭으로 열 수 있게 하기 위함입니다.

### 2.1 컴포넌트 트리

```
ActionListPage (서버)
├── ActionListPageHeader
└── ActionListContent                    action-list-content.tsx:76   ← 허브, 948줄
    ├── 출처 표시 바                       trained_through / model / horizon_end /
    │                                    demoted_since_forecast / SAMPLE 재고 경고
    │   ├── ModelCard                    모델 버전 팝오버
    │   └── ForecastServerStatus         상태 배지; onRecovered가 재조회를 트리거
    ├── RunForecast                      run-forecast.tsx:67, 기본 접힘
    ├── 섹션 토글                          "Forecast" 대 "Not forecast"
    ├── NotForecastSection               section === "not-forecast"일 때만
    │   └── NotForecastTable             13주 비율 표, 발주 수량 없음
    ├── 요약 필터 버튼                     클릭 가능한 7개 카운트; 유일한 우선순위 필터 역할도 겸함
    ├── PlanningControls                 리드타임 / 검토주기 / service-z / 위험 구간
    ├── 검색, 카테고리·등급·추세 선택, 초기화
    ├── ColumnPicker                     선택적 컬럼 표시 여부
    ├── CSV 내보내기                       csv-export.ts
    ├── PortfolioChart                   접힘; 필터된 SKU의 실적과 예측 합계
    ├── 데이터 품질 경고 줄                 r.flags 개수, 클릭하면 필터
    ├── ActionListTable                  action-list-table.tsx
    │   └── ColumnHeaderMenu             정렬 + 컬럼별 체크박스 필터
    └── PlanningError                    재시도 버튼이 있는 오류 카드
```

`rememberSkuSequence()`(`sku-sequence.ts:37`)가 화면에 보이는 필터·정렬 순서를 `sessionStorage`에
기록하므로, 상세 페이지의 이전/다음이 정규 순서가 아니라 사용자가 보고 있는 순서를 따라갑니다.

**`DEFAULT_SORT = []`**(`action-list-table.tsx:254`). 클라이언트 정렬이 없는 것은 의도입니다.
서버의 작업 목록 순서는 어떤 단일 컬럼으로도 재현할 수 없기 때문입니다.

### 2.2 발주 권장 수식

**두 화면을 통틀어 가장 중요한 코드입니다.** `src/planning/calc.py:410-427`.

```python
# 모든 구성요소는 합계를 내기 전에 정수 단위로 반올림된다. 그래야 내역에 표시된
# 숫자들이 권장 수량과 정확히 일치한다.
df["safety_stock"] = (z * df["error_used"] * df["coverage_demand"]).round()
df["coverage_demand"] = df["coverage_demand"].round()
for col in ("preorder_backlog", "available_inventory",
            "confirmed_inbound", "inbound_in_window", "inbound_excluded"):
    df[col] = df[col].round()

df["recommended_order_qty"] = (
    df["preorder_backlog"]
    + df["coverage_demand"]
    + df["safety_stock"]
    - df["available_inventory"]
    - df["inbound_in_window"]
).clip(lower=0).astype(int)
```

말로 풀면:

```
coverage_weeks = lead_time_weeks + review_period_weeks
safety_stock   = round(service_z × error_used × coverage_demand_raw)

recommended_order_qty = max(0, int(
      round(preorder_backlog)
    + round(coverage_demand)
    + safety_stock
    − round(available_inventory)
    − round(inbound_in_window)
))
```

`safety_stock`은 **반올림되지 않은** coverage demand로 계산되며, 그 바로 다음 줄에서 coverage
demand 자체가 반올림된다는 점에 유의하십시오.

**각 입력값의 정확한 정의.**

| 입력 | 출처 | 비고 |
|---|---|---|
| `coverage_demand` | `_coverage_demand`, `calc.py:97-110` | 처음 `coverage_weeks` 주의 `yhat` 합계. 지평이 구간보다 짧으면 부족분은 지평의 주간 평균으로 채움 |
| `available_inventory` | `SUM(available)` = 재고 − 할당분 | 백오더는 **차감하지 않음** |
| `preorder_backlog` | 같은 테이블의 `SUM(backorder)` | 소요량에 **더함** |
| `inbound_in_window` | `calc.py:398-408` | 확정 상태만, 그리고 **ETA가 `coverage_end` 이하**여야 함. **ETA가 없는 입고는 절대 인정되지 않음** |
| `safety_stock` | `calc.py:415` | `z × error_used × coverage_demand`. 고정된 일수 규칙이 아니라 해당 SKU의 **자체 측정 WAPE**에 감당해야 할 수요를 곱한 값 |
| `lead_time_weeks` | 파라미터 | `coverage_weeks`와 `coverage_end`를 통해서만 반영 |
| **초안(draft)** | | **절대 차감하지 않음.** 아래 참조 |

**초안은 표시하되 차감하지 않으며 이는 의도입니다**(`calc.py:238-240`, `types.ts:22-34`). 초안은
확약이 아니므로, 이를 인정하면 누군가 이미 조치한 SKU를 정확히 과소 발주하게 됩니다.

**로트 사이즈, MOQ, 컨테이너 반올림은 어디에도 존재하지 않습니다.** 비즈니스가 필요로 한다면
그것은 누락된 설정값이 아니라 새로운 작업입니다.

**기본값**, `calc.py:63-83`, `types.ts:142-147`에 반영:

```python
DEFAULT_PARAMS = {
    "lead_time_weeks": 8,          # 공급업체 + 운송
    "review_period_weeks": 1,      # 발주 주기
    "service_z": 1.0,              # 약 84% 서비스 수준; 1.65 ≈ 95%
    "best_seller_demand_share": 0.50,
    "stockout_horizon_days": 30,
}
```

`best_seller_demand_share`는 UI에 노출되지 않습니다.

### 2.3 내역과 반올림 순서가 중요한 이유

`calc.py:612-660`이 상세 페이지에 표시되는 산식을 만듭니다. 각 줄은 `Sign`을 갖습니다. `+1` 더함,
`−1` 뺌, `0` 합계, `None` 참고용 부기. 저장된 `recommended_order_qty`를 재계산보다 우선 사용하므로
내역이 목록과 어긋날 수 없습니다.

모든 구성요소를 합산 전에 반올림하는 이유도 같습니다. 합계만 반올림하면 표시된 줄들이 합계와
한두 단위 어긋나며, 그러면 산식을 보여주는 의미 자체가 사라집니다.

### 2.4 재고 소진 일수와 예상 품절일

`_days_until_consumed`, `calc.py:124-146`. 해당 SKU **자신의 예측 곡선**을 주 단위로 따라가며,
재고가 소진되는 주 안에서는 선형 보간하고, 지평을 넘어서면 지평의 평균 속도로 계속합니다. 예측이
0인 주는 소진으로 보지 않고 건너뜁니다.

`calc.py:435-457`에서 적용됩니다. 알아야 할 규칙:

```python
in_time = inb > 0 and np.isfinite(eta) and eta <= t_on_hand
```

**입고는 선반이 비기 전에 도착할 때만 커버를 연장합니다.** 그렇지 않으면 품절 이후의 보충이므로
`days_to_stockout`은 이를 완전히 무시합니다.

그 결과 같은 행에 상충하는 듯한 두 가정이 남습니다. `days_to_stockout`은 늦은 입고를 무시하고,
`recommended_order_qty`는 커버 구간 안의 입고를 인정합니다. **supply gap이 이를 조정합니다**
(`calc.py:528-544`). 입고가 예상 품절 이후에 도착하면 `supply_gap_days`가 그 차이이고,
`gap_closable_by_order`가 지금 발주하면 실제로 도움이 되는지를 알려줍니다.

### 2.5 우선순위 사다리

`calc.py:502-511`. 세 가지 상태, 먼저 일치하는 것이 이기고, 숫자가 작을수록 먼저입니다.

| 순위 | 라벨 | 조건 |
|---|---|---|
| 1 | **Preorder** | `preorder_backlog > 0` |
| 2 | **No Stock** | `available_inventory <= 0` |
| 99 | **Routine** | 나머지 전부 |

셋 다 하나의 변수, 즉 재고 상황이 어떤가에 대한 값이며, 그래서 사다리가 올바른 형태입니다.

**"Best Seller"는 예전에 3순위였고 제거되었습니다**(`calc.py:485-501`). 재고 상태가 아니라
중요도라는 다른 질문에 답하며, 항상 Preorder와 No Stock에 밀렸는데 그 둘이야말로 잘 팔리는
제품이 가장 자주 들어가는 대기열입니다. 지금은 `calc.py:472-483`에서 최근 수량의 50%를 차지하는
최소 SKU 집합으로 계산되는 독립 불리언입니다.

**작업 목록 순서**, `calc.py:555-559`: 우선순위, 그다음 best seller, 그다음 발주 수량. 대부분의
Routine 행이 0에서 동률이므로 안정 정렬을 위해 `mergesort`를 씁니다.

**클라이언트 쪽 사본은 대소문자를 구분합니다.** `action-list-table.tsx:169-184`의 `PRIORITY`는
정확히 그 문자열을 키로 하는 딕셔너리입니다. `"No stock"`은 오류를 내지 않고 조용히 Routine
스타일로 넘어갑니다.

### 2.6 신뢰도 등급

`src/planning/reliability.py:34-53`. `outputs/reports/ml_accuracy_by_sku.csv`에서 SKU별 WAPE를
읽습니다.

| 등급 | 경계 | 표시 |
|---|---|---|
| good | ≤ 0.15 | ●●● |
| fair | ≤ 0.30 | ●●○ |
| poor | 초과 | ●○○ |
| none | 미측정 | ○○○ |

**그 CSV는 낡았습니다.** 3.7절 참조. 이 화면에도 영향을 줍니다.

### 2.7 `error_basis`와 수요 대역 폴백

`calc.py:299-355`. 안전재고에는 SKU별 오차가 필요한데 대부분의 SKU에는 측정된 값이 없습니다.
폴백 순서:

**자체 측정 WAPE → 수요 대역 중앙값 → 세그먼트 중앙값 → 전체 중앙값 → 0.0**

`error_basis`가 어느 것이 사용되었는지 보고합니다. `"measured"`, `"demand band"`,
`"segment median"`, `"overall median"`.

대역은 주간 수량 기준이며 좌측 폐구간입니다. `calc.py:35`:

```python
ERROR_BAND_EDGES = [0.0, 2.0, 4.0, 6.0, 10.0, float("inf")]
MIN_BAND_MEASURED = 5
```

둥근 숫자가 아니라 오차 곡선이 실제로 측정된 대역입니다. pooled WAPE는 대역 안에서 평탄하고
대역 사이에서 계단처럼 변합니다.

**결정적인 세부사항 두 가지.**

대역 분류는 `recent_units / 4`가 아니라 프로파일러의 후행 **13주** 주간 평균인 **`recent_mean`**을
씁니다. 대역 경계가 13주 평균으로 측정되었고 `src/profile.py`의 모든 임계값이 같은 구간을
사용합니다. 4주로 분류하면 강등 규칙상 주당 2단위 미만인 SKU가 하나도 없어야 하는데 11개가
그 아래로 들어갔습니다.

측정 SKU가 5개 미만인 얇은 대역은 **수량 축에서 가장 가까운 신뢰 대역**에서 값을 빌리며, 동률일
때는 수량이 낮은 쪽으로 갑니다. 첫 버전은 대신 세그먼트 중앙값으로 넘어갔고, 가장 예측하기
어려운 0.357이면서 4개로 가장 얇았던 [0,2) 대역이 고수량 SKU가 지배하는 중앙값을 물려받았습니다.
신중하려고 만든 가드가 그 33개 구성원에게 오히려 0.038만큼 **적은** 여유를 주었습니다.

하드코딩 상수였던 `PROMOTED_ERROR_FALLBACK = 0.24`는 2026-08-12에 제거되었습니다. 기록은
`calc.py:42-61`에 있습니다.

**여기에 프런트엔드 낡음 버그가 두 개 있습니다.** `types.ts:89`는 여전히
`"measured" | "promoted cohort" | "segment median" | string`으로 선언하고
`reliability-card.tsx:71-75`는 여전히 `"promoted cohort"`를 매핑합니다. 둘 다 `"demand band"`와
`"overall median"`을 모르므로 `basisLabel[errorBasis] ?? errorBasis`를 통해 영어 원문이 그대로
렌더링됩니다. 크래시는 아니지만 **한국어 로케일에서 번역되지 않은 문자열이 보입니다.**

### 2.8 `forecast_runs_high` 경고

`calc.py:376-389`. 예측이 최근 4주 평균의 1.5배 이상이고 **동시에** 그보다 20단위 이상 클 때
플래그를 세웁니다. 두 조건 모두 필요합니다.

**표시만 하고 반영하지 않습니다.** 권장 수량은 여전히 모델 값을 씁니다. 상세 페이지에서 이 안내를
발주 카드 **위에** 배치한 것은 의도이며, 독자가 숫자보다 경고를 먼저 보게 하기 위함입니다.

### 2.9 Not-forecast 섹션

`calc.py:663-795`. intermittent SKU는 카탈로그의 약 87%이고 최근 수량의 5분의 1이므로, 화면에서
완전히 빼는 것은 정직하지 않았습니다.

**소속은 예측 파일이 아니라 계획 테이블에 없는 것으로 정의됩니다**(`calc.py:711-718`). 예측
파일을 기준으로 했더니 실행 이후 강등된 15개 SKU가 양쪽 섹션 모두에서 빠졌습니다. 계획 테이블을
기준으로 하면 두 섹션이 구조적으로 분할이 됩니다.

**다른 기준, 그리고 의도적으로 다른 컬럼 이름.** `recommended_order_qty`도, coverage demand도,
안전재고도, WAPE도, 품절일도 없습니다. 예측 없이는 어느 것도 존재할 수 없고, 없는 것이 정직한
답입니다. 대신 후행 13주 비율, 그로부터 계산한 재고 소진 일수, 그리고 커버가 리드타임보다
짧을 때의 불리언 `reorder_signal`이 있습니다. 표 머리글은 "13w demand", "per week", "cover"이며
다른 어디에도 등장하지 않으므로 독자가 채점된 예측 수치로 오인할 수 없습니다.

기억할 만한 작은 결정 두 가지. 판매가 전혀 없을 때 `days_of_cover`는 무한대가 아니라 `NaN`입니다.
0으로 나누면 화면에서 "절대 소진되지 않음"으로 읽히는데 질문 자체가 성립하지 않기 때문입니다.
그리고 재고 필드는 값이 없을 때 0이 아니라 **공란**으로 둡니다. "기록 없음"은 "없다고 기록됨"이
아니기 때문입니다.

SKU 수가 약 7배이므로 **섹션을 열었을 때만** 조회합니다.

### 2.10 Run Forecast 버튼

`run-forecast.tsx:67`, 기본 접힘, 출처 표시 바 바로 아래.

`POST /api/planning/run-forecast?horizon=N` → FastAPI `api/main.py:185-240` → 자체 프로세스
그룹에서 `scripts/ml_prepare_data.py --force --horizon N`을 실행하는 백그라운드 스레드.
`create_job("forecast")`가 `/planning/prepare-data`와 작업 유형을 공유하므로 동시 요청은
**409**를 받습니다. 같은 파일에 쓰기 때문입니다.

상태는 `GET /api/forecast/status/{jobId}`를 **2초마다** 폴링하며, `jobId`와 `status`를 키로 하여
스스로 멈춥니다.

**진행 바는 명시되지 않은 계약입니다.** `run-forecast.tsx:61`:

```ts
const m = /Step (\d)\/4/.exec(line);
if (m) seen = Math.max(seen, Number(m[1]));
```

스크립트 자체의 표준 출력을 정규식으로 읽습니다. `ml_prepare_data.py`에서 그 접두사를 바꾸면
진행 바가 조용히 깨집니다. 개수가 아니라 `Math.max`인 것은 이전 단계를 언급하는 줄이 진행을
뒤로 되돌리지 못하게 하기 위함입니다.

성공했을 때만 목록을 재조회하여 `trained_through`가 갱신됩니다. **실패 시에는 의도적으로 하지
않습니다.** 실패 후 재조회하면 변하지 않은 숫자가 마치 실패가 반영된 것처럼 보이기 때문입니다.

`/cancel-forecast/{job_id}`는 여전히 존재하지만 **취소 버튼은 없습니다.** 이 파이프라인을 중간에
멈추는 것이 끝까지 두는 것보다 나쁘기 때문입니다.

### 2.11 상세 페이지

`/planning/action-list/[sku]`. 발주 카드는 숫자가 아니라 **산식으로** 렌더링되고, 구간별 백테스트
표가 있는 신뢰도 카드, 지표 행, 그리고 차트 두 개가 있습니다.

모델이 예측하지 않는 SKU에 대해 `/planning/sku/{id}`는 **세 가지 서로 다른 상세 메시지**와 함께
404를 반환합니다(`api/main.py:502-520`). 예측 실행에는 있었으나 강등됨, 프로파일은 있으나 예측된
적 없음, 알 수 없는 SKU. 그러면 페이지는 `/planning/sku/[sku]/history`로 폴백하여 판매 이력만
보여줍니다.

`/planning/sku/{id}`는 이전/다음을 위한 두 번째 요청을 피하려고 전체 SKU 목록과 해당 행의 위치도
약 10 KB로 함께 반환합니다.

**타당 구간(plausible band)은 더 이상 없습니다.** `api/main.py:631-635`에 이유가 기록되어
있습니다. 안전재고가 더하는 것과 같은 오차만큼 coverage demand를 늘리고 있었으므로, 그 상단이
곧 권장 수량이었습니다.

---

## 3. Forecast Validation

**목적.** 예측을 신뢰할지 판단하게 하는 것. 위에서 아래로 읽는 여섯 개의 증거 섹션이며, 그 순서가
곧 논증입니다.

**라우트.** `src/app/planning/forecast-validation/page.tsx`.

**컴포넌트는 `forecast-validation/`이 아니라 `src/components/planning/validation/`에 있습니다.**
이 불일치가 가장 먼저 알아야 할 사항입니다.

### 3.1 렌더링 순서대로의 섹션

번호는 `VALIDATION_SECTIONS`(`section-heading.tsx:34-41`)에서 나오고 렌더링 순서는
`validation-content.tsx:214-297`입니다. 둘은 일치하며 **반드시 일치해야 합니다.** 03번 제목이
네 번째에 있는 것은 번호가 아예 없는 것보다 나쁩니다. 섹션을 추가한다는 것은 그 배열의 해당
위치에 추가한다는 뜻입니다.

| # | id | 제목 | 주장 |
|---|---|---|---|
| 01 | `comparison` | 모델 대 스프레드시트 | 동일 구간에서 pooled WAPE로 모델이 V1을 이김. **패배한 칸을 모두 표시** |
| 02 | `demand` | 수요의 형태 | 그 주장의 범위: 모델이 대변하는 SKU와 그 물량 비중. 여기에 예측은 없음 |
| 03 | `trajectory` | 수요 대 예측 | 같은 증거를 시간축으로: 오차가 언제 어느 방향으로 발생했는가 |
| 04 | `over-time` | 실제 제공된 예측의 성능 | 표본 외 기록: 결과를 알기 전에 발행된 예측 |
| 05 | `outliers` | SKU 단위 분해 | 개선이 전반적인지 소수의 큰 승리에 의한 것인지 |
| 06 | `final-test` | 최종 테스트 구간 | 아직 의도적으로 주장하지 않는 것 |

**혼동을 부르는 조건부 렌더링 두 가지.** 섹션 03은 추세 조회가 실패하거나 예측이 비어 있으면
오류 카드 없이 통째로 생략됩니다. 이는 의도입니다. 섹션 04는 이력 저장소가 비어 있을 때
억제되는데, 두 패널이 같은 말을 하는 것을 피하기 위함입니다.

**이 페이지가 스스로에게 부과한 설계 기준**, `comparison-section.tsx:1-14`:

> 헤드라인은 하나의 숫자이고, 하나의 숫자야말로 잘못된 결론을 부른다. 그래서 아래의 격자는
> 선택적 세부사항이 아니다. 스프레드시트가 여전히 이기는 칸을 포함해 모든 세그먼트와 모든
> 구간을 보여준다. **이긴 것만 보고하는 비교는 증거가 아니기 때문이다.**

모델 버전은 컴포넌트에 이름으로 박히지 않고 페이로드에서 읽으므로, 새 모델이 등장해도 프런트엔드
파일은 바뀌지 않습니다.

### 3.2 섹션별 데이터 출처

| 섹션 | 출처 | 계산 방식 |
|---|---|---|
| 01, 05 | `outputs/reports/ml_accuracy.csv`, `ml_accuracy_by_sku.csv` | **저장됨**, `scripts/ml_accuracy_report.py`가 생성 |
| 02 | `load_sales()` | 요청 시 실시간 |
| 03, 04 | `src/ml/serving/history.py` | 누적 저장소에서 실시간 |
| 06 | `outputs/reports/final_test.json` | 저장됨, 그리고 **현재 읽지 않음** |

`meta.accuracy_computed`는 기록된 타임스탬프가 아니라 `ml_accuracy.csv`의 **파일 mtime**입니다.

`comparison.grid`는 `(세그먼트, 구간)` 칸마다 **모델 버전을 동적 키로** 담으며, 그래서
`ValidationCell`에 인덱스 시그니처가 있습니다. `windows`는 이름이 아니라 컷오프 기준 시간순으로
정렬됩니다. 알파벳순으로는 Dec-Feb, Mar-May, Oct-Dec가 되어 추세를 거꾸로 읽게 만들기 때문입니다.

`outliers.rows`는 채점된 전체 풀 약 572행을 전송합니다. `top_n`은 페이지가 표시하는 개수이지
전송되는 개수가 아닙니다.

### 3.3 차트

**Plotly**, `react-plotly.js`를 통해 `dynamic(..., { ssr: false })`로 클라이언트 전용 로드. 이
페이지에 Recharts는 없습니다.

| 차트 | 표시 내용 |
|---|---|
| 주간 수요, 누적 영역 | 예측 대상 SKU 대 intermittent 잔여 |
| 파레토 집중도 곡선 | 누적 SKU 대비 누적 수요, 균등 분포 기준선 포함 |
| 수요 대 예측 궤적 | 실제 주간 수량, 선택한 리드의 저장 실행 예측, 전방 지평, V1 |

차트처럼 보이지만 Plotly가 **아닌** 두 가지: `outliers-section.tsx`의 SKU별 델타 히스토그램은
±1에서 잘리고 양끝에 라벨된 오버플로 버킷이 있는 CSS 크기 조절 div이며,
`over-time-section.tsx`의 실행별 오차 추세는 표 셀 안에 그린 막대입니다.

차트가 숨기지 않고 설명하는 두 가지 부재: v11이 점 예측만 산출하므로 **예측구간 없음**
(`has_intervals: false`), 그리고 이력 저장소가 모델 자체 예측만 보관하고 V1은 실행마다 재계산되므로
**V1은 전방 지평에만** 표시.

### 3.4 다국어 처리

**i18n 라이브러리가 아닙니다.** React 컨텍스트의 `pick(ko, en)` 함수이며
`src/lib/i18n/i18n-provider.tsx:67-72`에 있습니다. 문자열은 호출 지점에 인라인으로, 한국어를
먼저 씁니다.

`messages.ts`를 기반으로 하는 키 기반 사전 `t(key)`도 존재하지만 **`validation/` 아래 어떤
파일도 사용하지 않습니다.**

```tsx
"use client";
import { useI18n } from "@/lib/i18n/i18n-provider";

export function MySection() {
  const { pick } = useI18n();
  return <h2>{pick("예측 검증", "Forecast Validation")}</h2>;
}
```

보간은 `pick`을 감싸는 대신 **양쪽 인자 안에서** 템플릿을 각각 씁니다. 한국어가 어순을 바꾸기
때문입니다.

```tsx
{pick(
  `${headline.cells_total}개 구간 중 ${headline.cells_won}개에서 우세`,
  `ahead in ${headline.cells_won} of ${headline.cells_total} cells`,
)}
```

`pick`은 `title=` 툴팁과 Plotly 트레이스의 `name:` 필드에도 값을 공급하므로 번역된 텍스트가 차트
범례까지 도달합니다.

**새 섹션을 추가할 때의 점검 목록:** `"use client"`, `useI18n()`, 모든 리터럴을 `pick`으로 감싸기,
번역된 `title`과 `description`을 `SectionHeading`에 전달, 그리고 `VALIDATION_SECTIONS`의
**렌더링 위치에** `[ko, en]` 라벨 쌍과 함께 섹션 추가.

로케일 결정 순서: 기본 `"en"`, 그다음 `localStorage["demandpilot-locale"]`, 그다음
`GET /api/user/preferences`의 `app.locale`.

### 3.5 최종 테스트 데이터

`outputs/reports/final_test.json`, `scripts/ml_41_final_test.py`가 생성하며 덮어쓰기를 거부합니다.

```json
{
  "run_at": "2026-08-13T12:02:18-07:00",
  "commit": "4a19ca1d177bcb596d31af16b5fa818f1d458ecf",
  "snapshot": "2026-08-03-v2",
  "v1_orders_raw": { "path": "…/orders_raw.parquet",
                     "md5": "fd90514306d8601e126700262ab02c8c" },
  "cutoff": "2026-05-04",
  "test_weeks": ["2026-05-11", … , "2026-07-13"],
  "scores": {
    "v11":      { "smooth/long": 0.1324, "smooth/short": 0.2061, "TOTAL": 0.1784 },
    "V1":       { "smooth/long": 0.1872, "smooth/short": 0.3772, "TOTAL": 0.3059 },
    "baseline": { "smooth/long": 0.1282, "smooth/short": 0.2013, "TOTAL": 0.1739 }
  },
  "v11_vs_v1":       { "short": {"delta": -0.1711, "se": 0.0215, …},
                       "long":  {"delta": -0.0548, "se": 0.0174, …} },
  "v11_vs_baseline": { "short": {"delta":  0.0048, "se": 0.0108, …},
                       "long":  {"delta":  0.0042, "se": 0.0141, …} }
}
```

**이 파일은 버전 관리에 없습니다.** `.gitignore:38`이 `outputs/reports/*`를 제외하고 39~45행이
네 개 파일을 이름으로 다시 포함시키는데, `final_test.json`은 그중에 없습니다. 유일한 사본이 실행한
기기에만 있고, 테스트는 1회용이므로 재생성할 수 없습니다. `BACKLOG.md`는 "파일이 커밋되어 있다"고
적고 있으며 이는 틀렸습니다.

---

### 3.6 결함: 페이지가 최종 테스트를 실행하지 않았다고 표시함

**상태: 미해결. BACKLOG 30번 항목이 코드 변경 없이 닫혔습니다.**

`api/main.py:1057-1062`, 디스크의 현재 상태:

```python
"final_test": {
    # 고정되어 있고, 격리되어 있으며, 아직 실행되지 않음. 빈 패널을 설명 없이
    # 두는 대신 무엇이 예정되어 있는지 말할 수 있도록 보고한다.
    "cutoff": _ML_FINAL_TEST_CUTOFF,
    "evaluated": False,
},
```

`api/main.py`는 **`final_test.json`을 한 번도 열지 않습니다.** 그 경로에 대한 유일한 Python
참조는 생성 스크립트입니다.

그다음 `validation-content.tsx:273-296`이 `v.final_test.evaluated`의 **양쪽** 분기 모두에서
`EmptySection`을 렌더링합니다. API가 `false`를 보내므로 독자는 현재 다음을 봅니다.

> **아직 평가하지 않았습니다.**
> 모델 개발이 끝날 때까지 이 구간은 한 번도 사용하지 않습니다.

2026-08-13 이후로 거짓입니다. 예측을 신뢰할지 판단하게 하는 유일한 화면의 마지막 섹션에 있는
거짓 진술이며, 프로젝트가 가진 가장 강력한 증거를 담은 섹션입니다.

**수정은 세 부분입니다. 플래그만 뒤집는 것으로는 부족합니다.** `true` 분기 역시 "결과 표시 준비
중"이라고 적힌 자리표시자이기 때문입니다.

1. **파일을 제공하십시오.** `/planning/validation` 핸들러에서
   `outputs/reports/final_test.json`을 읽어 `final_test` 페이로드에 담으십시오. 별도의 형태를
   새로 만들지 말고 JSON을 따르게 하여 둘이 어긋날 수 없게 하십시오.
2. **타입을 넓히십시오.** `types.ts:149`가 `{ cutoff: string; evaluated: boolean }`입니다.
3. **양쪽 절반을 모두 렌더링하십시오.** v11은 V1을 큰 격차로 이기고 **동시에** 구조적
   베이스라인과 동률입니다. 앞의 것만 렌더링하는 것은 `comparison-section.tsx`가 자체 코드
   가이드에서 거부하는 실패 방식이며, 독자가 다른 섹션보다 더 무겁게 받아들이라고 안내받는
   섹션이므로 여기서는 더 강하게 적용됩니다. 캘리브레이션 수치, 즉 V1의 28% 과소 대 v11의 0.0%도
   포함하십시오. 두 언어 문자열이 모두 필요합니다. 이 페이지는 전체가 EN/KO이며 영어만 있는
   결과는 이 페이지 최초의 미번역 콘텐츠가 됩니다.

**패널 위의 섹션 설명은 유지하십시오.** 왜 그 구간이 격리되었고 왜 그 때문에 그 수치가 위의
것들보다 가치 있는지를 설명합니다. 테스트 실행 후에도 여전히 참이며, 그 결과가 또 하나의
백테스트로 읽히는 것을 막아주는 요소입니다.

**아무것도 이를 막고 있지 않습니다.** 테스트는 실행되었고 데이터는 존재하며 추가 측정은 필요하지도
바람직하지도 않습니다.

이와 함께 낡아지는 것이 두 가지 더 있습니다. `section-heading.tsx:12-15`는 페이지 순서가 "아직
의도적으로 주장하지 않는 것"으로 끝난다고 서술하는데, 결과가 렌더링되면 마지막 섹션은 페이지에서
가장 강한 주장이 됩니다. 그리고 그 같은 주석은 이미 20줄 아래의 배열과 어긋나는 **낡은 섹션
순서**를 나열하고 있습니다. 배열 쪽이 신뢰할 것입니다.

### 3.7 결함: 정확도 리포트가 낡음

**상태: 미해결, 이전에 어디에도 기록되지 않음.**

`outputs/reports/ml_accuracy.csv`와 `ml_accuracy_by_sku.csv`는 둘 다 **2026-07-30**자입니다.
따라서 다음 셋 모두보다 앞섭니다.

- 예측 가능한 카탈로그의 41%를 되살린 2026-08-11의 프로파일링 수정,
- smooth 집합을 467개에서 340개로 옮긴 2026-08-12의 임계값 정렬,
- 2026-08-13의 V1 as-of off-by-one 수정.

**결과.** Forecast Validation의 섹션 01, 섹션 05, 그리고 Action List의 신뢰도 등급이 모두 더 이상
존재하지 않는 모집단과 알려진 체계적 오차가 있는 V1 열로 계산되고 있습니다. 수치가 눈에 띄게
다릅니다. 한 예로 Oct-Dec의 v11 smooth/short는 CSV에서 **0.1783**이지만 올바른 값은
**0.2473**입니다.

**두 파일 모두 git에 추적되며**, 와일드카드 이후 `.gitignore:39-40`에서 다시 포함됩니다. 따라서
낡은 버전이 커밋되어 있고 서버가 배포하는 것도 그것입니다.

**수정 방법.** `scripts/ml_accuracy_report.py`를 다시 실행하십시오. 세 개의 개발 구간에서 각각
재학습하며 구조상 격리 구간은 건드리지 않습니다. 격자를 `OVERVIEW_KO.md` 6절과 대조한 뒤
**재생성된 파일을 커밋하십시오.** 그렇지 않으면 서버는 계속 낡은 것을 제공합니다.

**근본 원인은 문서의 공백입니다.** 스크립트의 docstring은 "모델 버전이 바뀔 때" 갱신하라고 되어
있습니다. 불완전합니다. **모집단**이 바뀔 때도 갱신해야 하며, 프로파일링이나 임계값 변경이 바로
그런 경우인데 지금까지 아무도 이를 기록하지 않았습니다.

---

## 4. 알려진 결함 정리

| # | 위치 | 내용 |
|---|---|---|
| 1 | `api/main.py:1061`, `validation-content.tsx:273` | 최종 테스트가 미실행으로 표시됨. 3.6절 |
| 2 | `outputs/reports/ml_accuracy*.csv` | 2026-07-30 기준으로 낡음. 3.7절 |
| 3 | `.gitignore:38` | `final_test.json`이 버전 관리에 없고 재생성 불가. 3.5절 |
| 4 | `types.ts:89`, `reliability-card.tsx:71-75` | `error_basis` 값이 낡음. 한국어 로케일에서 미번역 문자열 노출. 2.7절 |
| 5 | `section-heading.tsx:12-15` | 주석이 낡은 섹션 순서를 나열 |
| 6 | `run-forecast.tsx:61` | `ml_prepare_data.py` 표준 출력과의 명시되지 않은 `Step N/4` 계약 |
| 7 | `SkuForecastsService.getForecastBounds()` | `/api/forecast/bounds` 삭제로 고아가 됨. 통과하는 테스트가 남아 있어 유지되는 것처럼 보이는 죽은 코드 |

크래시를 일으키는 것은 없습니다. 1번과 2번이 독자에게 정확도를 오해하게 만드는 둘이며, 누군가
이 화면으로 의사결정을 하기 전에 고칠 가치가 있습니다.
