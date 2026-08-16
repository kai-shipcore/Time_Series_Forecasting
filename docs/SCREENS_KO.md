# 화면: Action List와 Forecast Validation

유지보수자용 참조 문서. 각 구성 요소가 무엇을 표시하는지, 어느 endpoint가 이를 제공하는지, 각 수치가 어디에서 계산되는지를 다룹니다.

**아키텍처 규칙.** 두 화면의 모든 수치는 `Time_Series_Forecasting/src/planning/` 내부의 Python에서 계산되며, 얇은 route handler를 통해 프록시됩니다.

**근거.** 주문 공식의 구현이 하나뿐이며, 브라우저 없이 테스트할 수 있습니다. 이 구조는 그대로 유지합니다.

## 1. 요청 경로

```
browser
  └─ fetch(apiPath("/api/planning/…"))          src/lib/api-path.ts:12
       └─ Next route handler                     src/app/api/planning/**/route.ts
            └─ proxyPlanning()                   src/lib/planning-api.ts:19
                 │  AI_SERVICE_URL, default http://localhost:8000
                 │  header x-forecast-token: FORECAST_API_TOKEN
                 └─ FastAPI                      api/main.py
                      └─ src/planning/calc.py    ← every number is computed here
```

Route handler에는 검증 로직이 없습니다. 경계값의 출처는 FastAPI의 `Query(ge=…, le=…)`이며, `types.ts:171-185`가 동일한 값으로 제한합니다.

### 프록시 오류 분류

`planning-api.ts:63-159`.

| 조건 | 결과 |
|---|---|
| 연결 실패 | `ensureForecastServer()` 실행 후 한 번 재시도. 로컬 서버는 자동으로 시작되고, 원격 서버에는 750 ms 대기가 적용됩니다. 그다음 `{kind:"unreachable"}` 503 |
| 업스트림 404이면서 `/planning/sku/` 경로가 아닌 경우 | `{kind:"outdated"}`, "예측 서버가 이전 리비전을 실행 중입니다" |
| 업스트림 500 이상 | `forecastHealth()`를 호출하고, `ready === false`이면 누락된 파일을 명시한 `{kind:"no_data"}` 503 |
| 그 밖의 모든 경우 | 상태 코드를 그대로 전달하므로, SKU endpoint의 404는 원래 의미를 유지합니다 |

자동 시작은 `DATA_AND_PIPELINE_KO.md` §9의 진단 함정을 유발합니다.

### Endpoint

| Next route | 업스트림 | 타임아웃 |
|---|---|---|
| `GET /api/planning/action-list` | `/planning/action-list` | 20 s |
| `GET /api/planning/sku/[sku]` | `/planning/sku/{id}` | 20 s |
| `GET /api/planning/sku/[sku]/history` | `/planning/sku/{id}/history` | 20 s |
| `GET /api/planning/not-forecast` | `/planning/not-forecast` | 40 s, SKU 수의 대략 7배 |
| `GET /api/planning/validation` | `/planning/validation` | 60 s, 저장된 실행을 채점할 때 콜드 상태의 서버에서는 전체 이력을 적재 |
| `GET /api/planning/demand-patterns` | `/planning/demand-patterns` | 30 s |
| `GET /api/planning/demand-vs-forecast` | `/planning/demand-vs-forecast` | 30 s |
| `POST /api/planning/demand-trend` | `/planning/demand-trend` | 30 s, 공용 헬퍼는 GET 전용이므로 이를 경유하지 않음 |
| `POST /api/planning/run-forecast` | `/planning/run-forecast` | 30 s |
| `GET /api/forecast/status/[jobId]` | `/forecast-status/{job_id}` | 5 s, 이 역시 헬퍼를 경유하지 않음 |

`/api/forecast/status`는 2026-08-13에 삭제된 14개의 `/api/forecast/*` 프록시 route 중 유일하게 남은 것입니다.

## 2. Action List

**목적.** 예측을 작업 목록으로 바꾸는 것. 즉 어떤 SKU를, 몇 개를, 어떤 순서로 주문할지를 제시합니다.

**Route.** `src/app/planning/action-list/page.tsx`(목록), `.../[sku]/page.tsx`(상세). SKU가 경로에 포함되므로 각 행을 공유할 수 있습니다.

### 2.1 컴포넌트 트리

```
ActionListPage (server)
├── ActionListPageHeader
└── ActionListContent                    action-list-content.tsx:76   ← the hub, 948 lines
    ├── provenance bar                   trained_through / model / horizon_end /
    │                                    demoted_since_forecast / SAMPLE-inventory warning
    │   ├── ModelCard                    model version popover
    │   └── ForecastServerStatus         up/down badge; onRecovered triggers a refetch
    ├── RunForecast                      run-forecast.tsx:67, collapsed by default
    ├── section toggle                   "Forecast" vs "Not forecast"
    ├── NotForecastSection               only when section === "not-forecast"
    │   └── NotForecastTable             13-week-rate table, no order quantity
    ├── summary filter buttons           7 clickable counts; the only priority filter
    ├── PlanningControls                 lead / review / service-z / risk window
    ├── search, category/tier/trend selects, Reset
    ├── ColumnPicker                     which optional columns are visible
    ├── Export CSV                       csv-export.ts
    ├── PortfolioChart                   collapsed; actual + forecast over filtered SKUs
    ├── data-quality warning line        counts of r.flags, click to filter
    ├── ActionListTable                  action-list-table.tsx
    │   └── ColumnHeaderMenu             sort + per-column checkbox filter
    └── PlanningError                    error card with retry
```

`rememberSkuSequence()`(`sku-sequence.ts:37`)는 필터링되고 정렬된 순서를 `sessionStorage`에 기록하여 상세 페이지의 Prev/Next에 사용합니다.

`DEFAULT_SORT = []`(`action-list-table.tsx:254`)은 클라이언트 측 정렬을 비활성화합니다.

**근거.** 서버가 만든 작업 목록 순서는 어떤 단일 열로도 재현할 수 없습니다.

### 2.2 추천 공식

`src/planning/calc.py:410-427`.

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

각 구성 요소는 합계를 내기 **전에** 정수 단위로 반올림되므로, 상세 페이지의 내역 합계가 수량과 정확히 일치합니다. `safety_stock`은 반올림하지 않은 커버리지 수요를 사용하며, 한 줄 뒤에서 반올림됩니다.

| 입력 | 출처 | 참고 |
|---|---|---|
| `coverage_demand` | `_coverage_demand`, `calc.py:97-110` | 예측 주차 중 처음 `coverage_weeks` 구간의 `yhat` 합계. 지평이 그보다 짧으면 부족분은 해당 지평의 주간 평균 속도로 채워집니다 |
| `available_inventory` | `SUM(available)` = on_hand − allocated | 이월 주문을 차감한 값이 **아닙니다** |
| `preorder_backlog` | `SUM(backorder)`, 동일 테이블 | 소요량에 **더해집니다** |
| `inbound_in_window` | `calc.py:398-408` | 확정 상태만 대상이며, ETA가 `coverage_end` 이전이거나 같아야 합니다. ETA가 없는 입고는 절대 인정되지 않습니다 |
| `safety_stock` | `calc.py:415` | `z × error_used × coverage_demand`. 고정된 커버 일수 규칙이 아니라, 해당 SKU의 실측 WAPE에 그 SKU가 감당해야 할 수요를 곱한 값입니다 |
| `lead_time_weeks` | 파라미터 | `coverage_weeks`와 `coverage_end`를 통해서만 반영됩니다 |
| 초안 | `calc.py:238-240`, `types.ts:22-34` | **절대 차감하지 않습니다.** 초안은 확정이 아니므로, 이를 인정하면 이미 누군가 조치한 SKU를 정확히 과소 주문하게 됩니다 |

로트 크기, MOQ, 컨테이너 단위 반올림은 어디에도 존재하지 않습니다.

기본값은 `calc.py:63-83`에 있으며 `types.ts:142-147`에 반영되어 있습니다.

```python
DEFAULT_PARAMS = {
    "lead_time_weeks": 8,          # supplier + transit
    "review_period_weeks": 1,      # how often orders are placed
    "service_z": 1.0,              # ≈84% service level; 1.65 ≈ 95%
    "best_seller_demand_share": 0.50,
    "stockout_horizon_days": 30,
}
```

`best_seller_demand_share`는 UI에 노출되지 않습니다.

### 2.3 내역

`calc.py:612-660`이 상세 페이지의 계산 내역을 구성합니다. 각 줄은 `Sign`을 가지며 `+1`은 가산, `−1`은 감산, `0`은 합계, `None`은 참고용입니다. 저장된 `recommended_order_qty`가 재계산보다 우선합니다.

### 2.4 커버 일수와 예상 재고 소진일

`_days_until_consumed`, `calc.py:124-146`. 예측 곡선을 주 단위로 따라가며, 재고가 소진되는 주 안에서는 선형 보간을, 지평 종료 이후에는 지평의 평균 속도를 사용합니다. 예측이 0인 주는 건너뜁니다.

`calc.py:435-457`에서 적용됩니다.

```python
in_time = inb > 0 and np.isfinite(eta) and eta <= t_on_hand
```

입고는 재고가 바닥나기 전에 도착하는 경우에만 커버를 연장합니다. `recommended_order_qty`는 커버리지 구간 내의 모든 입고를 인정하며, `calc.py:528-544`는 그 차이를 `supply_gap_days`로, 지금 주문하는 것이 도움이 되는지를 `gap_closable_by_order`로 설정합니다.

### 2.5 우선순위 사다리

`calc.py:502-511`. 세 가지 상태가 있고, 먼저 일치하는 것이 적용되며, 번호가 낮은 것이 앞섭니다.

| 순위 | 레이블 | 조건 |
|---|---|---|
| 1 | Preorder | `preorder_backlog > 0` |
| 2 | No Stock | `available_inventory <= 0` |
| 99 | Routine | 그 외 전부 |

순위 3 "Best Seller"는 `calc.py:485-501`에서 제거되었고, 지금은 `calc.py:472-483`의 독립 불리언 값으로, 최근 판매 수량의 50%를 차지하는 최소 SKU 집합을 뜻합니다.

작업 목록 순서는 `calc.py:555-559`에 있으며 우선순위, best seller, 주문 수량 순이고, 안정성을 위해 `mergesort`를 사용합니다.

주의: 클라이언트 측 사본은 대소문자를 구분합니다. `action-list-table.tsx:169-184`의 `PRIORITY`는 정확히 그 문자열을 키로 사용하므로, `"No stock"`은 아무 경고 없이 Routine 스타일로 넘어갑니다.

### 2.6 신뢰도 등급

`src/planning/reliability.py:34-53`. SKU별 WAPE는 `outputs/reports/ml_accuracy_by_sku.csv`에서 가져오며, 2026-08-14에 다시 실행되었습니다. 갱신 주기는 §3.7에 있습니다.

| 등급 | 경계 | 기호 |
|---|---|---|
| good | ≤ 0.15 | ●●● |
| fair | ≤ 0.30 | ●●○ |
| poor | 그 이상 | ●○○ |
| none | 미측정 | ○○○ |

### 2.7 `error_basis`와 수요 밴드 대체 규칙

`calc.py:299-355`. 대부분의 SKU에는 실측 오차가 없습니다. `error_basis`는 실제로 사용된 오차를 제공한 단계를 나타냅니다.

| 순서 | 출처 | `error_basis` |
|---|---|---|
| 1 | 자체 실측 WAPE | `"measured"` |
| 2 | 수요 밴드 중앙값 | `"demand band"` |
| 3 | 세그먼트 중앙값 | `"segment median"` |
| 4 | 전체 중앙값 | `"overall median"` |
| 5 | 0.0 | 없음 |

밴드는 주간 수량 기준이며 좌측 폐구간입니다. `calc.py:35`.

```python
ERROR_BAND_EDGES = [0.0, 2.0, 4.0, 6.0, 10.0, float("inf")]
MIN_BAND_MEASURED = 5
```

**근거.** 각 경계값은 오차 곡선이 실제로 측정된 지점입니다. Pooled WAPE는 밴드 안에서는 평평하고 밴드 사이에서 계단식으로 변합니다.

주의: 밴드 구분은 `recent_units / 4`가 아니라 프로파일러의 직전 13주 주간 평균인 `recent_mean`을 기준으로 합니다. `src/profile.py`의 모든 임계값이 그 구간을 사용합니다. 4주를 기준으로 밴드를 나누면 강등 규칙상 해당 없음이어야 할 SKU 11개가 주당 2단위 미만 구간에 들어갔습니다.

실측 SKU가 5개 미만인 밴드는 물량 축을 따라 가장 가까운 신뢰 가능한 밴드에서 값을 빌려오며, 동률일 때는 물량이 낮은 쪽을 택합니다.

**근거.** 세그먼트 중앙값을 쓰면 [0,2) 밴드에 속한 33개 SKU의 여유분이 0.038만큼 줄었습니다. 그 밴드는 0.357로 예측이 가장 어렵고 SKU가 4개로 가장 얇습니다.

`PROMOTED_ERROR_FALLBACK = 0.24`는 2026-08-12에 제거되었습니다. 기록은 `calc.py:42-61`에 남아 있습니다.

**미해결 결함.** `types.ts:89`는 `"measured" | "promoted cohort" | "segment median" | string`을 선언하고, `reliability-card.tsx:71-75`는 `"promoted cohort"`를 매핑합니다. `"demand band"`와 `"overall median"`은 `basisLabel[errorBasis] ?? errorBasis`를 통과하여 한국어 로케일에서 번역되지 않은 영어로 표시됩니다.

### 2.8 `forecast_runs_high` 단서 조항

`calc.py:376-389`. 예측값이 최근 4주 평균의 1.5배 이상이면서 **동시에** 그보다 20단위 이상 높은 SKU에 플래그를 붙입니다. 두 조건 모두 필요합니다.

표시만 될 뿐 조치로 이어지지는 않습니다. 수량은 여전히 모델 값을 사용합니다. 안내 문구는 주문 카드 위에 배치됩니다.

### 2.9 Not-forecast 섹션

`calc.py:663-795`. 카탈로그의 약 87%와 최근 물량의 5분의 1을 차지하는 intermittent SKU가 표시됩니다. 소속 기준은 예측 파일이 아니라 planning 테이블에서의 부재입니다(`calc.py:711-718`).

**근거.** 예측 파일을 기준으로 삼았을 때, 실행 이후 강등된 SKU 15개가 두 섹션 모두에서 누락되었습니다. planning 테이블을 기준으로 하면 두 섹션이 하나의 분할을 이룹니다.

예측이 없으면 `recommended_order_qty`, 커버리지 수요, 안전 재고, WAPE, 재고 소진일도 존재하지 않습니다. 이 섹션은 직전 13주 속도, 그로부터 계산한 커버 일수, 그리고 커버가 리드 타임보다 짧을 때의 `reorder_signal`을 제공합니다. "13w demand", "per week", "cover" 제목은 다른 어디에서도 쓰이지 않습니다.

| 경우 | 규칙 |
|---|---|
| 판매 이력이 전혀 없음 | `days_of_cover`는 무한대가 아니라 `NaN`입니다. 0으로 나눈 결과는 "절대 소진되지 않음"으로 읽히기 때문입니다 |
| 재고 기록 자체가 없음 | 필드는 0이 아니라 공란입니다. "기록 없음"은 "0이라고 기록됨"과 다릅니다 |

섹션을 열 때만 조회합니다. SKU 수가 대략 7배이기 때문입니다.

### 2.10 Run Forecast 버튼

`run-forecast.tsx:67`, 기본적으로 접혀 있으며 출처 표시줄 바로 아래에 있습니다.

`POST /api/planning/run-forecast?horizon=N` → FastAPI `api/main.py:185-240` → 자체 프로세스 그룹에서 `scripts/ml_prepare_data.py --force --horizon N`을 실행하는 백그라운드 스레드.

| 동작 | 상세 |
|---|---|
| 작업 유형 | `create_job("forecast")`는 `/planning/prepare-data`와 유형을 공유하므로 동시 요청은 **409**를 받습니다. 두 작업은 동일한 파일을 씁니다 |
| 폴링 | `GET /api/forecast/status/{jobId}`를 2초마다 호출하며, `jobId`와 `status`를 키로 삼으므로 스스로 멈춥니다 |
| 재조회 | 성공한 경우에만 수행하여 `trained_through`가 갱신되도록 합니다. 실패 시에도 재조회하면 목록이 변하지 않은 수치를 마치 실패가 반영된 것처럼 보여주게 됩니다 |
| 취소 | 버튼은 없으나 `/cancel-forecast/{job_id}`는 여전히 존재합니다 |

주의: 진행률 표시줄은 명시되지 않은 계약에 의존합니다. `run-forecast.tsx:61`.

```ts
const m = /Step (\d)\/4/.exec(line);
if (m) seen = Math.max(seen, Number(m[1]));
```

스크립트 자체의 stdout을 정규식으로 파싱하므로, `ml_prepare_data.py`에서 해당 접두사를 바꾸면 표시줄이 조용히 깨집니다. 개수 세기가 아니라 `Math.max`를 쓰므로, 앞선 단계가 진행률을 되돌릴 수는 없습니다.

### 2.11 상세 페이지

`/planning/action-list/[sku]`. 계산 내역 형태의 주문 카드, 구간별 백테스트 표가 포함된 신뢰도 카드, 지표 행, 차트 두 개로 구성됩니다.

예측이 없는 SKU의 경우 `/planning/sku/{id}`는 서로 다른 세 가지 상세 메시지와 함께 404를 반환합니다(`api/main.py:502-520`). 예측 실행에는 포함되었으나 강등된 경우, 프로파일링은 되었으나 한 번도 예측되지 않은 경우, 알 수 없는 SKU인 경우입니다. 페이지는 판매 이력만 제공하는 `/planning/sku/[sku]/history`로 대체 동작합니다.

`/planning/sku/{id}`는 전체 SKU 목록과 해당 행의 위치도 함께 반환하며, 약 10 KB 정도로 Prev/Next를 위한 두 번째 요청을 줄여 줍니다.

타당한 구간 표시는 존재하지 않습니다. `api/main.py:631-635`에 그 이유가 기록되어 있습니다. 안전 재고가 더하는 것과 동일한 오차만큼 커버리지 수요를 늘린 것이어서, 그 상한이 곧 추천값이었기 때문입니다.

## 3. Forecast Validation

**목적.** 예측을 신뢰할지 판단하는 것. 여섯 개의 근거 섹션을 위에서 아래로 읽도록 구성했으며, 그 순서 자체가 논증입니다.

**Route.** `src/app/planning/forecast-validation/page.tsx`. 컴포넌트는 `forecast-validation/`이 아니라 `src/components/planning/validation/`에 있습니다.

### 3.1 렌더 순서상의 섹션

번호는 `VALIDATION_SECTIONS`(`section-heading.tsx:34-41`)에서, 렌더 순서는 `validation-content.tsx:214-297`에서 정해집니다. 새 섹션은 렌더 위치에 맞춰 그 배열에 들어갑니다.

| # | id | 제목 | 주장 |
|---|---|---|---|
| 01 | `comparison` | 모델 대 스프레드시트 | 동일한 구간 전반에서 모델이 pooled WAPE 기준으로 V1을 능가하며, 지는 셀도 모두 표시됩니다 |
| 02 | `demand` | 수요의 구성 | 범위. 모델이 대변하는 SKU가 무엇이고 그 물량이 얼마인지를 다룹니다. 이 섹션의 어떤 것도 예측이 아닙니다 |
| 03 | `trajectory` | 수요 대 예측 | 동일한 근거를 시간축으로 본 것. 오차가 언제, 어느 방향으로 발생했는지 |
| 04 | `over-time` | 실제로 제공된 예측의 성능 | 결과가 알려지기 전에 발행된 예측 |
| 05 | `outliers` | SKU 단위 분해 | pooled 개선이 폭넓은 것인지, 소수의 큰 성과에 기댄 것인지 |
| 06 | `final-test` | 최종 테스트 구간 | 격리된 결과 |

| 조건부 렌더 | 조건 |
|---|---|
| 03 생략, 오류 카드 없음 | 추세 조회가 실패하거나 예측값을 반환하지 않는 경우 |
| 04 억제 | 이력 저장소가 비어 있어, 같은 내용을 말하는 패널이 둘이 되는 경우 |

**설계 기준**, `comparison-section.tsx:1-14`. 그리드는 스프레드시트가 이기는 셀을 포함해 모든 세그먼트와 모든 구간을 표시합니다.

모델 버전은 페이로드에서 읽으므로, 새 모델이 나와도 프런트엔드 변경은 필요하지 않습니다.

### 3.2 섹션별 데이터 출처

| 섹션 | 출처 | 계산 방식 | 시계 |
|---|---|---|---|
| 01, 05 | `ml_accuracy.csv`, `ml_accuracy_by_sku.csv` | 저장됨, `scripts/ml_accuracy_report.py`가 생성 | 고정 |
| 02 | `load_sales()` | 요청 시 실시간 | 실시간 |
| 03, 04 | `src/ml/serving/history.py` | 실시간, 누적 저장소에서 | 실시간 |
| 06 | `outputs/reports/final_test.json` | 저장됨 | 고정 |

| 시계 | 읽는 대상 | 기대 동작 |
|---|---|---|
| 실시간 | 화요일 cron이 다시 쓰는 `data/processed` | 매주 변합니다 |
| 고정 | `ML_DATA_SNAPSHOT`이 지정하는 스냅샷 | 변하지 않습니다. 그 값어치는 모델 버전 간 비교 가능성에 있습니다 |

응답의 `basis`가 섹션별로 어느 쪽인지 알려 줍니다. `demand-patterns`는 자체 값을 함께 전달하므로 섹션 02는 스스로 날짜를 표시합니다. `meta.accuracy_computed`는 `ml_accuracy_meta.json`의 `run_at`입니다. 매니페스트가 없으면 파일 mtime이 사용되고 `basis.accuracy.computed_at_is_mtime`이 이를 알리며, 단서 조항으로 렌더링됩니다.

### 3.3 드리프트 점검

`src/planning/provenance.py`. 고정 섹션은 측정에 사용된 스냅샷이 실제로 제공되는 것과 유사한 동안에만 유효합니다.

| 플래그 | 조건 | 이름 비교로 잡히는가? |
|---|---|---|
| `snapshot_stale` | 리포트의 스냅샷이 `ML_DATA_SNAPSHOT`과 다름 | 예 |
| `population_stale` | 예측 가능 코호트가 채점 당시로부터 5% 넘게 이동함 | 아니오 |

`population_stale`은 2026-08-11처럼 같은 이름으로 다시 프로파일링된 모집단과 함께 스냅샷이 제자리에서 다시 잘린 경우를 포괄합니다.

주의: 드리프트 측정에서 카탈로그의 87%를 차지하고 한 번도 예측되거나 채점되지 않는 intermittent 꼬리는 제외됩니다. 2026-08-11 재프로파일링은 카탈로그 전체 기준으로 3.8%, 예측 가능 코호트 기준으로 42.1%로 나타납니다.

이동량이 절반이 되지는 않습니다. smooth에서 강등된 SKU는 코호트 안에서 교체되는 것이 아니라 코호트를 벗어나기 때문입니다.

드리프트는 보고될 뿐 복구되지 않습니다. `/health`가 이를 담고, cron이 경고를 출력하며, 페이지가 배너를 표시합니다. §3.7을 참조합니다.

| 페이로드 필드 | 상세 |
|---|---|
| `comparison.grid` | `(segment, window)` 셀마다 모델 버전이 동적 키로 들어가며, 그래서 `ValidationCell`에 인덱스 시그니처가 있습니다 |
| `windows` | 컷오프 기준 시간순 정렬. 알파벳순으로 하면 Dec-Feb, Mar-May, Oct-Dec로 읽혀 추세를 거꾸로 해석하게 만듭니다 |
| `outliers.rows` | 채점된 전체 풀로 약 572행. `top_n`은 페이지가 표시하는 개수이지 전송되는 개수가 아닙니다 |

### 3.4 차트

| 차트 | 표시 내용 |
|---|---|
| 주간 수요, 누적 영역 | 예측 SKU와 intermittent 꼬리의 대비 |
| 파레토 집중 곡선 | 누적 SKU 대비 누적 수요, 균등 수요 기준선 포함 |
| 수요 대 예측 궤적 | 실제 주간 수량, 선택한 lead에서의 저장 실행 예측값, 전방 지평, V1 |

위 세 가지에는 Plotly를 사용합니다. 차트처럼 보이지만 차트가 아닌 요소가 두 개 더 있습니다.

| 요소 | 구현 |
|---|---|
| SKU별 델타 히스토그램, `outliers-section.tsx` | ±1에서 잘라내고 초과분을 레이블이 붙은 버킷으로 표시하는 CSS 크기 조정 div |
| 실행별 오차 추세, `over-time-section.tsx` | 표 셀 안에 그린 막대 |

없는 것:

| 없는 것 | 이유 |
|---|---|
| 예측 구간 없음 | v11은 점 예측만 내보냅니다(`has_intervals: false`) |
| V1은 전방 지평에만 표시 | 이력 저장소는 모델의 예측값을 보관하며, V1은 실행마다 다시 계산됩니다 |

### 3.5 국제화

두 페이지의 문자열은 한국어를 먼저 두는 `pick(ko, en)`을 사용합니다. `validation/` 아래 어떤 파일도 키 기반 `t(key)` 사전을 사용하지 않습니다.

새 검증 섹션을 추가할 때의 체크리스트:

1. `"use client"`
2. `useI18n()`
3. 모든 리터럴을 `pick`으로 감쌉니다
4. 번역된 `title`과 `description`을 `SectionHeading`에 전달합니다
5. 해당 섹션을 렌더 순서상의 위치에 맞추어 `[ko, en]` 레이블 쌍과 함께 `VALIDATION_SECTIONS`에 추가합니다

참고: 한국어는 피연산자 순서를 바꾸므로, 보간 시 템플릿이 두 인자 안에 각각 중복됩니다. `pick`은 `title=` 툴팁과 Plotly 트레이스의 `name:` 필드에도 값을 공급하므로, 번역된 텍스트가 차트 범례까지 전달됩니다.

### 3.6 최종 테스트 데이터와 패널

`outputs/reports/final_test.json`은 `scripts/ml_41_final_test.py`가 작성하며, 이 스크립트는 덮어쓰기를 거부합니다.

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

2026-08-14부터 추적됩니다. `.gitignore:38`이 `outputs/reports/*`를 제외하고, 그 아래 줄들이 특정 파일을 이름으로 다시 포함시킵니다. 이 테스트는 일회성이며, 재실행은 복구 수단이 아닙니다.

`api/main.py`의 `_final_test_payload()`는 `scores`와 출처 필드를 변경 없이 그대로 전달합니다. 두 개 필드는 파생되며, 그 덕분에 웹 앱은 모델 버전을 이름으로 알 필요가 없습니다.

| 파생 필드 | 내용 |
|---|---|
| `methods` | `scores`의 어느 키가 모델이고, 스프레드시트이고, 구조적 베이스라인인지. 역할은 소거법으로 정해집니다. `V1`과 `baseline`은 실행기가 기록하는 고정 이름이고, 나머지가 테스트 대상 모델입니다 |
| `comparisons` | `<model>_vs_<other>` 블록을 목록으로 펼친 것으로, 각 항목이 무엇을 비교하는지를 함께 담습니다. 파일에서는 버전 이름이 포함된 키이므로, 하드코딩하지 않고서는 TypeScript로 기술할 수 없습니다 |

파일이 없거나 읽을 수 없으면 `evaluated: false`가 반환되고 섹션은 "아직 표시할 내용 없음"으로 렌더링됩니다. `types.ts`는 `FinalTest`를 구별 유니언으로 정의하므로, 미평가 상태에서 결과가 반쪽만 렌더링될 수는 없습니다. `cutoff`는 양쪽 분기에 모두 존재하며 섹션 설명과 `ModelCard`가 읽습니다.

`final-test-section.tsx`는 같은 크기의 판정 패널 두 개를 나란히 렌더링합니다. 유의성은 플래그로 전달되지 않고, 95% 구간이 0을 배제하는지 여부로 컴포넌트 안에서 계산됩니다.

보정은 렌더링되지 않습니다. 실행기는 pooled WAPE와 부트스트랩만 기록하며, 이 구간의 편향은 `ML_FORECAST_DESIGN.md` §4.35에 있습니다. 페이로드는 `has_bias: false`를 담고, 패널은 해당 수치가 어디에 있는지를 안내합니다. 이를 렌더링하려면 `ml_41_final_test.py`가 그 값을 기록해야 합니다.

**알려진 오래된 주석.** `section-heading.tsx:12-15`는 이 페이지가 "아직 의도적으로 주장하지 않는 것"으로 끝난다고 설명하지만 더 이상 사실이 아니며, 20줄 아래의 `VALIDATION_SECTIONS` 배열과 어긋나는 섹션 순서를 나열합니다. 배열이 기준입니다.

### 3.7 정확도 리포트 갱신 주기

`scripts/ml_accuracy_report.py`는 스냅샷이 다시 잘리거나 프로파일러가 바뀔 때 갱신이 필요합니다. 마지막 실행은 2026-08-14, 스냅샷 `2026-08-03-v2` 기준입니다.

**근거.** 주간 cron에 넣지 않은 이유는 이 스크립트가 고정 스냅샷을 읽기 때문입니다. 매주 실행하면 세 구간을 재학습해서 동일한 바이트를 다시 쓰게 됩니다. 트리거는 모집단 변화이며, 이는 cron이 예측할 수 없고 드리프트 점검이 보고합니다.

**근거.** 아예 자동화하지 않은 이유는 출력이 설계 문서와 소수점 셋째 자리까지 대조되고 경영 제안서에 인용되기 때문입니다. 결정 없이 재생성하면 공표된 수치가 소리 없이 바뀝니다.

## 4. 미해결 결함

어느 것도 크래시를 일으키지 않습니다. 어느 것도 정확도에 관해 독자를 오도하지 않습니다.

| # | 위치 | 내용 |
|---|---|---|
| 2 | `types.ts:89`, `reliability-card.tsx:71-75` | `error_basis` 값이 최신이 아니며, 한국어 로케일에서 번역되지 않은 문자열이 표시됩니다. §2.7 |
| 4 | `run-forecast.tsx:61` | `ml_prepare_data.py`와의 명시되지 않은 `Step N/4` stdout 계약. §2.10 |
| 5 | `SkuForecastsService.getForecastBounds()` | `/api/forecast/bounds`가 삭제되면서 고아가 되었습니다. 통과하는 테스트가 아직 남아 있어, 관리되는 것처럼 보이는 죽은 코드입니다 |
