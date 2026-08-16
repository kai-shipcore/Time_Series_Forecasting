# 데이터와 파이프라인

숫자가 어디에서 오는지, 그리고 매주 무엇이 실행되는지를 다룹니다.

## 1. 주 단위 규칙

| 항목 | 값 |
|---|---|
| 기간 | 화요일부터 월요일까지, 양끝 포함 |
| 라벨 | 해당 주가 끝나는 월요일이며, pandas 주기는 `W-MON` |
| 예시 | `2026-07-13`은 7월 7일 화요일부터 7월 13일 월요일까지를 포함합니다 |

기간(`src/clean.py`의 `closed="right"`), `W-MON` 라벨, 화요일 cron은 하나의 결정이며 함께 움직입니다.

주의: cron 실행 요일을 바꾸지 않은 채 이 규칙만 바꾸면 파이프라인이 조용히 한 주를 잃습니다. 이 때문에 별개의 결함이 두 차례 발생했습니다.

수요는 네 가지 주문 유형 `sales`, `preorder`, `ttm`, `ttm_preorder`를 모두 포함하며, 출고된 주가 아니라 주문이 **접수된** 주에 귀속됩니다. 출시 선주문은 갓 출시된 SKU의 짧은 이력을 지배할 수 있습니다.

## 2. 입력

| 소스 | 내용 | 단위 |
|---|---|---|
| `shipcore.fc_velocity_link_snapshot_forecast` | 전체 주문 이력이며, 모든 채널을 포함하고 날짜 상한이 없습니다. **원천 데이터** | 주문 라인당 한 행 |
| `data/processed/sales_clean.parquet` | 위 데이터를 주간 ingest가 집계한 결과입니다. 그리드가 완전하며 0으로 채워집니다 | SKU별 주별 한 행 |
| `data/processed/sku_profiles.csv` | 세그먼트 라벨과 학습 시작일 | SKU당 한 행 |
| `ecommerce_data.coverland_inventory_by_warehouse` | 보유, 할당, 이월 주문 | SKU별 창고별 한 행 |
| `shipcore.fc_container_items`와 `shipcore.fc_containers` 조인 | ETA가 있는 확정 및 초안 입고 | 컨테이너 라인당 한 행 |

재고 테이블과 컨테이너 테이블은 Action List 용도로만 실시간 조회합니다. 예측 입력이 아닙니다.

주의: velocity 테이블은 두 개가 있으며 서로 대체할 수 없습니다. `..._snapshot_forecast`는 상한이 없고 예측에 사용됩니다. `..._snapshot`은 120일 상한이 있으며 Velocity UI 페이지에만 사용됩니다.

## 3. 출력

| 대상 | 내용 |
|---|---|
| `shipcore.ml_forward_forecasts` | 현재 13주 예측이며, SKU별 미래 주별 한 행 |
| `shipcore.ml_forecast_history` | 제공된 모든 예측이 누적되며, 실행별 SKU별 대상 주별 한 행 |
| `data/processed/ml_forward_forecasts.parquet` | 파일 사본이며, 읽기 대체 경로 |
| `data/processed/ml_forecast_history.parquet` | 누적 이력의 파일 사본 |
| `data/history_backups/` | 이력의 날짜별 사본이며, 최근 12개를 보관 |

두 `ml_` 테이블은 같은 컬럼 정의를 공유합니다. 키는 `(model_version, week_of, unique_id, ds)`이며, `week_of`는 해당 실행이 학습에 사용한 주이고 `ds`는 예측 대상 주입니다. 모델 버전은 파일 이름이 아니라 컬럼에 담기므로 여러 버전이 공존합니다.

**근거.** 누적 이력은 다시 만들 수 없지만, 그 외의 모든 것은 데이터베이스에서 재생성됩니다. 이 이력은 결과를 알기 전에 무엇을 예측했는지를 기록합니다. 과거 모델을 과거 컷오프에 다시 돌리면 백테스트가 되며, 이는 더 약한 근거입니다. 그래서 테이블, 파일 사본, 주간 백업을 함께 둡니다.

참고: `shipcore.fc_forward_forecasts`와 `shipcore.fc_forecast_history`는 2026-08-13 값으로 고정되어 있으며, 은퇴한 statsforecast 계열에 속합니다. 현재 예측이 아닙니다.

## 4. 파이프라인

`scripts/ml_prepare_data.py`가 전체 순서를 실행합니다.

```bash
.venv/bin/python scripts/ml_prepare_data.py --force --horizon 13
```

| 단계 | 동작 | 기록 대상 |
|---|---|---|
| 1/4 sync | 주문 소스에서 velocity snapshot을 갱신합니다 | `fc_velocity_link_snapshot_forecast` |
| 2/4 ingest + clean | 주문을 가져와 주간 그리드로 집계합니다 | `sales_clean.parquet` |
| 3/4 profile | 각 SKU를 버킷과 이력 길이로 분류합니다 | `sku_profiles.csv` |
| 4/4 forecast | LightGBM v11을 실행하고 forward와 history를 기록합니다 | `ml_forward_forecasts`, `ml_forecast_history` |

| 플래그 | 동작 |
|---|---|
| `--force` | 필수입니다. 파이프라인은 기본적으로 운영 파일 덮어쓰기를 거부합니다 |
| `--snapshot live` | 주간 실행에 필수입니다. 이 값이 없으면 ML 스크립트가 고정된 snapshot을 기본값으로 사용하므로, 매주 같은 예측이 생성되면서도 정상 동작하는 것처럼 보입니다 |
| `--horizon` | 하한은 13이며, UI 옵션과 엔드포인트 시그니처(`Query(default=13, ge=13, le=104)`)에서는 강제되지만 스크립트에서는 강제되지 않습니다. 각 실행은 해당 학습 주에 저장된 예측을 대체하므로, 더 짧은 실행은 완전한 snapshot을 덮어써 버립니다 |

### 4.1 실패한 실행은 안전합니다

산출물은 `data/processed/` 옆의 스테이징 디렉터리에 기록되며, 예측이 성공한 뒤에만 `os.replace`로 제자리에 옮겨집니다. 크래시, 단계 실패, SSH 세션 끊김이 발생해도 지난주 파일이 손상 없이 남아 계속 제공됩니다. 스크립트는 0이 아닌 코드로 종료하므로 실패 시 cron이 메일을 보냅니다.

주의: `os.replace`는 같은 파일 시스템 안에서만 원자적입니다. 스테이징 디렉터리는 반드시 `data/processed/`의 형제 디렉터리여야 하며, `/tmp`여서는 안 됩니다.

참고: 오래된 예측은 화면에서 정상 예측과 구분되지 않습니다. Action List의 `trained_through`를 달력과 대조해 확인합니다.

## 5. 주간 실행

서버에 `coverland` 사용자로 등록된 cron 항목 하나입니다.

```
0 10 * * 2 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
```

이 스크립트는 `ml_prepare_data.py --force`를 실행하고, `/health`를 호출해 `ready: true`를 확인한 뒤, 누적 이력을 `data/history_backups/`로 복사하며 최근 12개를 보관합니다.

주의: 화요일(day 2)이어야 합니다. 월요일 L로 라벨링된 주는 월요일 L 하루 내내 아직 열려 있으므로, 월요일에 실행하면 그 전 월요일에 닫힌 주까지만 사용할 수 있습니다.

참고: 10:00 UTC는 설정 당시 태평양 시간 오전 3시였습니다. 서버는 UTC를 유지하므로, 태평양 기준 실제 시각은 서머타임 전환마다 한 시간씩 어긋납니다.

## 6. 두 개의 고정값

| 고정값 | 위치 | 고정 대상 |
|---|---|---|
| `ML_FINAL_TEST_CUTOFF` | `config.py:44`, 현재 `2026-05-04` | 각 평가 구간이 포함하는 주 |
| `ML_DATA_SNAPSHOT` | `config.py:67`, 현재 `2026-08-03-v2` | 그 주 안에 들어 있는 값 |

기록된 결과를 재현하려면 둘 다 필요합니다. 주간 갱신은 늦게 등록되는 주문에 따라 최근 실적을 수정하기 때문입니다. 둘 중 하나라도 앞당기면 프로젝트의 모든 수치가 다시 기준화되며, 하나를 앞당긴다고 다른 하나가 함께 움직이지는 않습니다.

평가 입력은 `data/processed/`가 아니라 `ML_DATA_SNAPSHOT`이 지정한 snapshot에서 가져오므로, 주간 갱신이 기록된 결과를 움직일 수 없습니다.

## 7. 실행을 놓친 뒤 따라잡기

`data_freshness`가 `ok: false`를 보고할 때, 또는 화요일 cron이 완료되지 않은 주 이후에 적용됩니다. 이 절차는 운영 쪽만 진행시킵니다.

주의: `ML_DATA_SNAPSHOT`은 건드리지 않습니다. `outputs/reports/final_test.json`은 측정에 사용한 snapshot을 기록하고 있고, 러너는 이 파일 덮어쓰기를 거부하며, 해당 구간은 다시 실행할 수 없습니다.

1. 제공 중인 데이터와 예측을 갱신하십시오.

   ```bash
   cd ~/Documents/Time_Series_Forecasting
   .venv/bin/pip install -r requirements.txt     # lightgbm is not installed by default
   .venv/bin/python scripts/ml_prepare_data.py --force
   ```

   | 확인 항목 | 기대값 |
   |---|---|
   | 출력의 `Training through` | 가장 최근의 완전한 W-MON 라벨 |
   | 예측된 SKU 수 | 약 340 |
   | `sales_clean.parquet`의 최신 `ds` | `Training through`과 일치 |

   SKU 개수가 결정적인 지표입니다. 340에서 크게 벗어난 개수는 프로파일러가 실행되지 않았거나 임계값이 바뀌었음을 뜻합니다.

2. snapshot을 다시 잘라 냈거나 프로파일러가 바뀐 경우에만 정확도 리포트를 다시 실행하십시오. 이 리포트는 고정된 snapshot을 읽으므로, 1단계는 이 리포트에 영향을 주지도 않고 이를 대신하지도 않습니다.

   ```bash
   .venv/bin/python scripts/ml_accuracy_report.py
   ```

   커밋하기 전에 출력된 표를 `OVERVIEW_KO.md` §6과 대조하십시오.

3. 파일들을 함께 커밋하십시오. 매니페스트는 CSV와 함께 이동해야 하며, 그러지 않으면 드리프트 점검이 모든 배포를 날짜 확인 불가로 보고합니다.

   ```bash
   git add outputs/reports/ml_accuracy.csv \
           outputs/reports/ml_accuracy_by_sku.csv \
           outputs/reports/ml_accuracy_meta.json
   git commit -m "Re-run the accuracy report against snapshot 2026-08-03-v2"
   ```

4. 두 점검이 모두 통과하는지 확인하십시오.

   ```bash
   curl -s http://127.0.0.1:8000/health | python3 -m json.tool | grep -A3 -E "data_freshness|accuracy_report"
   ```

   둘 다 `"ok": true`로 표시되고, Forecast Validation에서 02번 섹션 위의 주황색 줄과 01번 섹션 위의 배너가 모두 사라집니다. 둘 중 하나라도 남아 있으면 `detail` 문자열이 어느 단계가 반영되지 않았는지 알려 줍니다.

## 8. 헬스 체크

```bash
curl -s http://144.24.40.252:8000/health | python3 -m json.tool
```

`/health`는 토큰 검사 바깥에 있으므로, 토큰이 잘못되어도 응답합니다.

| 필드 | 의미 |
|---|---|
| `ready` | `true`는 필요한 데이터 파일이 모두 존재한다는 뜻입니다. `false`는 서비스는 살아 있으나 제공할 것이 없다는 뜻이며, 모든 계획 페이지가 500을 반환합니다 |
| `missing_required` | `ready`가 false일 때 어떤 파일이 없는지 |
| `commit` | 실제로 서비스 중인 리비전입니다. `main`의 최신 커밋과 비교합니다. 배포되지 않은 푸시와 포트를 잡지 못한 배포를 잡아냅니다 |
| `data_freshness` | 제공 중인 주가 가장 최근의 완전한 주인지 여부입니다. `ok: false`는 화면의 예측이 오래되었다는 뜻입니다 |
| `accuracy_report` | 고정된 정확도 리포트가 여전히 제공 중인 모집단을 설명하는지 여부입니다. `ok: false`는 Forecast Validation의 01번과 05번 섹션이 이미 달라진 집단을 설명하고 있다는 뜻입니다 |

**근거.** `data_freshness`와 `accuracy_report`는 의도적으로 `ready`에서 제외했습니다. 오래된 예측은 없는 답이 아니라 틀린 답이며, 503을 반환하면 정상 동작하는 화면까지 내려가고 이미 실행 중인 서버를 상대로 로컬 자동 시작 경로가 작동하기 때문입니다. `ready`가 본문에 있는 이유는 "서버 없음"과 "데이터 없는 서버"의 해결 방법이 서로 다르기 때문입니다.

`data_freshness`가 비정상일 때는 해결 방법을 스스로 알려 줍니다.

```jsonc
"data_freshness":  { "ok": false, "detail": "served data ends 2026-08-03 but the last complete week is 2026-08-10, 1 week(s) behind",
                     "fix": "scripts/ml_prepare_data.py --force" }
```

감시 장치는 세 가지입니다. 매시간 실행되는 GitHub Actions 워크플로가 있으며, 토큰으로 보호된 엔드포인트에서 `200`이 나오면 API가 인터넷에 열려 있다는 뜻이므로 실패로 처리합니다. 주간 cron은 실패 시 메일을 보냅니다. 그리고 계획 페이지의 서비스 상태 표시가 있습니다.

API 모듈을 변경한 뒤에는 다음을 실행합니다.

```bash
.venv/bin/python scripts/check_route_parity.py --probe
```

이 스크립트는 데이터베이스와 네트워크 없이도 API가 제공해야 할 라우트를 여전히 제공하는지 확인합니다. 라우터 트리를 순회하며 애플리케이션의 실제 매칭 로직을 구동하고, 실행할 때마다 음성 대조군을 함께 넣습니다.

**근거.** FastAPI는 포함된 라우터를 하나의 불투명한 객체로 저장하므로, 라우트 개수 비교로는 아무것도 비교되지 않습니다.

푸시하기 전에는 다음을 실행합니다.

```bash
.venv/bin/python scripts/verify_repo.py          # 8 static checks
.venv/bin/python scripts/smoke_planning_api.py   # every planning endpoint, in-process
```

스모크 테스트는 정적 점검이 잡지 못하는 계획 페이지 실패를 잡아냅니다.

## 9. 문제 해결

| 증상 | 유력한 원인 | 우선 확인 |
|---|---|---|
| 페이지가 예측 서버에 접근하지 못함 | 서비스 중단, 또는 `AI_SERVICE_URL`이 잘못됨 | **로컬 서버가 실행되지 않는 머신에서** `curl /health` |
| `/health`는 응답하지만 `ready`가 false | cron이 실행되지 않았거나 실패함 | `crontab -l \| grep run_forecast_cron` 확인 후 `logs/forecast_cron.log` |
| 모든 계획 페이지가 500을 반환함 | 같은 원인이며, 제공할 데이터가 없음 | `/health`의 `missing_required` |
| 예측이 주마다 변하지 않음 | cron이 커밋 전에 실패함. 실패한 실행은 설계상 지난주 파일을 그대로 두므로 밖에서는 정상으로 보임 | `logs/forecast_cron.log`, 그리고 `trained_through`를 달력과 대조 |
| `commit`이 `main`의 최신 커밋이 아님 | 배포에 실패한 푸시, 또는 포트를 점유한 오래된 프로세스 | 해당 푸시의 GitHub Actions 실행 |
| Action List에 "SAMPLE inventory data"가 표시됨 | 데이터베이스 자격 증명이 잘못됨 | `DB_*`와 `COMMERCE_DB_*`가 **모두** 설정되어야 하며, 일부만 설정하면 조용히 성능이 저하됨 |

주의: Demand Pilot은 설정된 서비스가 응답하지 않으면 로컬 예측 서비스를 자동으로 시작하므로, 잘못 설정된 `AI_SERVICE_URL`도 정상으로 보입니다. 확실한 판별 방법은 로컬 서버가 없는 머신에서 `curl`을 실행하는 것입니다.

주의: `AI_SERVICE_URL`은 `.env`와 `.env.local` 중 어느 쪽에서든 올 수 있습니다. 수정 전에 어느 쪽인지 확인하고, 수정 후에는 재시작합니다.

## 10. 배포

전체 참조는 `DEPLOYMENT_KO.md`에 있습니다.

| 규칙 | 세부 내용 |
|---|---|
| `main`에 푸시하면 코드가 배포됩니다 | GitHub Actions를 통해 이루어지며, 수동 단계는 없습니다 |
| 데이터는 배포되지 않습니다 | 배포에서 `data/`, `outputs/`, `logs/`를 제외하므로, 서버 데이터의 소유권은 cron이 단독으로 갖습니다 |
| 두 저장소가 함께 바뀔 때는 순서가 중요합니다 | Commerce를 먼저 배포하거나 둘을 함께 배포합니다. Python 쪽은 기존 페이지가 호출하는 엔드포인트를 해제합니다 |

배포 후에는 `/health`의 `commit`이 `main`의 최신 커밋과 일치합니다.

## 11. 유지보수 일정

| 시점 | 조치 |
|---|---|
| 매주, 자동 | 화요일 cron입니다. cron이 실패 메일을 보내면 로그를 확인합니다 |
| 각 배포 후 | `/health`의 `commit`이 `main`의 최신 커밋과 일치하는지 확인 |
| API 코드를 변경한 후 | `scripts/check_route_parity.py --probe` |
| 제공 모델 버전 **또는 모집단**이 바뀔 때 | `scripts/ml_accuracy_report.py`로 `outputs/reports/ml_accuracy.csv` 재생성 |
| 연 2회 | cron의 UTC 시각이 서머타임 전환마다 태평양 시간 대비 한 시간씩 어긋납니다 |
| 의존성이 바뀔 때 | PyPI가 아니라 **배포 호스트에서** `pip freeze`로 버전을 고정합니다 |

**정확도 리포트 갱신 조건.** 독스트링은 제공 모델 버전 변경만 언급합니다. 모집단이 바뀔 때도 갱신이 필요하며, 프로파일링이나 임계값 변경이 여기에 해당합니다. `SCREENS_KO.md` §3.7 참조.

**의존성 고정.** `requirements.txt`의 모든 버전은 정확히 고정되어 있습니다. ML 관련 고정 다섯 개가 존재하는 이유는 결과를 소수점 셋째 자리에서 비교하기 때문이며, 이는 고정되지 않은 solver 업데이트가 만드는 드리프트보다 더 미세한 단위입니다. 서비스 관련 고정은 특정되지 않은 FastAPI 버전에서 라우트 등록에 발생한, 검증할 수 없는 변경 이후에 도입되었습니다. 두 그룹 모두 완화하지 않습니다.

## 12. 결합된 제약

결합된 요소 가운데 하나만 바꾸고 나머지를 그대로 두면 어딘가가 조용히 망가집니다.

| 결합 | 제약 |
|---|---|
| 화요일 cron, `W-MON` 라벨, 화요일부터 월요일까지의 기간 | 하나의 결정 |
| `ML_FINAL_TEST_CUTOFF`와 `ML_DATA_SNAPSHOT` | 재현성을 위해 둘 다 필요 |
| `ml_prepare_data.py`와 주간 cron | cron의 유일한 역할은 이 스크립트를 호출하는 것이며, Action List의 Run Forecast 버튼이 호출하는 것도 같은 스크립트입니다. 스크립트 하나에 트리거가 둘이고, 같은 순서를 복제한 두 번째 사본은 없습니다 |
| 양쪽의 `FORECAST_API_TOKEN` | Python 서비스와 Demand Pilot이 일치해야 하며, 그렇지 않으면 `/health`를 제외한 모든 요청이 401을 반환합니다 |
| `run-forecast.tsx`의 진행 표시줄과 `ml_prepare_data.py`의 stdout | 이 컴포넌트는 스크립트 출력에서 `Step N/4`를 정규식으로 찾습니다. 해당 접두사의 이름을 바꾸면 진행 표시줄이 조용히 망가집니다. 이 계약은 코드에 명시되어 있지 않습니다 |
