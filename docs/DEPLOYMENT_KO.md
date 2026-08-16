# 배포

예측 API는 Demand Pilot과 서버를 공유합니다. 이 서비스는 systemd가 관리합니다.

**근거.** 예측 요청은 서버 측에서 프록시되므로, 동료 누구도 Python이나 virtualenv, 데이터 사본을 갖출 필요가 없습니다.

2026-08-07부터 포트 8000에도 직접 접근할 수 있으며, `FORECAST_API_TOKEN`으로 보호됩니다. 정상 경로는 프록시이고, 직접 접근은 실제 데이터를 대상으로 한 로컬 개발용입니다.

## 1. 소유 관계

| 대상 | 소유자 | 전달 방식 |
|---|---|---|
| 코드 | GitHub Actions, `main`에 push할 때 | 체크아웃에서 `rsync` |
| 데이터 | 서버의 `scripts/run_forecast_cron.sh` | 제자리에 기록되며 전송할 것이 없음 |

둘 다 소유하는 주체는 없습니다. 주간 실행은 `coverland` 계정의 서버 cron이며, 서비스가 읽는 `/opt/coverland-forecast-api/data/processed`에 기록합니다.

배포는 `data/`와 `outputs/`를 제외합니다. `rsync --delete` 아래에서 제외는 "업로드하지 않음"과 "파괴하지 않음"을 동시에 의미합니다. 이것이 없으면 배포할 때마다 서버의 데이터가 지워집니다.

참고: 이 제외 목록은 `.gitignore`가 아닙니다. `.gitignore`는 `data/raw/`, `data/processed/`와 `outputs/`의 대부분을 무시하면서 `data/snapshots/`, `data/dev_seed/`, `outputs/reports/` 아래 CSV 세 개는 추적합니다. 배포 규칙의 기준은 소유 관계이며, `data/` 아래의 추적 대상 파일은 서버에 도달하지 않습니다.

## 2. 노트북에서 배포된 서비스에 접근하기

| 옵션 | 설정 | 데이터 |
|---|---|---|
| 1. 로컬 서비스 | `setup.cmd`(Windows) 또는 `python3 scripts/setup_local.py` 실행 후 `AI_SERVICE_URL=http://localhost:8000` | 커밋된 픽스처. 데이터베이스도 서버 접근도 필요 없습니다. planning 페이지 작업에 가장 적합합니다 |
| 2. SSH 터널 | 터널을 열어 둔 채 `AI_SERVICE_URL=http://localhost:8000`, `FORECAST_API_TOKEN`은 서버 값과 동일하게 | 실시간, SSH 세션을 경유합니다. 외부에 노출되는 것은 없습니다 |
| 3. 직접 접근 | `AI_SERVICE_URL`을 서버 주소로, `FORECAST_API_TOKEN`을 서버 값으로 | 실시간. 터널도, 로컬 서비스도, 저장소 사본도 필요 없습니다 |

옵션 2:

```bash
ssh -L 8000:127.0.0.1:8000 coverland@144.24.40.252
```

옵션 3:

```
AI_SERVICE_URL=http://144.24.40.252:8000
FORECAST_API_TOKEN=<the server's value>
```

주의: 옵션 2에서도 앱은 여전히 `localhost`를 자신이 시작해도 되는 서버로 취급합니다. uvicorn은 응답하는 것이 없을 때만 생성되므로, 터널이 끊기면 조용히 로컬 데이터가 제공됩니다. Action List에서 `trained_through`를 확인해야 합니다.

주의: 옵션 3에서는 `FORECAST_API_TOKEN`이 보안 경계 전부입니다. SSH를 쓸 수 있으면 옵션 2를 우선합니다.

참고: `AI_SERVICE_URL`은 `.env`나 `.env.local` 어느 쪽에도 설정할 수 있고 둘 다 gitignore 대상이므로, 두 대의 장비가 무기한 어긋난 상태로 남을 수 있습니다. 편집 전에 `findstr /C:"AI_SERVICE_URL" .env .env.local`로 확인해야 합니다.

### 네트워크 상태

| 통제 수단 | 상태 |
|---|---|
| 호스트 패킷 필터링 | 없음. `iptables` INPUT 정책은 ACCEPT이고 REJECT가 없으며, firewalld와 ufw는 비활성 |
| Oracle VCN 보안 목록 | 유일한 네트워크 통제 수단이며, 이미 8000을 허용 |
| 바인드 주소 | systemd 유닛은 `--host 0.0.0.0`으로 바인드합니다. `127.0.0.1`로 바인드하면 해당 장비에서 시작된 연결만 받으며, 호출자에게는 "connection refused"로 보입니다 |

주의: `FORECAST_API_TOKEN`이 애플리케이션 보안 경계 전부입니다. `api/main.py`는 `/health`를 제외한 모든 경로에서 이를 강제하되, 값이 설정된 경우에만 그렇습니다. 설정되지 않으면 POST endpoint 네 개가 인터넷에 열리며, 그중 둘은 파이프라인 서브프로세스를 생성합니다. §9를 참조합니다.

## 3. 로컬에서 실행하기

planning 페이지 작업이나 서비스 작업을 위한 것입니다. 시드는 2026-07-20 주로 고정되어 있으며, 그 수치를 근거로 조치해서는 안 됩니다.

```bash
git clone https://github.com/kai-shipcore/Time_Series_Forecasting.git
cd Time_Series_Forecasting

python3 scripts/setup_local.py            # macOS / Linux
setup.cmd                                 # Windows
```

`setup_local.py`:

| 측면 | 동작 |
|---|---|
| 구성 | virtualenv, 의존성, `data/processed/` 시드, `.env` 작성. 그 뒤 데이터 파일이 존재하는지와 두 데이터베이스가 응답하는지 확인 |
| 재실행 | 이미 끝난 작업은 건너뜁니다. pull 이후에도 안전하며, `requirements.txt`가 바뀌었을 때 적절한 조치입니다 |
| 인터프리터 | 시스템 Python. virtualenv를 생성하는 주체이므로 그 안에서 실행될 수 없습니다. 표준 라이브러리만 import합니다 |
| 데이터베이스 설정 | 가까이 있는 `Commerce_Integration` 체크아웃의 `.env`에서 가져옵니다. `--commerce-env <path>`로 재정의할 수 있습니다. 하나도 없으면 키가 빈 템플릿을 작성하는데, 시드 데이터는 데이터베이스가 필요 없으므로 치명적이지 않습니다 |

`setup.cmd`는 Windows의 인터프리터 이름 세 가지를 처리합니다.

| 이름 | 동작 |
|---|---|
| `py -3` | 가장 먼저 시도 |
| `python` | 실제 인터프리터이거나, Microsoft Store 실행 별칭. 후자는 Store를 열고 아무것도 실행하지 않은 채 성공으로 종료되는 스텁입니다 |
| `python3` | 해당 환경에는 없음 |

`python`은 실제로 코드를 실행해 보는 방식으로 검증합니다.

직접 수행하는 경우:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/seed_dev_data.py
```

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\seed_dev_data.py
```

인터프리터는 직접 호출합니다. Windows의 기본 실행 정책이 `Activate.ps1`을 차단하기 때문입니다.

`AI_SERVICE_URL`이 localhost이면 Demand Pilot이 서비스를 시작하므로, Planning 페이지를 여는 것만으로 충분합니다. 직접 시작하려면 다음과 같이 합니다.

```
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

그러면 `GET /health`가 `missing_required`가 빈 상태로 `ready: true`를 보고합니다.

시드는 저장소 안의 파일 네 개를 `data/processed/`로 복사하고, 이미 존재하는 파일은 덮어쓰지 않으므로 cron 장비에서도 안전하며, `.env`나 데이터베이스 접근이 필요 없습니다.

참고: 페이지가 서버에 접근하지 못할 때는 로컬 `.env`의 `FORECAST_SERVER_DIR`를 먼저 확인해야 합니다. 동료에게서 복사한 값이라면 그 사람의 체크아웃을 가리킵니다. 설정하지 않으면 앱이 스스로 체크아웃을 찾습니다.

## 4. GitHub 시크릿

저장소의 **Settings > Secrets and variables > Actions**.

| 이름 | 예시 | 비고 |
|---|---|---|
| `DEPLOY_HOST` | Demand Pilot 서버 | |
| `DEPLOY_USER` | `coverland` | `DEPLOY_PATH`에 대한 쓰기 권한이 필요합니다 |
| `DEPLOY_SSH_KEY` | 개인 키 | |
| `DEPLOY_PATH` | `/opt/coverland-forecast-api` | Demand Pilot 체크아웃과 분리해서 둡니다 |
| `DEPLOY_PORT` | `22` | 서버가 22를 쓰면 생략합니다 |

## 5. 서버 초기 설정, 1회

1. 디렉터리를 생성합니다.

```bash
sudo mkdir -p /opt/coverland-forecast-api
sudo chown -R coverland:coverland /opt/coverland-forecast-api
```

2. `/opt/coverland-forecast-api/.env`를 생성합니다. 배포는 이 파일을 절대 덮어쓰지 않습니다. 2026-08-13 기준으로 값 11개입니다.

```
FORECAST_API_TOKEN=<same value as Demand Pilot's FORECAST_API_TOKEN>

DB_HOST=...
DB_PORT=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...

COMMERCE_DB_HOST=...
COMMERCE_DB_PORT=...
COMMERCE_DB_NAME=...
COMMERCE_DB_USER=...
COMMERCE_DB_PASSWORD=...
```

3. 유닛을 설치합니다.

```ini
[Unit]
Description=Coverland Forecast API
After=network.target

[Service]
User=coverland
WorkingDirectory=/opt/coverland-forecast-api
EnvironmentFile=/opt/coverland-forecast-api/.env
ExecStart=/opt/coverland-forecast-api/.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo tee /etc/systemd/system/coverland-forecast-api.service < coverland-forecast-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now coverland-forecast-api
```

4. 배포 사용자가 이 유닛 하나만 비밀번호 없는 sudo로 재시작할 수 있도록 허용합니다.

5. 서버의 `Commerce_Integration/.env`에 다음을 설정합니다.

```
AI_SERVICE_URL="http://localhost:8000"
FORECAST_API_TOKEN=<same value as above>
```

| 키 | 요구 사항 |
|---|---|
| `DB_*`와 `COMMERCE_DB_*` | 두 묶음 모두 필요합니다. `src/planning/inventory.py`가 이들을 함께 열고 어느 한쪽이라도 없으면 아무것도 반환하지 않으며, 오류를 내는 대신 재고를 샘플 스냅샷으로 격하합니다. `_engine`은 잘못된 비밀번호를 포함한 모든 실패에 대해 `None`을 반환하므로, 오타는 Action List에 "SAMPLE inventory data"로 드러납니다. 현재 서비스를 실행 중인 장비의 `.env`에서 복사합니다 |
| `FORECAST_SERVER_DIR` | 운영 환경에서는 설정하지 않습니다. 두 번째 감독 프로세스가 systemd의 `Restart=always`와 포트 8000 바인딩을 두고 경합하면, 깨끗한 재시작이 반쯤 살아 있는 서버 두 개로 바뀝니다. 설정하지 않으면 앱이 장애를 보고하고 `systemctl`을 안내합니다 |
| `LLM_*`(네 개)와 `FORECAST_SELF_URL` | 사용하지 않습니다. 삭제된 Demand Forecast 페이지의 AI 어시스턴트를 설정하던 값입니다. 서버의 `.env`에서 제거합니다 |

예측값은 파일에서 옵니다. 현재고, 선주문 잔량, 확정 입고는 실시간으로 읽습니다.

## 6. 주간 실행

서버에서 `coverland` 사용자로 실행합니다.

```
0 10 * * 2 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
```

주의: 화요일(day 2)이어야 합니다. 한 주는 화요일부터 월요일까지이며, 그 주가 끝나는 월요일로 이름이 붙습니다. 월요일에 실행하면 직전 월요일에 끝난 주까지만 쓸 수 있어, 모든 예측이 7일 더 오래된 상태가 됩니다.

주 단위 규약은 세 곳에 인코딩되어 있지만 결정은 하나입니다.

| 위치 | 인코딩 |
|---|---|
| `src/clean.py` | W-MON grouper의 `closed="right"` |
| `src/weeks.py:last_complete_week` | 월요일에는 한 주 더 뒤로 이동 |
| crontab | day 2(화요일) |

예측이 한 주 오래되어 보이면, 무엇이든 바꾸기 전에 세 곳을 모두 확인해야 합니다. 설계 문서 §4.30을 참조합니다.

스크립트가 하는 일은 다음과 같습니다(상세 내용은 `DATA_AND_PIPELINE_KO.md` §5).

1. `scripts/ml_prepare_data.py --force`를 실행합니다. 속도 동기화, 수집, 정제, 프로파일링 후 `ml_forward_forecast.py --snapshot live`를 수행합니다.
2. 서비스에 아직 제공 가능한지 묻고, 아니면 0이 아닌 코드로 종료하여 cron이 실패 메일을 보내도록 합니다.
3. 누적 이력을 백업합니다.

이 스크립트는 `ml_forward_forecasts`를 쓰고 `ml_forecast_history`에 SKU당 한 행을 추가하며, 이 데이터가 Action List와 Forecast Validation의 원본입니다.

주의: `--snapshot live`는 선택 사항이 아닙니다. 이것이 없으면 스크립트는 기록된 평가 수치가 흔들리지 않도록 고정해 둔 사본인 `config.ML_DATA_SNAPSHOT`을 기본값으로 사용하며, 주간 실행이 정상 동작하는 것처럼 보이면서 같은 예측을 반복합니다.

산출물은 `data/processed/` 옆에 스테이징되었다가 예측이 성공한 뒤에만 `os.replace`로 확정되므로, 중단된 실행은 지난주 파일이 계속 제공되는 상태로 남습니다.

참고: 10:00 UTC는 설정 당시 Pacific 기준 오전 3시였습니다. 서버는 UTC를 유지하므로, Pacific 벽시계 시각은 DST 전환마다 한 시간씩 이동합니다.

참고: 주간 작업은 이것 하나뿐입니다. 이전에는 개발자의 Mac에서 두 번째 cron이 `push_data_to_server.sh`를 실행했습니다. 그 줄은 해당 장비에서 `crontab -e`로 제거합니다.

## 7. 데이터를 수동으로 푸시하기

주간 주기의 일부가 아닙니다. 서버 밖에서 생성된 데이터(노트북에서 다시 실행한 예측, 새로 만든 `scripts/export_inventory_snapshot.py` 결과)를 다음 화요일 전에 서버에 반영하려는 경우를 위한 것입니다.

```bash
scripts/push_data_to_server.sh
```

이 저장소의 `.env`에서 설정합니다.

```
FORECAST_DEPLOY_HOST=...
FORECAST_DEPLOY_USER=coverland
FORECAST_DEPLOY_PATH=/opt/coverland-forecast-api
FORECAST_DEPLOY_KEY=~/.ssh/id_ed25519     # a path to a key, not the key; optional
```

`src/planning/data.py`가 읽는 파일 아홉 개, 약 1.5 MB만 푸시한 뒤 서버에 제공 가능한지 묻고, 아니면 0이 아닌 코드로 종료합니다. 실행 주체에는 서버에 등록된 키가 필요하며, `scripts/verify_deployment.sh` 역시 같은 키를 요구합니다.

## 8. 점검하기

`GET /health`는 서비스 생존 여부와 데이터 준비 상태를 함께 보고합니다.

```json
{
  "status": "ok",
  "ready": true,
  "missing_required": [],
  "repo_root": "/opt/coverland-forecast-api"
}
```

데이터가 없어도 200을 반환하며, `ready`가 별개의 질문입니다. `repo_root`가 `DEPLOY_PATH`가 아니면, 실행 중인 서비스가 다른 체크아웃을 제공하고 있는 것입니다. planning 페이지의 상태 표시기도 같은 정보를 보여 줍니다.

## 9. 런북: 예측 API가 응답하지 않음

### 1단계: 실제로 다운되었는가?

```bash
bash scripts/_test_port_8000.sh          # from any machine that is not the server
```

`/health`에서는 JSON을, `/segmentation`에서는 **401**을 기대합니다. 401은 정상 신호입니다. 포트에 도달했고 토큰이 강제되고 있다는 뜻입니다.

주의: 노트북에서 실행하고 서버에서는 절대 실행하지 않습니다. 서버 장비에서 확인하면 서비스가 루프백에 바인드되어 있어도 통과합니다.

### 2단계: 실패 양상을 읽는다

| 증상 | 의미 |
|---|---|
| 멈춰 있다가 타임아웃 | 패킷이 폐기되고 있습니다. Oracle Cloud VCN 보안 목록 |
| 즉시 connection refused | 패킷은 도달하지만 공용 인터페이스에서 아무것도 수신 대기하지 않습니다. 서비스가 내려갔거나 `127.0.0.1`에 바인드되어 있습니다 |
| 토큰 없이 `/segmentation`에 200 | 장애보다 나쁩니다. API가 공용 포트에서 인증 없이 열려 있습니다. 다른 무엇보다 먼저 `FORECAST_API_TOKEN`을 바로잡습니다 |

거부되었다는 것은 네트워크에 문제가 없다는 뜻이며, 방화벽이 관련된 경우는 멈춤뿐입니다.

### 3단계: 서버에 물어본다

GitHub → Actions → **Server diagnostics** → Run workflow. 읽기 전용이고 비밀번호가 필요 없으며 약 7초가 걸립니다. 포트 8000의 소유자, 유닛 상태, 시작 시각이 포함된 모든 uvicorn 프로세스, 그리고 API의 응답 여부를 보고합니다.

### 지금까지 확인된 세 가지 실패 유형

| 유형 | 증상과 확인 | 조치 |
|---|---|---|
| 1. 다른 무언가가 포트 8000을 점유 | 진단 결과에 `127.0.0.1:8000`이 나오고, 유닛 상태가 `active`가 아니라 `activating`이며, 저널에 8초마다 `[Errno 98] address already in use`가 기록됩니다 | `bash scripts/_kill_squatter.sh`를 실행하면 유닛이 몇 초 안에 바인드합니다. 원인은 배포 자체의 폴백 분기였고 소스에서 수정되었습니다(BACKLOG 21). 서버에서 직접 서버를 띄워도 같은 상태가 됩니다 |
| 2. 서비스가 루프백에 바인드됨 | 외부에서는 연결이 거부되고 서버 장비에서는 동작합니다. `systemctl show coverland-forecast-api -p ExecStart --value`의 끝이 `--host 0.0.0.0 --port 8000 --workers 1`이어야 합니다 | 올바른 유닛은 `deploy/coverland-forecast-api.service`에 버전 관리되어 있습니다. 이를 `/etc/systemd/system/`에 복사하고 `daemon-reload` 후 재시작합니다 |
| 3. 도달은 되지만 쓸모없음 | `/health`가 `"ready": false`와 함께 200을 반환합니다. 프로세스는 살아 있지만 데이터가 없어 모든 planning 페이지가 500을 냅니다. `crontab -l \| grep run_forecast_cron`(day 2, 화요일이어야 함)과 `data/processed/` 아래 mtime을 확인합니다 | 대개 주간 cron이 실행되지 않았거나 실패한 경우입니다 |

### 자동 모니터링

| 점검 | 동작 |
|---|---|
| 매시간 | `.github/workflows/api-reachable.yml`이 GitHub 러너에서 공용 URL을 curl합니다. 네트워크 밖에 있으므로 노트북이 쓰는 경로를 그대로 검증합니다. 도달 불가, `ready: false`, 인증 미강제 시 실패합니다 |
| 배포마다 | `ci-cd.yml`이 재시작 후 5초 간격의 두 번의 점검에서 유닛이 `active`인지, 그리고 `0.0.0.0:8000`이 바인드되어 있는지 확인합니다. 이것이 없으면 서비스가 기동에 실패해도 `systemctl restart`는 0으로 종료되므로, 실행된 적 없는 코드를 배포하고도 배포가 성공으로 보고됩니다 |

참고: GitHub은 저장소 활동이 60일간 없으면 예약된 워크플로를 비활성화하며, 그 뒤 Actions 탭에 재활성화 버튼이 나타납니다.

참고: 실패 알림 메일은 워크플로 파일을 마지막으로 커밋한 사람에게 가는데, 인수인계 이후로는 잘못된 사람입니다. 의도적으로 변경해야 합니다.

### 보안 상태

| 통제 수단 | 상태 |
|---|---|
| 호스트 패킷 필터링 | 없음 |
| Oracle VCN 보안 목록 | 유일한 네트워크 통제 수단 |
| `FORECAST_API_TOKEN` | 유일한 애플리케이션 통제 수단 |
| `/health` | 설계상 토큰 검사 대상 밖 |
| POST endpoint 네 개 | 토큰 뒤에 있으며, 각각 파이프라인 실행을 생성하는 `/planning/run-forecast`와 `/planning/prepare-data`를 포함 |

매시간 점검은 토큰이 설정되지 않은 상태를 탐지합니다.
