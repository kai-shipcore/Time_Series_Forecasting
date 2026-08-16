# Deployment

The forecast API shares a server with Demand Pilot. systemd manages this service.

**Rationale.** Forecast requests are proxied server-side, so no colleague needs Python, a virtualenv or a copy of the data.

Since 2026-08-07 port 8000 is also directly reachable, protected by `FORECAST_API_TOKEN`. The proxy is the normal path; direct access serves local development against real data.

## 1. Ownership

| What | Owner | Arrives by |
|---|---|---|
| Code | GitHub Actions, on push to `main` | `rsync` from the checkout |
| Data | `scripts/run_forecast_cron.sh`, on the server | Written in place, nothing to transfer |

Nothing owns both. The weekly run is server cron under `coverland`, writing into `/opt/coverland-forecast-api/data/processed` where the service reads.

The deploy excludes `data/` and `outputs/`. Under `rsync --delete` an exclude means both "do not upload" and "do not destroy". Without it every deploy wipes the server's data.

Note: the excludes are not `.gitignore`, which ignores `data/raw/`, `data/processed/` and most of `outputs/` while tracking `data/snapshots/`, `data/dev_seed/` and three CSVs under `outputs/reports/`. The deploy rule is ownership: no tracked file under `data/` reaches the server.

## 2. Reaching the deployed service from a laptop

| Option | Configuration | Data |
|---|---|---|
| 1. Local service | `setup.cmd` (Windows) or `python3 scripts/setup_local.py`, then `AI_SERVICE_URL=http://localhost:8000` | Committed fixture. No database, no server access. Best for planning-page work |
| 2. SSH tunnel | Tunnel left open, `AI_SERVICE_URL=http://localhost:8000`, `FORECAST_API_TOKEN` at the server's value | Live, through the SSH session. Nothing exposed publicly |
| 3. Direct | `AI_SERVICE_URL` at the server, `FORECAST_API_TOKEN` at the server's value | Live. No tunnel, local service or repo copy |

Option 2:

```bash
ssh -L 8000:127.0.0.1:8000 coverland@144.24.40.252
```

Option 3:

```
AI_SERVICE_URL=http://144.24.40.252:8000
FORECAST_API_TOKEN=<the server's value>
```

Warning: under option 2 the app still treats `localhost` as a server it may start. uvicorn spawns only when nothing answers, so a dropped tunnel silently serves local data. Check `trained_through` on the Action List.

Warning: option 3 makes `FORECAST_API_TOKEN` the entire perimeter. Prefer option 2 when SSH is open.

Note: `AI_SERVICE_URL` may be set in either `.env` or `.env.local`, and both are gitignored, so two machines can disagree indefinitely. Check with `findstr /C:"AI_SERVICE_URL" .env .env.local` before editing.

### Network posture

| Control | State |
|---|---|
| Host packet filtering | None. `iptables` INPUT policy ACCEPT with no REJECT; firewalld and ufw inactive |
| Oracle VCN security list | The only network control, and already permits 8000 |
| Bind address | The systemd unit binds `--host 0.0.0.0`. Binding `127.0.0.1` accepts only connections originating on the box, seen by a caller as "connection refused" |

Warning: `FORECAST_API_TOKEN` is the entire application perimeter. `api/main.py` enforces it on every path except `/health`, and only when set. Unset, the four POST endpoints are open to the internet, two of them spawning a pipeline subprocess. See §9.

## 3. Running it locally

For planning-page or service work. The seed is frozen at the week of 2026-07-20; the figures are not to be acted on.

```bash
git clone https://github.com/kai-shipcore/Time_Series_Forecasting.git
cd Time_Series_Forecasting

python3 scripts/setup_local.py            # macOS / Linux
setup.cmd                                 # Windows
```

`setup_local.py`:

| Aspect | Behaviour |
|---|---|
| Builds | Virtualenv, dependencies, seeds `data/processed/`, writes `.env`, then checks the data files are present and both databases answer |
| Re-runs | Skips work already done. Safe after a pull, and the right move when `requirements.txt` changes |
| Interpreter | System Python, because it creates the virtualenv and cannot live inside one. Imports only the standard library |
| Database settings | Derived from a nearby `Commerce_Integration` checkout's `.env`. Override with `--commerce-env <path>`. Without one it writes a template with blank keys, which is not fatal because the seeded data needs no database |

`setup.cmd` covers three Windows interpreter names:

| Name | Behaviour |
|---|---|
| `py -3` | Tried first |
| `python` | Real, or the Microsoft Store execution alias: a stub that opens the Store and exits successfully without running anything |
| `python3` | Absent there |

It verifies `python` by running code through it.

By hand instead:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/seed_dev_data.py
```

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\seed_dev_data.py
```

Call the interpreter directly. The default Windows execution policy blocks `Activate.ps1`.

Demand Pilot starts the service when `AI_SERVICE_URL` is localhost, so opening a Planning page suffices. To start it by hand:

```
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`GET /health` then reports `ready: true` with an empty `missing_required`.

The seed copies four in-repository files into `data/processed/`, refuses to overwrite an existing one, so it is safe on the cron machine, and needs no `.env` or database access.

Note: when the pages cannot reach the server, check `FORECAST_SERVER_DIR` in the local `.env` first. Copied from a colleague it points at their checkout. Unset, the app finds the checkout itself.

## 4. GitHub secrets

Repository **Settings > Secrets and variables > Actions**.

| Name | Example | Notes |
|---|---|---|
| `DEPLOY_HOST` | The Demand Pilot server | |
| `DEPLOY_USER` | `coverland` | Needs write access to `DEPLOY_PATH` |
| `DEPLOY_SSH_KEY` | Private key | |
| `DEPLOY_PATH` | `/opt/coverland-forecast-api` | Kept separate from the Demand Pilot checkout |
| `DEPLOY_PORT` | `22` | Omit when the server uses 22 |

## 5. Server setup, once

1. Create the directory.

```bash
sudo mkdir -p /opt/coverland-forecast-api
sudo chown -R coverland:coverland /opt/coverland-forecast-api
```

2. Create `/opt/coverland-forecast-api/.env`, which the deploy never overwrites. Eleven values as of 2026-08-13:

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

3. Install the unit.

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

4. Allow the deploy user to restart this one unit with passwordless sudo.

5. Set these in the server's `Commerce_Integration/.env`.

```
AI_SERVICE_URL="http://localhost:8000"
FORECAST_API_TOKEN=<same value as above>
```

| Key | Requirement |
|---|---|
| `DB_*` and `COMMERCE_DB_*` | Both sets required. `src/planning/inventory.py` opens them together and returns nothing if either is missing, degrading inventory to the sample snapshot instead of erroring. `_engine` returns `None` on any failure including a bad password, so a typo surfaces as "SAMPLE inventory data" on the Action List. Copy from the `.env` of the machine currently running the service |
| `FORECAST_SERVER_DIR` | Unset in production. A second supervisor racing systemd's `Restart=always` to bind port 8000 turns a clean restart into two half-alive servers. Unset, the app reports the outage and points at `systemctl` |
| `LLM_*` (four) and `FORECAST_SELF_URL` | Obsolete. They configured the AI assistant on the deleted Demand Forecast page. Remove from the server's `.env` |

Forecasts come from files. On-hand stock, preorder backlog and confirmed inbound are read live.

## 6. The weekly run

On the server, as the `coverland` user:

```
0 10 * * 2 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
```

Warning: Tuesday (day 2) is required. A week runs Tuesday through Monday, labelled by the Monday it ends on. A Monday run could use only the week ended the previous Monday, putting every forecast seven days staler.

Three places encode the week convention, one decision:

| Location | Encoding |
|---|---|
| `src/clean.py` | `closed="right"` on the W-MON grouper |
| `src/weeks.py:last_complete_week` | Steps back an extra week on Mondays |
| The crontab | Day 2 (Tuesday) |

If the forecast looks a week stale, check all three before changing any. See design doc §4.30.

What the script does (detail in `DATA_AND_PIPELINE.md` §5):

1. Runs `scripts/ml_prepare_data.py --force`: velocity sync, ingest, clean, profile, then `ml_forward_forecast.py --snapshot live`.
2. Asks the service whether it can still serve, exiting non-zero if not, so cron mails a failure.
3. Backs up the accumulating history.

It writes `ml_forward_forecasts` and appends a row per SKU to `ml_forecast_history`, the source for the Action List and Forecast Validation.

Warning: `--snapshot live` is not optional. Without it the script defaults to `config.ML_DATA_SNAPSHOT`, the pinned copy that keeps recorded evaluation figures from drifting, and weekly runs reproduce the same forecast while appearing to work.

Artifacts stage beside `data/processed/` and commit with `os.replace` only after the forecast succeeds, so an interrupted run leaves last week's files served.

Note: 10:00 UTC was 3am Pacific when set up. The server stays on UTC, so Pacific wall-clock time moves an hour at each DST transition.

Note: this is the only weekly job. A second cron previously ran `push_data_to_server.sh` on a developer's Mac; remove that line with `crontab -e` there.

## 7. Pushing data by hand

Not part of the weekly cycle. For data produced off the server (a laptop forecast re-run, a fresh `scripts/export_inventory_snapshot.py`) and wanted there before the next Tuesday.

```bash
scripts/push_data_to_server.sh
```

Configure it in this repo's `.env`:

```
FORECAST_DEPLOY_HOST=...
FORECAST_DEPLOY_USER=coverland
FORECAST_DEPLOY_PATH=/opt/coverland-forecast-api
FORECAST_DEPLOY_KEY=~/.ssh/id_ed25519     # a path to a key, not the key; optional
```

It pushes only the nine files `src/planning/data.py` reads, about 1.5 MB, then asks the server whether it can serve, exiting non-zero if not. The runner needs a key authorised on the server, which `scripts/verify_deployment.sh` also requires.

## 8. Checking it

`GET /health` reports both liveness and data readiness:

```json
{
  "status": "ok",
  "ready": true,
  "missing_required": [],
  "repo_root": "/opt/coverland-forecast-api"
}
```

It returns 200 even when data is missing; `ready` is the separate question. If `repo_root` is not `DEPLOY_PATH`, the running service is serving a different checkout. The planning pages' status indicator shows the same information.

## 9. Runbook: the forecast API is not responding

### Step 1: is it actually down?

```bash
bash scripts/_test_port_8000.sh          # from any machine that is not the server
```

Expect JSON from `/health` and **401** from `/segmentation`. The 401 is a pass: the port is reachable and the token enforced.

Warning: run from a laptop, never from the server. Checking from the box passes even when the service is bound to loopback.

### Step 2: read the failure

| Symptom | Meaning |
|---|---|
| Hangs, then times out | Packets are being dropped. Oracle Cloud VCN security list |
| Connection refused, immediately | Packets arrive, nothing is listening on the public interface. The service is down, or bound to `127.0.0.1` |
| 200 on `/segmentation` with no token | Worse than an outage. The API is unauthenticated on a public port. Fix `FORECAST_API_TOKEN` before anything else |

A refusal means the network is fine; only a hang implicates a firewall.

### Step 3: ask the server

GitHub → Actions → **Server diagnostics** → Run workflow. Read-only, no password, about seven seconds. It reports port 8000's owner, the unit state, every uvicorn process with its start time, and whether the API answers.

### The three failure modes seen so far

| Mode | Symptom and check | Fix |
|---|---|---|
| 1. Something else is holding port 8000 | Diagnostics shows `127.0.0.1:8000`, unit state `activating` instead of `active`, journal saying `[Errno 98] address already in use` every eight seconds | `bash scripts/_kill_squatter.sh`; the unit binds within a few seconds. Cause: the deploy's own fallback branch, fixed at source (BACKLOG 21). Starting a server by hand on the box produces the same state |
| 2. The service is bound to loopback | Connection refused from outside, works on the box. `systemctl show coverland-forecast-api -p ExecStart --value` must end `--host 0.0.0.0 --port 8000 --workers 1` | The correct unit is version-controlled at `deploy/coverland-forecast-api.service`. Copy it to `/etc/systemd/system/`, `daemon-reload`, restart |
| 3. Reachable but useless | `/health` returns 200 with `"ready": false`; the process is alive with no data, so every planning page 500s. Check `crontab -l \| grep run_forecast_cron` (expect day 2, Tuesday) and the mtimes under `data/processed/` | Usually the weekly cron has not run or failed |

### Automated monitoring

| Check | Behaviour |
|---|---|
| Hourly | `.github/workflows/api-reachable.yml` curls the public URL from a GitHub runner, off the network and so testing the path a laptop uses. Fails on unreachable, on `ready: false`, and on auth not being enforced |
| Every deploy | `ci-cd.yml` asserts after restarting that the unit is `active` on two checks five seconds apart, and that `0.0.0.0:8000` is bound. Without it `systemctl restart` exits 0 while the service fails to start, so a deploy reports green having shipped code that never ran |

Note: GitHub disables scheduled workflows after 60 days without repository activity; the Actions tab then offers a re-enable button.

Note: the failure email goes to whoever last committed to the workflow file, the wrong person since the handover. Change it deliberately.

### Security posture

| Control | State |
|---|---|
| Host packet filtering | None |
| Oracle VCN security list | The only network control |
| `FORECAST_API_TOKEN` | The only application control |
| `/health` | Outside the token check by design |
| Four POST endpoints | Behind the token, including `/planning/run-forecast` and `/planning/prepare-data`, which each spawn a pipeline run |

The hourly check tests for the token being unset.
