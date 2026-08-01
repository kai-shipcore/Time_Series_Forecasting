# Deployment

The forecast API runs on the same server as Demand Pilot, the Next.js app in
`Commerce_Integration`. Both are managed there: Next.js by pm2, this service by
systemd.

## Why the same box

Next.js proxies every forecast request from its own server process, never from
the browser, so a service on `127.0.0.1:8000` is reachable to it and to nothing
else. That gives three things for free. No public port and no firewall rule. No
CORS. And no colleague needing Python, a virtualenv, or a copy of the data,
because they open the deployed app and the forecast is already behind it.

The alternative, everyone running the service locally, is what produced the
original problem: a fresh clone has the code and none of the data, so the
service starts, answers a liveness check, and raises on every real request.

That is still the right answer for anyone who only wants to read a forecast. It
is the wrong answer for someone working on the planning pages themselves, who
needs the service in front of them. See "Running it locally" below: since
`data/dev_seed` was added, that no longer requires a database or a copy of
anyone's working tree.

## Two owners, no overlap

| What | Owner | Arrives by |
| --- | --- | --- |
| Code | GitHub Actions, on push to `main` | `rsync` from the checkout |
| Data | The weekly cron on the machine that runs the forecast | `scripts/push_data_to_server.sh` |

Nothing owns both, deliberately. The deploy excludes `data/` and `outputs/`,
which under `rsync --delete` means "do not upload" and equally "do not destroy".
Without those excludes every deploy would wipe the server's data and leave the
API serving 500s until the next Monday.

Those excludes are not a restatement of `.gitignore`, and reading them that way
is a mistake this document used to make. `.gitignore` covers `data/raw/`,
`data/processed/` and most of `outputs/`, but `data/snapshots/`, `data/dev_seed/`
and three CSVs under `outputs/reports/` are tracked on purpose. The deploy
excludes those paths anyway, because the rule is about ownership rather than
about what happens to be in git: the cron owns the server's data, and the deploy
declines to touch it. Adding a tracked file under `data/` therefore cannot reach
the server, which is exactly the property that makes a committed development
fixture safe.

## Running it locally

For working on the planning pages. Not for figures to act on: the seed is frozen
at the week of 2026-07-20, and the current forecast lives on the deployed app.

```bash
git clone <this repo> && cd Time_Series_Forecasting
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/seed_dev_data.py
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`GET /health` should then report `ready: true` with an empty `missing_required`.

No `.env` and no database access are needed for any of this. The seed copies
four files that are already in the repository into `data/processed/`, which is
gitignored and therefore the one thing a clone lacks. It refuses to overwrite an
existing `data/processed/`, so running it on the cron machine is safe.

Demand Pilot starts this service itself when `AI_SERVICE_URL` is localhost, so
in practice the last command is only needed to see startup errors directly. If
the planning pages report that the server has no data to read, the card names
the seed command.

## Required GitHub secrets

In the repository, open **Settings > Secrets and variables > Actions**.

| Name | Example | Notes |
| --- | --- | --- |
| `DEPLOY_HOST` | the Demand Pilot server | Same host the Next.js app deploys to |
| `DEPLOY_USER` | `coverland` | Needs write access to `DEPLOY_PATH` |
| `DEPLOY_SSH_KEY` | private key | |
| `DEPLOY_PATH` | `/opt/coverland-forecast-api` | Kept separate from the Next.js checkout |
| `DEPLOY_PORT` | `22` | Omit when the server uses 22 |

## Server setup, once

```bash
sudo mkdir -p /opt/coverland-forecast-api
sudo chown -R coverland:coverland /opt/coverland-forecast-api
```

Create `/opt/coverland-forecast-api/.env`. The deploy never overwrites it.

```
FORECAST_API_TOKEN=<same value as the Next.js app's FORECAST_API_TOKEN>

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

# The AI assistant on the Demand Forecast page. Without these, /chat returns
# 503 and that feature stops working, while everything else carries on.
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...

# Where the chat's tool loop calls back into this service. It must match the
# port in ExecStart below, or the assistant answers without ever reading the
# data it is being asked about.
FORECAST_SELF_URL=http://127.0.0.1:8000
```

Fifteen values. An earlier version of this document listed eleven and omitted
the three `LLM_*` ones, which is enough to start the service and lose the chat
feature without anything saying so.

Both prefixes are required, not one. `src/planning/inventory.py` opens `DB_*`
and `COMMERCE_DB_*` together and returns nothing if either is missing, so a
partial set degrades inventory to the sample snapshot silently rather than
erroring. Forecasts come from files, but on-hand stock, preorder backlog and
confirmed inbound are read live.

Copy the values from the `.env` of the machine that currently runs the service.
`_engine` returns `None` on any failure, including a bad password, so a typo
here surfaces as "SAMPLE inventory data" on the Action List rather than as a
connection error.

Install the service:

```ini
[Unit]
Description=Coverland Forecast API
After=network.target

[Service]
User=coverland
WorkingDirectory=/opt/coverland-forecast-api
EnvironmentFile=/opt/coverland-forecast-api/.env
ExecStart=/opt/coverland-forecast-api/.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
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

Bound to `127.0.0.1`, not `0.0.0.0`. On a shared box the service has no reason
to accept connections from anywhere but the Next.js process beside it, and
binding to the loopback enforces that whatever the firewall says.

Allow the deploy user to restart this one unit with passwordless sudo.

## Next.js side

On the server's `Commerce_Integration/.env`:

```
AI_SERVICE_URL="http://localhost:8000"
FORECAST_API_TOKEN=<same value as above>
```

Leave `FORECAST_SERVER_DIR` **unset in production.** It enables the app to start
the service itself when it is down, which is right on a laptop and wrong here:
systemd already supervises the process with `Restart=always`, and a second
supervisor racing it to bind port 8000 turns a clean restart into two half-alive
servers. Unset, the app reports the outage and points at `systemctl` instead.

## The weekly data push

After the Monday forecast, on the machine that produced the files:

```bash
scripts/push_data_to_server.sh
```

Configure it in this repo's `.env`:

```
FORECAST_DEPLOY_HOST=...
FORECAST_DEPLOY_USER=coverland
FORECAST_DEPLOY_PATH=/opt/coverland-forecast-api
FORECAST_DEPLOY_KEY=~/.ssh/id_ed25519     # optional
```

It pushes only the nine files `src/planning/data.py` reads, about 1.5 MB, rather
than the 19 MB of experiment plots and CV dumps in `outputs/` that the server has
no use for. Then it asks the server whether it can actually serve, and exits
non-zero if not. That exit code is the point: cron mails a failure on the Monday
it breaks, instead of a colleague finding a broken page on Thursday.

A crontab line, after the forecast run:

```
0 10 * * 1 cd /path/to/Time_Series_Forecasting && scripts/push_data_to_server.sh >> logs/push.log 2>&1
```

## Checking it

`GET /health` reports both liveness and data readiness:

```json
{
  "status": "ok",
  "ready": true,
  "missing_required": [],
  "repo_root": "/opt/coverland-forecast-api"
}
```

It returns 200 even when data is missing, because the process is alive and
answering that is what a health check is for. `ready` is the separate question.
`repo_root` is worth reading when something looks wrong: if it is not
`DEPLOY_PATH`, the running service is serving a different checkout.

The same information appears in the app, in the status indicator on the planning
pages, so a reader who is not on the server can still tell an outage from a data
problem.
