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
| Data | `scripts/run_forecast_cron.sh`, on the server itself | written in place, nothing to transfer |

The data row used to read differently, and the change matters. The weekly run
happens on the server, as cron for the `coverland` user, writing straight into
`/opt/coverland-forecast-api/data/processed` where the service reads. So no
machine outside the server is involved in keeping the forecast current, and none
has to be powered on for it to happen. `scripts/push_data_to_server.sh` still
exists and still works, but it is now the out-of-band path for data generated on
a laptop rather than the weekly step. See "Pushing data by hand" below.

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

## Why the deployed service cannot be used from a laptop

Asked often enough to belong here. The systemd unit binds
`--host 127.0.0.1 --port 8000`, so the service accepts connections only from
processes on that machine, which in practice means the Next.js process sitting
in front of it. There is no public port, no firewall rule and no CORS, and that
is the point rather than an oversight: the service holds credentials for both
databases and has no authentication of its own beyond a shared token.

So pointing a local `AI_SERVICE_URL` at the server does not work and should not
be made to work by opening a port. Three honest options, cheapest first.

**1. Use the deployed app.** For anyone who wants to read a forecast rather than
change one, `app.shipcore.com` already has it, and no local setup is involved at
all. This is the right answer far more often than it is taken.

**2. Tunnel to it.** For running the Next.js app locally against real, current
data without a Python environment or a copy of the database:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<server>
```

Leave that open, set `AI_SERVICE_URL=http://localhost:8000` and
`FORECAST_API_TOKEN` to the deployed value, and the local app talks to the
deployed service over the tunnel. Nothing is exposed publicly: the forwarding
lives inside an SSH session that ends when the terminal closes. Note the app
will still consider `localhost` a server it may start, which is harmless while
the tunnel is up, since it only spawns uvicorn when nothing answers.

**3. Run the service locally**, below. Needed only when the Python side itself
is being changed.

## Running it locally

For working on the planning pages or the service. Not for figures to act on: the
seed is frozen at the week of 2026-07-20, and the current forecast lives on the
deployed app.

One command, once, on a new machine:

```bash
git clone https://github.com/kai-shipcore/Time_Series_Forecasting.git
cd Time_Series_Forecasting

python3 scripts/setup_local.py            # macOS / Linux
setup.cmd                                 # Windows
```

`setup.cmd` exists because the interpreter has three plausible names on Windows
and only one of them works on any given machine. `python3` is a Unix convention
and is absent there, which is exactly the name every macOS instruction uses.
`python` may be real or may be the Microsoft Store's execution alias, a stub
that opens the Store and exits successfully without running anything, which is
the worst case because it looks like it worked. The batch file tries `py -3`,
then verifies `python` by running code through it rather than trusting that the
command resolved, and says what to install if neither answers.

It creates the virtualenv, installs the dependencies, seeds `data/processed/`,
writes `.env`, and then checks that the data files are present and that both
databases answer. Every step detects work already done and skips it, so
re-running after a pull is safe and is the right thing to do when
`requirements.txt` changes.

It runs on the system Python on purpose: it creates the virtualenv, so it cannot
live inside one, and it imports nothing outside the standard library.

If a `Commerce_Integration` checkout sits nearby it finds that repo's `.env` and
derives the database settings from it, because the two describe the same
databases in different shapes. Point it somewhere else with
`--commerce-env <path>`. Without one it writes a template with the keys blank,
which is not fatal: the seeded data needs no database.

Doing it by hand instead is four commands:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/seed_dev_data.py
```

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\seed_dev_data.py
```

Call the interpreter directly rather than activating the virtualenv. On Windows
`Activate.ps1` is a script and the default execution policy blocks it, which is
the same obstacle the Commerce app documents for `npm run dev`. Activation buys
nothing here.

Either way, that is the whole setup. Demand Pilot starts the service itself when
`AI_SERVICE_URL` is localhost, so opening a Planning page is enough; start it by
hand only to watch it fail:

```
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`GET /health` should then report `ready: true` with an empty `missing_required`.

No `.env` and no database access are needed for any of this. The seed copies
four files that are already in the repository into `data/processed/`, which is
gitignored and therefore the one thing a clone lacks. It refuses to overwrite an
existing `data/processed/`, so running it on the cron machine is safe.

One thing to check before anything else if the pages report they cannot reach
the server: `FORECAST_SERVER_DIR` in the local `.env`. Copied from a colleague it
points at their checkout, which does not exist here, and that is the most common
cause. Unset it and the app finds the checkout itself.

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

## The weekly run

On the server, as the `coverland` user:

```
0 10 * * 1 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
```

It runs the forecast, then asks the service whether it can still serve and exits
non-zero if not, so cron mails a failure on the Monday it breaks rather than
leaving a colleague to find a stale page on Thursday.

10:00 UTC was 3am Pacific when this was set up. The server stays on UTC, so the
Pacific wall-clock time moves by an hour at each DST transition; adjust the line
then if it matters.

Nothing else has to happen. The run writes into the same checkout the service
reads from, so there is no transfer step and no laptop in the loop.

## Pushing data by hand

Not part of the weekly cycle. This is for data produced somewhere other than the
server: a forecast re-run on a laptop, or a fresh
`scripts/export_inventory_snapshot.py`, that is wanted on the server before the
next Monday.

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

It pushes only the nine files `src/planning/data.py` reads, about 1.5 MB, rather
than the 19 MB of experiment plots and CV dumps in `outputs/` that the server has
no use for. Then it asks the server whether it can actually serve, and exits
non-zero if not.

Whoever runs it needs a key authorised on the server. That is worth having for
`scripts/verify_deployment.sh` too, but it is not needed to keep the forecast
current, which the server does on its own.

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
