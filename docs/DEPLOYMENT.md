# Deployment

The forecast API runs on the same server as Demand Pilot, the Next.js app in
`Commerce_Integration`. Both are managed there: Next.js by pm2, this service by
systemd.

## Why the same box

Next.js proxies every forecast request from its own server process rather than
from the browser, so there is no CORS to configure and no colleague needing
Python, a virtualenv or a copy of the data: they open the deployed app and the
forecast is already behind it.

It also meant no public port for the API. That changed on 2026-08-07: 8000 is
now reachable directly, protected by `FORECAST_API_TOKEN` rather than by being
unreachable. See "Reaching the deployed service from a laptop" below. The
co-location still earns its place for the reasons above, and the proxy remains
the normal path; direct access is for local development against real data.

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

## Reaching the deployed service from a laptop

**Since 2026-08-07: directly.** `http://144.24.40.252:8000` is reachable, with
`FORECAST_API_TOKEN` in the `x-forecast-token` header. Set that and
`AI_SERVICE_URL=http://144.24.40.252:8000` and the Planning pages work against
real data with no local Python, no virtualenv and no copy of this repo.

**Why it did not work before, which was never a firewall rule.** The systemd
unit ran uvicorn with `--host 127.0.0.1`, so the process only accepted
connections originating on the box. Nothing was listening on the public
interface, so the kernel answered with a RST and the caller saw "connection
refused". That was read as a closed port and written down here as "a per-port
firewall rule". It was not. A firewall DROP produces a timeout; a refusal means
the packet arrived and no socket wanted it. The distinction is the whole
diagnosis and it was inverted here for months.

Checked on 2026-08-07, after changing the unit to `--host 0.0.0.0`: the host is
not filtering at all (`iptables` INPUT policy ACCEPT, one redundant rule for
port 3000, no REJECT; firewalld and ufw both inactive), and the Oracle VCN
security list already permitted 8000. No firewall change was made or needed.

**Consequence worth knowing.** The host has no packet filtering, so the VCN
security list is the only network control this server has. `FORECAST_API_TOKEN`
is what protects the API, and it is load-bearing: `api/main.py` enforces it on
every path except `/health`, and only when the variable is set. If it is ever
unset, `POST /run-forecast` and the six other POST endpoints are open to the
internet.

The earlier mistake in this section is left recorded below, because the lesson
still holds. On 2026-07-31 a developer machine appeared to be talking to the
server with `AI_SERVICE_URL` set to that address. It was not: that machine had a
local forecast server running, and the app falls back to starting and using a
local one, so the configured URL looked like it worked while something else
answered. The test that settles it is `curl` against the address from a machine
with no local server, and that is the first thing to do rather than the last.

Three ways to work, cheapest first.

**1. Run the service locally.** `setup.cmd` on Windows, `python3
scripts/setup_local.py` elsewhere, then set `AI_SERVICE_URL=http://localhost:8000`.
Serves the committed fixture, needs no database and no network access to the
server. This is the right answer for anyone working on the planning pages.

**2. Tunnel, when current data is wanted.**

```bash
ssh -L 8000:127.0.0.1:8000 coverland@144.24.40.252
```

Leave it open, keep `AI_SERVICE_URL=http://localhost:8000`, and set
`FORECAST_API_TOKEN` to the server's value. The local app then reads live data
through the SSH session, with nothing exposed publicly. Note that the app still
considers `localhost` a server it may start, which is harmless while the tunnel
is up since it only spawns uvicorn when nothing answers, and confusing when the
tunnel drops, since it will then quietly serve local data instead. If in doubt,
check `trained_through` on the Action List.

**3. Point straight at the server.** Done on 2026-08-07, so this is now the
cheapest option for reading live data:

```
AI_SERVICE_URL=http://144.24.40.252:8000
FORECAST_API_TOKEN=<the server's value>
```

No tunnel, no local service, no copy of this repo. The trade is that
`FORECAST_API_TOKEN` is now the entire perimeter: the service holds credentials
for both databases, `/health` sits outside the token check, and the host does no
packet filtering. Option 2's tunnel still exposes nothing and remains the right
choice if you already have SSH open.

**Which file to set `AI_SERVICE_URL` in.** `Commerce_Integration` has both `.env`
and `.env.local`, Next.js loads `.env.local` at higher precedence, and both are
gitignored so two machines can disagree indefinitely. Check with
`findstr /C:"AI_SERVICE_URL" .env .env.local` before editing, and restart
`npm run dev` afterwards, because env is read at startup.

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

```

Eleven values, as of 2026-08-13.

**The four `LLM_*` and `FORECAST_SELF_URL` values are gone.** They configured the
AI assistant on the Demand Forecast page, which was deleted along with `/chat`
and `src/chat.py` when the statsforecast track was separated out. It had never
worked: `src/chat.py` addressed port 8001, where nothing has ever listened, so
its tool calls failed silently for the whole life of the deployment. Setting
them now does nothing. Remove them from the server's `.env` at the next
opportunity rather than leaving values that appear to configure something.

The history is worth one line, because this list has been wrong in both
directions. An earlier version of this document listed eleven values and omitted
the three `LLM_*` ones, which was enough to start the service and lose the chat
feature with nothing saying so. It is back to eleven for a different reason: the
feature those values fed no longer exists.

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
0 10 * * 2 cd /opt/coverland-forecast-api && scripts/run_forecast_cron.sh >> logs/forecast_cron.log 2>&1
```

**Tuesday (day 2), and this is not cosmetic.** A week runs Tuesday through
Monday and is labelled by the Monday it ends on, so the bucket labelled Monday L
is still open for the whole of Monday L. A Monday run can only use the week that
ended the *previous* Monday, which makes every forecast seven days staler than
it needs to be. A Tuesday run picks up the week that closed hours earlier.

This moved from Monday to Tuesday on 2026-08-06, together with `clean.py`
returning to `closed="right"` and `last_complete_week` restoring its extra
Monday step. The three are one decision and must be deployed together; see
`src/weeks.py`. If you are reading this because the forecast looks a week out of
date, check all three before changing any of them.

It runs **two** pipelines, then asks the service whether it can still serve and
exits non-zero if not, so cron mails a failure on the Tuesday it breaks rather
than leaving a colleague to find a stale page on Thursday.

The order is load-bearing:

1. `run_forward_forecast.py`, the legacy statsforecast run. It performs the
   ingest, pulling fresh order lines and rewriting `sales_clean.parquet`, and
   writes the `shipcore.fc_*` tables that SKU Planning still reads.
2. `ml_forward_forecast.py --snapshot live`, which has no ingest of its own. It
   reads the sales file the first run just refreshed, writes
   `ml_forward_forecasts`, and appends a row per SKU to `ml_forecast_history`.
   That table is what the Action List and Forecast Validation serve.

`--snapshot live` is not optional. Without it the script defaults to
`config.ML_DATA_SNAPSHOT`, the pinned copy that exists so recorded evaluation
figures cannot drift, and every weekly run would reproduce the same forecast
while appearing to work.

Until 2026-08-04 this job ran only the first of the two. The consequence was
invisible for weeks: the legacy tables stayed current, so nothing looked broken,
while the two live planning screens served whichever ML forecast someone had last
produced by hand and the history store never gained a real run. The readiness
check and the history backup in this script were both already written for the ML
run, which is what made the omission easy to miss.

**This is the only weekly job.** There was previously a second one on a
developer's Mac running `push_data_to_server.sh`, from when the pipeline ran
there and the results had to be copied up. `CUTOVER_TASK.md` recorded that as a
decision to settle rather than keep: either the pipeline moves to the server and
the Mac cron is retired, or the Run Forecast button goes. The pipeline moved, so
the Mac cron is retired. Retire it with `crontab -e` on that machine and delete
the `push_data_to_server.sh` line; the script stays in the repo for a one-off
manual push, but nothing schedules it. Two machines writing the same files was
the failure that decision existed to prevent.

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

## Runbook: the forecast API is not responding

Written for whoever picks this up. Everything here was hit for real in August 2026
and cost a day, so the symptoms are described the way they actually appear rather
than the way they ought to.

### First: is it actually down?

```bash
bash scripts/_test_port_8000.sh          # from any machine that is not the server
```

Expect JSON from `/health` and **401** from `/segmentation`. The 401 is a pass, not
a failure: it proves the port is reachable and that the token is being enforced.

Run this from a laptop, never from the server. Checking from the box passes even
when the service is bound to loopback and no one else can reach it, which is the
most common failure here.

### Reading the failure

The distinction that matters, and the one that was misread for months:

| Symptom | Meaning |
|---|---|
| Hangs, then times out | Packets are being dropped. Oracle Cloud VCN security list. |
| Connection refused, immediately | Packets arrive, nothing is listening on the public interface. The service is down, or bound to `127.0.0.1`. |
| 200 on `/segmentation` with no token | Worse than an outage. The API is unauthenticated on a public port. Fix `FORECAST_API_TOKEN` before anything else. |

A refusal means the network is fine. Only a hang implicates a firewall.

### Then: what does the server say?

GitHub → Actions → **Server diagnostics** → Run workflow. Read-only, no password,
about seven seconds. It reports who owns port 8000, whether the unit is healthy or
restart-looping, every uvicorn process with its start time, and whether the API
answers.

### The three failure modes seen so far

**1. Something else is holding port 8000.**
Symptom: diagnostics shows `127.0.0.1:8000`, and the unit state is `activating`
rather than `active`. The journal says `[Errno 98] address already in use` every
eight seconds.

```bash
bash scripts/_kill_squatter.sh
```

The unit binds within a few seconds. This was caused by the deploy's own fallback
branch and is fixed at source (see BACKLOG 21), but a person starting a server by
hand on the box produces the same state.

**2. The service is bound to loopback.**
Symptom: connection refused from outside, works on the box.
Check `systemctl show coverland-forecast-api -p ExecStart --value`. It must end
`--host 0.0.0.0 --port 8000 --workers 1`. If it says `127.0.0.1`, the unit file was
replaced or the server was rebuilt. The correct unit is version-controlled at
`deploy/coverland-forecast-api.service`; copy it to
`/etc/systemd/system/`, `daemon-reload`, restart.

**3. Reachable but useless.**
Symptom: `/health` returns 200 with `"ready": false`. The process is alive and has
no data to serve, so every planning page 500s. Usually means the weekly cron has
not run or failed. Check `crontab -l | grep run_forecast_cron` (expect day 2,
Tuesday) and the mtimes under `data/processed/`.

### What watches this when nobody is looking

**Hourly:** `.github/workflows/api-reachable.yml` curls the public URL from a
GitHub runner, which is off the network and therefore tests the same path a laptop
uses. It fails the run on unreachable, on `ready: false`, and on auth not being
enforced. A red run emails the person who last touched that workflow file.

**Every deploy:** `ci-cd.yml` asserts after restarting that the unit is `active`
on two checks five seconds apart AND that `0.0.0.0:8000` is bound. A crash-looping
unit now fails the build with the socket owner and journal attached. Before this,
`systemctl restart` exited 0 while the service failed to start, so a deploy could
report green while shipping code that never ran.

**Two things to know about that monitoring.** GitHub disables scheduled workflows
after 60 days without repository activity, and the Actions tab then offers a
re-enable button; a monitor that silently stops is worse than none. And the failure
email goes to whoever last committed to the workflow file, which after the handoff
is the wrong person. Change it deliberately.

### Things that are only correct together

The week convention is encoded in three places and breaking one silently corrupts
the forecast rather than erroring:

- `src/clean.py`, `closed="right"` on the W-MON grouper
- `src/weeks.py:last_complete_week`, steps back an extra week on Mondays
- the crontab, day 2 (Tuesday), because bucket L stays open all of Monday L

If the forecast ever looks a week stale, check all three before changing any of
them. Design doc Section 4.30 has the evidence for why the convention is
Tuesday-to-Monday rather than the Monday-to-Sunday most of the older
documentation used to claim.

### Security posture, stated plainly

The host does no packet filtering: `iptables` INPUT policy is ACCEPT with no
REJECT, and firewalld and ufw are both inactive. The Oracle VCN security list is
the only network control, and `FORECAST_API_TOKEN` is the only application
control. `/health` sits outside the token check by design, and seven POST
endpoints sit behind it, including `/run-forecast` and `/planning/run-forecast`,
which each spawn a pipeline run. If the token is ever unset, those are open to
the internet. The hourly check tests for exactly that.

It was eight until 2026-08-13. `/chat` was the eighth, and it was the worst of
them, being an LLM call on somebody else's bill. It is gone.
