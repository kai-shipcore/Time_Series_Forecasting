# Task: put the forecast API on the Demand Pilot server

Hand this to Claude Code on the Mac that runs the weekly forecast.

You are wiring an existing, already-written deployment to a real host for the
first time. Almost nothing here is code: it is secrets, one-time server setup,
and a cron line. `docs/DEPLOYMENT.md` in this repo is the reference and should
stay true; correct it if reality differs.

## What is already done

Do not rebuild these.

- `.github/workflows/ci-cd.yml` deploys on push to `main`: rsync over SSH, then
  restart a systemd unit named `coverland-forecast-api`. It excludes `data/`,
  `outputs/` and `logs/` so it cannot delete server data.
- `scripts/push_data_to_server.sh` pushes the nine data files and verifies the
  server can serve afterwards.
- `GET /health` reports data readiness, not just liveness.
- The Next.js app shows service status on planning pages and starts a local
  service when one is missing.

## Ask before starting

You cannot infer these. Ask, and do not invent values.

1. The hostname or IP of the Demand Pilot server, and the SSH user.
2. Whether SSH to it uses a key or a password. The Next.js deploy workflow in
   `Commerce_Integration/.github/workflows/deploy.yml` uses `SSH_PASSWORD`; this
   one needs a key. If no deploy key exists, generate one, install the public
   half in the server user's `authorized_keys`, and confirm before storing the
   private half as a GitHub secret.
3. Whether `gh` is authenticated for both repositories (`gh auth status`). If
   not, the secrets have to be set in the GitHub web UI by hand; produce the
   list rather than guessing.
4. Whether the server user has passwordless sudo for
   `systemctl restart coverland-forecast-api`, and can create `/opt`.

## Order matters

### 0. Merge first

The work is on branches. `Time_Series_Forecasting` is on `feat/planning-api`,
4 commits ahead of `main`. `Commerce_Integration` is on `feat/action-list-page`,
7 commits ahead.

The deploy triggers on push to `main`. Merging is therefore the act that
deploys. Do not merge until steps 1 to 3 are done, or the first deploy lands on
a server with no service, no `.env` and no data.

### 1. GitHub secrets on Time_Series_Forecasting

`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`, `DEPLOY_PATH`, and `DEPLOY_PORT`
if not 22. Use `/opt/coverland-forecast-api` for the path, deliberately separate
from the Next.js checkout at `/home/coverland/app`.

Verify: `gh secret list --repo <owner>/Time_Series_Forecasting` shows all five.

### 2. Server, once

Check Python first. CI pins 3.13; `requirements.txt` pins pandas, numpy,
scikit-learn, lightgbm and statsforecast to exact versions because model results
are compared at the third decimal. If the server's Python cannot install those
pins, stop and report it rather than relaxing a pin.

```bash
sudo mkdir -p /opt/coverland-forecast-api
sudo chown -R <deploy-user>:<deploy-user> /opt/coverland-forecast-api
```

Write `/opt/coverland-forecast-api/.env`. The deploy never overwrites it. It
needs `FORECAST_API_TOKEN`, the five `DB_*` variables and the five
`COMMERCE_DB_*` variables. Copy the values from this repo's local `.env`; both
prefixes are required, and a partial set makes the Action List fall back to
sample inventory silently instead of erroring.

Install the systemd unit exactly as written in `docs/DEPLOYMENT.md`. Note it
binds `127.0.0.1`, not `0.0.0.0`: the only client is the Next.js process on the
same box, and loopback enforces that regardless of the firewall.

Verify: `systemctl status coverland-forecast-api` is active, and
`curl -s localhost:8000/health` returns JSON with `"ready": false` at this
point. False is correct here. No data has been pushed yet.

### 3. Next.js side, on the server

In `/home/coverland/app/.env`:

- `AI_SERVICE_URL="http://localhost:8000"`
- `FORECAST_API_TOKEN` identical to the value in the API's `.env`. A mismatch
  gives 401 on everything except `/health`, which means the status indicator
  shows the service up while every page fails.
- `FORECAST_SERVER_DIR` **must not be set.** Unset is how the app declines to
  start the service itself. systemd already supervises it with
  `Restart=always`, and two supervisors on one port is worse than an outage.
  If the variable is present, remove it.

### 4. Data, from the Mac

Add to this repo's local `.env`: `FORECAST_DEPLOY_HOST`, `FORECAST_DEPLOY_USER`,
`FORECAST_DEPLOY_PATH=/opt/coverland-forecast-api`, and `FORECAST_DEPLOY_KEY` if
the key is not ssh's default.

Run `scripts/push_data_to_server.sh`. It exits 0 only if the server reports
ready afterwards. If it exits non-zero, read its output: it prints which files
are still missing and which directory the service is reading from. A `repo_root`
that is not `/opt/coverland-forecast-api` means the service is running from a
different checkout, which is a different bug from a failed push.

Then add the cron line, after the existing Monday forecast job so it pushes what
that run produced:

```
0 10 * * 1 cd <repo path> && scripts/push_data_to_server.sh >> logs/push.log 2>&1
```

Do not replace the existing forecast cron entry. Add to it.

### 5. Merge and deploy

Merge `feat/planning-api` to `main` in `Time_Series_Forecasting`, watch the
Actions run, and confirm the final step reports readiness rather than warning.
Then merge `feat/action-list-page` to `main` in `Commerce_Integration`.

## How to know it worked

Not "the deploy was green". Run:

```bash
scripts/verify_deployment.sh
```

It checks all of the below except the last, over SSH against `127.0.0.1:8000`,
which is the address the Next.js process actually uses. Exits non-zero and
prints the fix under each failure. The checks it runs:

1. `curl -s localhost:8000/health | python3 -m json.tool` on the server shows
   `"ready": true` and a `repo_root` of `/opt/coverland-forecast-api`.
2. `/planning/action-list` in a browser renders a table, and the status chip in
   the header reads "Forecast server up".
3. `/planning/forecast-validation` shows the comparison, with a headline near
   0.1596 against 0.2591. If those numbers are absent, `ml_accuracy.csv` did not
   arrive.
4. The Action List does not say "SAMPLE inventory data". If it does, the
   `DB_*` or `COMMERCE_DB_*` credentials on the server are wrong or incomplete.
5. From a colleague's machine, with nothing installed and no local server
   running, the same pages work. This is the whole point of the exercise, and it
   is the only check that actually tests it.

## Deliberate non-goals

- Do not expose port 8000 publicly or add a firewall rule for it.
- Do not commit anything under `data/`. Only `outputs/reports/ml_accuracy.csv`
  and `ml_accuracy_by_sku.csv` are tracked, deliberately.
- Do not set `FORECAST_SERVER_DIR` on the server.
- Do not relax a version pin in `requirements.txt` to make an install succeed.
