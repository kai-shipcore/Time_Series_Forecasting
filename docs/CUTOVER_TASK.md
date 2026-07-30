# Task: cut over from coverland-forecast to coverland-forecast-api

Hand this to Claude Code. The server has been probed and this describes what is
actually there, not a general procedure.

## What is on the server now

| Unit | State | Directory | Bind |
| --- | --- | --- | --- |
| `coverland-forecast.service` | active, enabled | `/home/coverland/Time_Series_Forecasting` | `0.0.0.0:8000` |
| `coverland-forecast-api.service` | inactive, disabled | `/opt/coverland-forecast-api` | `127.0.0.1:8000` |
| `pm2-coverland.service` | active | the Next.js app | |

Both units run `api.main:app` from the same codebase. They are two checkouts of
one repository competing for one port, so this is a **retirement, not a
coexistence**. The old one currently holds 8000 and answers `/health` with
`{"status":"ok"}` and no `repo_root`, which places it before the readiness
work.

Nothing listens on 8001.

## Fix these before starting

### 1. Four missing variables in `/opt/coverland-forecast-api/.env`

The service reads fifteen; the file has eleven. Add:

```
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
FORECAST_SELF_URL=http://127.0.0.1:8000
```

Copy the three `LLM_*` values from `/home/coverland/Time_Series_Forecasting/.env`
or the Mac's `.env`. Without them `/chat` returns 503 and the assistant on the
Demand Forecast page stops working, while everything else looks fine.

`FORECAST_SELF_URL` is where the chat's tool loop calls back into this service.
It must match the port in `ExecStart`.

### 2. The assistant is already answering without data

`src/chat.py` has hardcoded `http://127.0.0.1:8001` since the commit that
introduced it, and nothing has ever listened on 8001 here. Its tool calls have
been failing for the whole life of this deployment. That does not surface as an
error: the model answers from its own words instead of from the figures, which
is worse than a visible failure.

Setting `FORECAST_SELF_URL=http://127.0.0.1:8000` repairs it. Confirm after
cutover by asking the assistant something only a tool call can answer, such as
the recommended order quantity for a specific SKU, and checking the figure
against the Action List.

### 3. Port 8000 is bound to `0.0.0.0` on a public host

The current unit binds all interfaces on `144.24.40.252`. Whether that is
reachable from outside depends on the cloud security list and any local
firewall, which this task has not checked. **Check it before anything else,**
from a machine that is not the server:

```bash
curl -s --max-time 5 http://144.24.40.252:8000/health
```

If that answers, the forecast API has been internet-facing. Then check whether
the token was enforcing anything, from the server:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/segmentation
```

`401` means the token was set and the exposure was limited to `/health`. `200`
means it was open, and `POST /run-forecast` was reachable by anyone who found
the port, since that endpoint spawns a pipeline run.

Report what these return. The cutover closes the exposure either way, because
the new unit binds `127.0.0.1`, but if it was open that is worth knowing
separately and may warrant rotating `FORECAST_API_TOKEN`.

## The cutover

There is a brief outage of the existing Demand Forecast page between steps 2
and 3. Everything else is reversible.

1. Add the four variables above. Do not restart anything yet.

2. Stop the old service, and disable it so a reboot cannot bring it back to
   race for the port:

   ```bash
   sudo systemctl disable --now coverland-forecast.service
   ```

3. Start the new one:

   ```bash
   sudo systemctl enable --now coverland-forecast-api.service
   sudo systemctl status coverland-forecast-api.service
   ```

   If it does not come up, roll back immediately and report:

   ```bash
   sudo systemctl disable --now coverland-forecast-api.service
   sudo systemctl enable --now coverland-forecast.service
   ```

4. From the Mac:

   ```bash
   scripts/verify_deployment.sh
   ```

   Expect all checks to pass, `repo_root` of `/opt/coverland-forecast-api`, 432
   rows, live inventory, and 0.1596 against 0.2591.

5. In `/home/coverland/app/.env`: set `AI_SERVICE_URL="http://localhost:8000"`
   and `FORECAST_API_TOKEN` to the value in the API's `.env`. Confirm
   `FORECAST_SERVER_DIR` is **absent**; systemd supervises the service and a
   second supervisor on the same port is worse than an outage. Then
   `pm2 restart demand-pilot --update-env`.

6. Check the **existing** Demand Forecast page before the new ones. It was
   served by the unit just retired and is the thing most likely to regress:
   segmentation counts, a SKU forecast chart, the accuracy trend, and the
   assistant. Only then check `/planning/action-list` and
   `/planning/forecast-validation`.

7. Merge `feat/planning-api` into `main` in `Time_Series_Forecasting`. CI will
   rsync the current code to `/opt` and restart the unit. Its readiness step now
   fails if port 8000 is answered by a service running from a different
   directory, so a regression here is loud.

8. Merge `feat/action-list-page` into `main` in `Commerce_Integration`.

## After: one thing that needs deciding, not doing

Two machines can now write the same data. The Mac's Monday cron pushes files to
`/opt`, and the Demand Forecast page's **Run Forecast** button spawns
`scripts/run_forward_forecast.py` on the server, writing into the same
directory. A run triggered from the UI on Wednesday is silently replaced by the
Mac's push the following Monday.

Do not fix this during the cutover. Report it, and it will be settled
separately: either the pipeline moves to the server and the Mac cron is retired,
or the button is removed. Both are defensible; picking one by accident is not.

## Do not

- Do not delete `/home/coverland/Time_Series_Forecasting`. It is the rollback,
  and it may hold cron entries or an `.env` that is the only copy of something.
  Check `crontab -l` for the `coverland` user and report what is there.
- Do not change the new unit's bind address to `0.0.0.0` to make something work.
  If the app cannot reach it on loopback, they are not on the same host and that
  is the thing to report.
- Do not merge before step 6 passes.
