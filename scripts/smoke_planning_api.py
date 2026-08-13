#!/usr/bin/env python3
"""Call every endpoint the two live planning pages need, and check the answers.

    .venv/bin/python scripts/smoke_planning_api.py

Runs the app in-process with FastAPI's TestClient, so there is no server to
start, no port to pick and no token to supply. It needs `data/processed/`
populated; on a machine without it, run `scripts/seed_dev_data.py` first.

WHY THIS EXISTS
---------------
The static checks in `scripts/verify_repo.py` prove the code is internally
consistent: modules import, routes are registered, no path points at a moved
file. They cannot prove a page works, because a handler can be perfectly wired
and still raise on the first line of its body.

That is not hypothetical. On 2026-08-12 the `forecast_date` column was renamed to
`week_of`. The writer was updated, the loader was updated, and
`forecast_snapshot_date()` was not, so it asked a normalised frame for a column
that no longer existed. It raised `KeyError` every time, not intermittently, and
its three callers are `/planning/action-list`, `/planning/sku/{sku_id}` and
`/planning/validation`. The Action List page and the Forecast Validation page
both returned 500 for a full day. Every static check passed throughout: the
import graph was fine, the routes were registered, the column was spelled
consistently everywhere it was spelled.

The only thing that catches that class of bug is calling the endpoint. So this
calls all of them.

WHAT IT ASSERTS
---------------
Each endpoint returns 200, and returns something with content rather than an
empty envelope. It does not check that the numbers are right; that is the final
test's job and the accuracy reports'. It checks that the pages have something to
render.

EXIT CODE
---------
0 if everything passes, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FORWARD = ROOT / "data" / "processed" / "ml_forward_forecasts.parquet"


def main() -> int:
    if not FORWARD.exists():
        print(f"No forecast data at {FORWARD.relative_to(ROOT)}.")
        print("Run scripts/seed_dev_data.py (committed fixtures, no credentials needed).")
        return 1

    import pandas as pd
    from fastapi.testclient import TestClient
    import api.main as main
    from api.main import app

    # The token middleware answers 401 for every path except /health BEFORE
    # routing, so without this header every check below returns 401 and the whole
    # script reports failure regardless of what the handlers do. Taken from the
    # app's own loaded value rather than re-read from .env, so the two cannot
    # disagree. The same behaviour has now misled three separate checks in this
    # repository, which is why it is spelled out here rather than just handled.
    headers = ({"x-forecast-token": main.FORECAST_API_TOKEN}
               if main.FORECAST_API_TOKEN else {})

    sku = str(pd.read_parquet(FORWARD)["unique_id"].iloc[0])
    print(f"probe SKU: {sku}")
    print(f"token: {'sent' if headers else 'not set, so not sent'}\n")

    # (label, method, path, body). Grouped by the page that breaks without them.
    checks = [
        ("Action List", "GET", "/planning/action-list", None),
        ("Action List", "GET", "/planning/not-forecast", None),
        ("Action List", "GET", f"/planning/sku/{sku}", None),
        ("Action List", "GET", f"/planning/sku/{sku}/history", None),
        ("Action List", "POST", "/planning/demand-trend", {"skus": [sku]}),
        ("Forecast Validation", "GET", "/planning/validation", None),
        ("Forecast Validation", "GET", "/planning/demand-patterns", None),
        ("Forecast Validation", "GET", "/planning/demand-vs-forecast", None),
        ("both", "GET", "/health", None),
    ]

    failures: list[str] = []
    page = None
    # raise_server_exceptions=False so a handler blowing up becomes a 500 to
    # inspect rather than an exception that aborts the remaining checks.
    with TestClient(app, raise_server_exceptions=False) as client:
        for label, method, path, body in checks:
            if label != page:
                print(f"-- {label} --")
                page = label
            try:
                r = client.request(method, path, json=body, headers=headers)
            except Exception as e:
                failures.append(f"{method} {path} raised {type(e).__name__}: {e}")
                print(f"  {path:44s} RAISED {type(e).__name__}")
                continue

            size = len(r.content)
            note = ""
            if r.status_code != 200:
                failures.append(f"{method} {path} returned {r.status_code}")
                note = "  <-- FAIL"
            else:
                try:
                    payload = r.json()
                except Exception:
                    payload = None
                if payload in (None, {}, []):
                    failures.append(f"{method} {path} returned 200 with an empty body")
                    note = "  <-- empty"
            print(f"  {path:44s} {r.status_code}  {size:>9,} bytes{note}")

        # The retired statsforecast endpoints must NOT be served. A remount would
        # widen public surface area on a host with no packet filtering.
        print("-- retired, must not resolve --")
        for path in ("/segmentation", "/all-skus", f"/forecast/{sku}", "/backtest/ABC"):
            r = client.get(path, headers=headers)
            ok = r.status_code == 404
            if not ok:
                failures.append(f"{path} returned {r.status_code}, expected 404; is api/legacy mounted?")
            print(f"  {path:44s} {r.status_code}{'' if ok else '  <-- FAIL'}")

    # A summary line worth printing, because it is the number an operator quotes.
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            meta = client.get("/planning/action-list", headers=headers).json().get("meta", {})
        print("\naction-list meta: " + json.dumps(
            {k: meta.get(k) for k in
             ("sku_count", "not_forecast_count", "trained_through", "model_version")}))
    except Exception:
        pass

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("All planning endpoints healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
