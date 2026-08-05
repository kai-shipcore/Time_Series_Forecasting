"""Ask the app to refresh its velocity snapshot before a forecast run.

Extracted from scripts/run_forward_forecast.py so both pipelines can call it.
It lived there because that script was the only thing that ran weekly; the
on-demand ML pipeline needs the same call, and importing it from the legacy
runner would have pulled statsforecast, the model menu and the conformal
interval machinery into a script written specifically to avoid all of that. A
twenty-line HTTP POST is not worth that dependency, and a second copy of the
call would be free to disagree with this one about the endpoint or the header
name.
"""

import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

#: Generous, and deliberately so. The sync upserts the full order-line table on
#: the app side, which takes minutes on a cold run. A shorter timeout does not
#: cancel that work, it only stops us hearing how it went.
SYNC_TIMEOUT_SECONDS = 900


def sync_velocity_snapshot() -> None:
    """Trigger the app's velocity sync so the run trains on fresh order data.

    Requires VELOCITY_SYNC_URL and VELOCITY_SYNC_TOKEN in the environment
    (.env). Failure is logged but does not abort the run: a forecast on
    slightly stale data beats no forecast, and the caller has no better
    response available than continuing.

    Prints rather than returns, because both callers stream stdout as the
    progress record of the run and a silent success would leave the sync
    indistinguishable from a step that never happened.
    """
    # override=True, per the note in CLAUDE.md: the user's shell carries stale
    # DB_* exports that otherwise shadow .env, an incident that cost an
    # afternoon to a truncated password.
    load_dotenv(override=True)

    url = os.getenv("VELOCITY_SYNC_URL")
    token = os.getenv("VELOCITY_SYNC_TOKEN")
    if not url or not token:
        print("  VELOCITY_SYNC_URL / VELOCITY_SYNC_TOKEN not set — skipping sync")
        return
    # Said before the request, not after, because there is nothing to say after
    # until it returns. The app upserts the order-line table in batches and
    # holds the connection open for the whole thing, so this call blocks for
    # minutes with no output. Anyone watching a silent step for that long
    # reasonably concludes it has hung and interrupts it, which does not stop
    # the sync: the POST has already been delivered and the app finishes it
    # regardless of whether anyone is still listening.
    print("  Refreshing the velocity snapshot. This runs on the app server and "
          "usually takes a few minutes.", flush=True)
    try:
        req = urllib.request.Request(url, method="POST", headers={"x-sync-token": token})
        with urllib.request.urlopen(req, timeout=SYNC_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode())
            if body.get("success"):
                # Formatted only when it is a number. The original wrote
                # `{body.get('linkUpserted', '?'):,}`, where the fallback can
                # never be formatted: applying a thousands separator to the
                # string "?" raises ValueError, which the handler below catches
                # and reports as "Sync failed", turning a sync that worked into
                # a logged failure. Latent today because the route always
                # returns the field, and one response-shape change away from
                # firing.
                n = body.get("linkUpserted")
                count = f"{n:,}" if isinstance(n, int) else "an unreported number of"
                print(f"  Sync OK — {count} link rows upserted")
            else:
                print(f"  Sync responded but failed: {body.get('error')}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:200]
        print(f"  Sync failed (HTTP {e.code}): {detail} — continuing with existing snapshot")
    except Exception as e:  # noqa: BLE001 — never let the sync kill the forecast run
        print(f"  Sync failed: {e} — continuing with existing snapshot")
