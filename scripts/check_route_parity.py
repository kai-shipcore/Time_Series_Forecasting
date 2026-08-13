"""Assert that the API still serves exactly the routes it is expected to serve.

WHY THIS EXISTS
---------------
The statsforecast endpoints were extracted from `api/main.py` into
`api/legacy.py` on 2026-08-13. An earlier attempt at the same extraction, on
2026-08-12, was reverted because it could not be verified: FastAPI registers an
included router as a single opaque object rather than copying its routes into
`app.routes`, so the obvious check --

    len(app.routes) == 35     # before and after

-- compares nothing once a router is involved. It can pass while the router is
mounted at the wrong prefix, or half its routes fail to register, or a decorator
was left pointing at a router that is never included. That is a check that
reports success for a broken deployment, which is worse than no check.

This script walks the router tree instead, resolving each route to the full path
a request would actually match, and compares the resulting set against a
recorded expectation. It also fails on duplicate paths and on shadowing, where
one route's pattern would swallow another's literal path before it is reached.

USAGE
-----
    .venv/bin/python scripts/check_route_parity.py            # check
    .venv/bin/python scripts/check_route_parity.py --write    # re-record

Re-record deliberately, and only when you have added or removed an endpoint on
purpose. The expectation file is committed so a diff to it shows up in review.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

EXPECTED = ROOT / "outputs" / "reports" / "route_parity.json"


def resolved_routes(app) -> list[tuple[str, str]]:
    """Every (methods, path) the app will actually match, walking included routers.

    An included router appears in `app.routes` as a single `_IncludedRouter`,
    which has no `.path` and no `.routes`. Its contents are reachable only
    through `.original_router`, and its mount prefix through `.include_context`.
    Miss that and a mounted router silently contributes nothing to this list,
    which is the failure mode this whole script exists to catch -- so an
    unrecognised route object raises rather than being skipped.
    """
    out: list[tuple[str, str]] = []

    def walk(routes, prefix: str) -> None:
        for r in routes:
            inner = getattr(r, "original_router", None)          # FastAPI included router
            if inner is not None:
                ctx = getattr(r, "include_context", None)
                walk(inner.routes, prefix + (getattr(ctx, "prefix", "") or ""))
                continue
            sub = getattr(r, "routes", None)                      # Starlette Mount / sub-app
            if sub is not None:
                walk(sub, prefix + (getattr(r, "path", "") or ""))
                continue
            path = getattr(r, "path", None)
            if path is None:
                raise RuntimeError(
                    f"Unrecognised route object {type(r)!r} with neither a path nor "
                    "children. It may be a router this walker cannot see into, in which "
                    "case its routes are missing from this check. Teach the walker about "
                    "it rather than skipping it."
                )
            out.append((",".join(sorted(getattr(r, "methods", None) or [])), prefix + path))

    walk(app.routes, "")
    return sorted(set(out))


def probe_routes(app, routes: list[tuple[str, str]]) -> list[str]:
    """Ask the app's own router whether each path resolves, the way a request does.

    The structural walk above reports what the route objects say about themselves.
    This exercises the matching logic instead, which is the check the reverted
    2026-08-12 attempt lacked. It calls `BaseRoute.matches()` on each top-level
    route, including the `_IncludedRouter` holding the legacy endpoints, so a
    router mounted at the wrong prefix or not mounted at all shows up here.

    Two earlier versions of this function were wrong, both recorded because the
    failure mode is a check that passes while testing nothing:

    1. Sending real requests and treating 404 as "not routed" reported everything
       as fine, because the token middleware answers 401 before routing and a
       nonexistent path never reached a 404.
    2. Adding the token then produced false failures on /planning/sku/{sku_id},
       whose handler legitimately raises 404 for a SKU that does not exist. A
       status code cannot tell "no route" from "route says not found".

    Matching avoids both: no middleware, no handler, no database.
    """
    from starlette.routing import Match

    def match(method: str, path: str) -> "Match":
        scope = {"type": "http", "method": method, "path": path, "headers": [],
                 "root_path": "", "query_string": b"", "app": app}
        best = Match.NONE
        for r in app.routes:
            m, _ = r.matches(scope)
            if m == Match.FULL:
                return Match.FULL
            if m == Match.PARTIAL:
                best = Match.PARTIAL
        return best

    # Negative control, run every time rather than trusted: a path that cannot
    # exist must not match. If it does, this probe cannot detect a missing route,
    # and must say so instead of returning a clean pass.
    if match("GET", "/__route_parity_negative_control__") != Match.NONE:
        return ["probe is blind: a nonexistent path matched a route. No route was verified."]

    failures = []
    for methods, path in routes:
        if path.startswith(("/openapi", "/docs", "/redoc")):
            continue
        method = "GET" if "GET" in methods else (methods.split(",")[0] if methods else "GET")
        concrete = re.sub(r"\{[^}]+\}", "probe", path)
        if match(method, concrete) != Match.FULL:
            failures.append(f"{method} {concrete} does not resolve to a route")
    return failures


def _pattern(path: str) -> re.Pattern:
    """A route path as the regex it matches, so shadowing can be detected."""
    return re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", re.escape(path).replace(r"\{", "{").replace(r"\}", "}")) + "$")


def find_shadowing(routes: list[tuple[str, str]]) -> list[str]:
    """Routes that can never be reached because an earlier one matches first.

    FastAPI resolves in registration order, so a parameterised path registered
    before a literal one with the same shape hides it. This is the failure the
    extraction could plausibly have introduced, by changing the order in which
    the two modules' routes are registered.
    """
    problems = []
    for i, (m_a, a) in enumerate(routes):
        if "{" not in a:
            continue
        pat = _pattern(a)
        for m_b, b in routes[i + 1:]:
            if "{" in b or not (set(m_a.split(",")) & set(m_b.split(","))):
                continue
            if pat.match(b):
                problems.append(f"{m_a} {a} is registered before {m_b} {b} and would match it first")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="record the current routes as expected")
    ap.add_argument("--probe", action="store_true",
                    help="also drive the app with real requests (needs no database)")
    args = ap.parse_args()

    from api.main import app
    routes = resolved_routes(app)

    dupes = [p for p in {r[1] for r in routes} if [r[1] for r in routes].count(p) > 1]
    shadowed = find_shadowing(routes)

    if args.write:
        EXPECTED.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED.write_text(json.dumps([list(r) for r in routes], indent=1) + "\n")
        print(f"Recorded {len(routes)} routes to {EXPECTED.relative_to(ROOT)}")
        return 0

    if not EXPECTED.exists():
        print(f"No expectation recorded. Run with --write first.\n  missing: {EXPECTED}")
        return 2

    expected = sorted(tuple(r) for r in json.loads(EXPECTED.read_text()))
    added = [r for r in routes if r not in expected]
    removed = [r for r in expected if r not in routes]

    probed = probe_routes(app, routes) if args.probe else []

    ok = not (added or removed or dupes or shadowed or [p for p in probed if not p.startswith("skipped")])
    print(f"{len(routes)} routes resolved, {len(expected)} expected")
    for label, items in (("ADDED", added), ("REMOVED", removed)):
        for m, p in items:
            print(f"  {label:8s} {m:12s} {p}")
    for p in dupes:
        print(f"  DUPLICATE path registered twice: {p}")
    for s in shadowed:
        print(f"  SHADOWED {s}")
    for p in probed:
        print(f"  PROBE {p}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
