#!/usr/bin/env python3
"""Whole-repository sanity check. Needs no database and no network.

    .venv/bin/python scripts/verify_repo.py

WHY THIS EXISTS
---------------
On 2026-08-13 the statsforecast track was moved into `src/legacy/`, `api/legacy/`
and `scripts/legacy/`, its API router was unmounted, and the weekly cron was
repointed. A lot of files moved. Moves are the change most likely to leave
something broken in a way that only shows up weeks later, because Python resolves
imports at call time and a file path in a shell script or a workflow is not
checked by anything at all until it runs.

Each check below corresponds to something a move can plausibly break. None of
them needs credentials, so this is runnable anywhere, including on a fresh clone.

WHAT IT DOES NOT COVER
----------------------
Anything requiring the database or the browser. It cannot tell you the forecast
is correct, only that the code is internally consistent. `docs/OPERATIONS.md`
section 6 covers the live checks.

EXIT CODE
---------
0 if every check passes, 1 otherwise. Safe to wire into CI.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".ipynb_checkpoints"}

failures: list[str] = []
notes: list[str] = []


def py_files() -> list[Path]:
    return [p for p in ROOT.rglob("*.py")
            if not any(part in SKIP_DIRS for part in p.parts)]


def check(name: str, problems: list[str], detail: str = "") -> None:
    if problems:
        failures.append(name)
        print(f"FAIL  {name}")
        for p in problems[:15]:
            print(f"        {p}")
        if len(problems) > 15:
            print(f"        ... and {len(problems) - 15} more")
    else:
        print(f"ok    {name}{('  (' + detail + ')') if detail else ''}")


# ---------------------------------------------------------------------------
def check_compiles() -> None:
    """Syntax. Cheap, and catches a truncated file from a bad edit."""
    bad = []
    for p in py_files():
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(ROOT)}:{e.lineno}: {e.msg}")
    check("every .py parses", bad, f"{len(py_files())} files")


# ---------------------------------------------------------------------------
def check_modules_import() -> None:
    """Import every module under src/ and api/ for real.

    Stronger than parsing: this resolves every top-level import, so a module
    still pointing at a moved path fails here. The archived legacy packages are
    included deliberately. They are not run any more, but a record that no longer
    imports is a record nobody can check, so it is worth knowing if it rots.
    """
    names = sorted({m.name for pkg in ("src", "api")
                    for m in pkgutil.walk_packages([str(ROOT / pkg)], prefix=pkg + ".")})
    bad = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:
            bad.append(f"{name}: {type(e).__name__}: {e}")
    check("every src/ and api/ module imports", bad, f"{len(names)} modules")


# ---------------------------------------------------------------------------
def check_script_imports() -> None:
    """Every module a script imports must be resolvable.

    Static rather than executed: scripts do work at import time and running 90 of
    them is neither fast nor safe. This walks the import statements and asks
    whether each target can be found, which is what a move breaks.
    """
    bad = []
    for p in sorted(ROOT.glob("scripts/**/*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue                                   # already reported above
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                targets.add(node.module)
            elif isinstance(node, ast.Import):
                targets.update(a.name for a in node.names)
        for t in sorted(targets):
            if not t.startswith(("src", "api", "config", "scripts")):
                continue                               # third-party: pip's problem
            try:
                if importlib.util.find_spec(t) is None:
                    bad.append(f"{p.relative_to(ROOT)} imports {t}, which does not resolve")
            except (ImportError, ModuleNotFoundError, ValueError):
                bad.append(f"{p.relative_to(ROOT)} imports {t}, which does not resolve")
    check("script imports resolve", bad)


# ---------------------------------------------------------------------------
def check_referenced_paths() -> None:
    """Repo-relative file paths named in executable files must exist.

    This is the check that would have caught the spawn path in api/legacy/routes.py
    after run_forward_forecast.py moved, and it is the one nothing else does: a
    wrong path in a .sh or a workflow is invisible until the day it runs.

    Markdown is deliberately excluded, and that is not laziness. WORKLOG.md and
    BACKLOG.md exist to record what happened, so they name files that were
    deliberately deleted and must go on naming them. A check that failed on those
    would be demanding that the history be falsified, and would be silenced within
    a week. Docs are reported separately, as information rather than failure.
    """
    pattern = re.compile(r'(?<![\w./-])((?:scripts|src|api|data|deploy)/[\w./\[\]{}-]+\.(?:py|sh|csv|parquet|json|service|yml))')

    def candidates(p: Path) -> set[str]:
        """Paths that could actually be used, excluding ones merely discussed.

        For Python this means string literals only. A comment or docstring
        explaining that `src/chat.py` was deleted is correct and must not fail a
        check; a string literal naming it is a path something may open. Comments
        elsewhere in this repository are load-bearing documentation, so a checker
        that cannot tell the two apart would force them to be written vaguely.

        For shell and YAML there is no such structure, so comment lines are
        stripped and the rest is scanned.
        """
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return set()
        if p.suffix == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError:
                return set()
            # Docstrings are string literals to the parser and prose to a reader.
            # They are excluded for the same reason comments are: this repository
            # documents deleted files on purpose, and a check that punished that
            # would be asking the record to lie.
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    body = getattr(node, "body", None) or []
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        docstrings.add(id(body[0].value))
            found: set[str] = set()
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and id(node) not in docstrings):
                    found |= set(pattern.findall(node.value))
            return found
        body = "\n".join(l for l in text.splitlines()
                         if not l.strip().startswith("#"))
        return set(pattern.findall(body))

    def scan(paths, literals_only: bool):
        out = []
        for p in paths:
            found = candidates(p) if literals_only else set(
                pattern.findall(p.read_text(encoding="utf-8", errors="ignore")))
            for m in found:
                if any(ch in m for ch in "{}[]*") or "XX" in m:
                    continue                           # templated, glob, or placeholder
                if not (ROOT / m).exists():
                    out.append(f"{p.relative_to(ROOT)} references {m}, which does not exist")
        return out

    code = [p for p in ROOT.rglob("*")
            if p.is_file() and p.suffix in {".py", ".sh", ".yml", ".yaml", ".cmd"}
            and not any(part in SKIP_DIRS for part in p.parts)]
    check("paths used in code, shell and CI exist", scan(code, True), f"scanned {len(code)} files")

    # Informational. A live document pointing at a moved file is worth fixing; a
    # historical record naming a deleted one is correct and must be left alone.
    historical = {"WORKLOG.md", "BACKLOG.md", "CUTOVER_TASK.md", "DEPLOY_TASK.md",
                  "INVENTORY_EXPORT_TASK.md", "V1_AND_DASHBOARD_WIRING_TASK.md",
                  "PLANNING_PLAN.md", "PLANNING_REQUIREMENTS.md"}
    live_docs = [p for p in ROOT.glob("docs/*.md") if p.name not in historical]
    stale = scan(live_docs, False)
    if stale:
        notes.append("NOTE  live documents naming files that do not exist "
                     "(not a failure, but worth a look):")
        notes.extend(f"        {s}" for s in stale)


# ---------------------------------------------------------------------------
def check_routes() -> None:
    """The API serves exactly the recorded routes, and the check can still fail."""
    from scripts.check_route_parity import resolved_routes, find_shadowing, probe_routes, EXPECTED
    import json

    from api.main import app
    routes = resolved_routes(app)
    expected = sorted(tuple(r) for r in json.loads(EXPECTED.read_text()))

    bad = []
    for m, p in [r for r in routes if r not in expected]:
        bad.append(f"unexpected route {m} {p}")
    for m, p in [r for r in expected if r not in routes]:
        bad.append(f"missing route {m} {p}")
    bad += find_shadowing(routes)
    bad += probe_routes(app, routes)

    # Negative control: the probe must be able to fail.
    if not probe_routes(app, [("GET", "/__verify_repo_should_not_exist__")]):
        bad.append("the route probe did not flag a nonexistent path, so it is not testing anything")

    check("API routes match the recorded expectation", bad, f"{len(routes)} routes")


# ---------------------------------------------------------------------------
def check_legacy_not_mounted() -> None:
    """The retired track must stay retired.

    Remounting it is a deliberate act (two commented lines in api/main.py). This
    turns an accidental remount into a failed check rather than a quiet increase
    in public surface area on a host with no packet filtering.
    """
    from api.main import app
    from scripts.check_route_parity import resolved_routes
    live = {p for _, p in resolved_routes(app)}
    legacy_markers = {"/segmentation", "/all-skus", "/forecast/{sku_id}", "/sku-search"}
    found = sorted(legacy_markers & live)
    check("statsforecast router is not mounted",
          [f"{p} is being served; api/legacy is mounted again" for p in found])


# ---------------------------------------------------------------------------
def check_cron() -> None:
    """The weekly job parses, and the scripts it names exist."""
    cron = ROOT / "scripts" / "run_forecast_cron.sh"
    bad = []
    if not cron.exists():
        check("weekly cron script", [f"{cron.relative_to(ROOT)} is missing"]); return
    r = subprocess.run(["bash", "-n", str(cron)], capture_output=True, text=True)
    if r.returncode != 0:
        bad.append(f"bash -n failed: {r.stderr.strip()}")

    text = cron.read_text(encoding="utf-8")
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
    # Every $VAR used must be assigned somewhere, because `set -u` makes an
    # unassigned one abort the run rather than read as empty.
    known_env = {"BASH_SOURCE", "REPO_ROOT", "FORECAST_API_TOKEN", "PATH", "HOME", "PWD", "HIST"}
    used = set(re.findall(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)', body))
    assigned = set(re.findall(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=', body, re.M))
    for v in sorted(used - assigned - known_env):
        bad.append(f"${v} is used but never assigned; `set -u` will abort the run")

    for script in re.findall(r'(scripts/[\w./-]+\.py)', body):
        if not (ROOT / script).exists():
            bad.append(f"cron calls {script}, which does not exist")
    check("weekly cron is valid", bad)


# ---------------------------------------------------------------------------
def check_no_stale_imports() -> None:
    """Nothing may import the pre-move module paths."""
    moved = ["src.models", "src.selector", "src.backtest", "src.baselines", "src.chat"]
    bad = []
    for p in py_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in moved:
            if re.search(rf'\b{re.escape(m)}\b(?!\.)', text) and "src.legacy" not in text.split(m)[0][-40:]:
                for i, line in enumerate(text.splitlines(), 1):
                    if re.search(rf'(from|import)\s+{re.escape(m)}\b', line):
                        bad.append(f"{p.relative_to(ROOT)}:{i}: imports {m}, which moved")
    check("no imports of moved modules", bad)


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Verifying {ROOT}\n")
    for fn in (check_compiles, check_modules_import, check_script_imports,
               check_referenced_paths, check_routes, check_legacy_not_mounted,
               check_cron, check_no_stale_imports):
        try:
            fn()
        except Exception as e:
            failures.append(fn.__name__)
            print(f"FAIL  {fn.__name__} raised {type(e).__name__}: {e}")
    print()
    for n in notes:
        print(n)
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
