"""One-time local setup. Run once on a new machine, then forget about it.

    python3 scripts/setup_local.py              # macOS / Linux
    py scripts\\setup_local.py                   # Windows PowerShell

Runs on the system Python deliberately: it creates the virtualenv, so it cannot
live inside one, and it imports nothing outside the standard library.

What it does, skipping anything already done:

  1. Checks the Python version.
  2. Creates .venv and installs requirements.txt into it.
  3. Seeds data/processed/ from the repository, via scripts/seed_dev_data.py.
  4. Writes .env, deriving the database settings from a Commerce_Integration
     .env when one can be found. See below.
  5. Verifies: readiness of the data files, and whether both databases answer.

Re-running is safe. Every step detects its own completed state and says so
rather than redoing it. Nothing is overwritten without --force.

Deriving .env, which is the part worth explaining
-------------------------------------------------
This repo and Commerce_Integration read the same two databases, but describe
them differently. Commerce holds one connection URL per database; this repo
holds five discrete variables per database, and names them confusingly:

    DB_*          the primary shipcore database   <- Commerce DATABASE_URL
    COMMERCE_DB_* the Supabase lookup database    <- Commerce
                                                     SUPABASE_LOOKUP_DATABASE_URL

Note that "COMMERCE_DB" is the Supabase one, not the Commerce app's primary.
Wiring those two the wrong way round is easy and produces confusing failures, so
this script does the translation rather than leaving it to be done by hand.

Two tokens are shared verbatim and are copied across when present:
FORECAST_API_TOKEN, which must match or every request but /health returns 401,
and VELOCITY_SYNC_TOKEN, used by the weekly run.

Nothing is derived if a Commerce .env cannot be found. The script then writes a
template with the keys blank and says which ones matter, and the planning pages
still work, because the seeded data needs no database at all.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
MIN_PYTHON = (3, 10)

#: Where a Commerce_Integration checkout might be, relative to this one. Both
#: layouts are in use: siblings, and one nested a level down inside a folder of
#: the same name, which is how the repo clones by default.
COMMERCE_GUESSES = [
    "../Commerce_Integration/Commerce_Integration",
    "../Commerce_Integration",
    "../commerce_integration",
    "../../Commerce_Integration/Commerce_Integration",
    "../../Commerce_Integration",
]


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def did(msg: str) -> None:
    print(f"  [done] {msg}")


def warn(msg: str) -> None:
    print(f"  [!]    {msg}")


def step(n: int, title: str) -> None:
    print(f"\n{n}. {title}")


def venv_python() -> Path:
    """The interpreter inside .venv, on either platform."""
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str], what: str) -> bool:
    """Run a command, showing its output only when it fails.

    A successful install prints several hundred lines that nobody reads and
    which bury the steps around it; a failed one prints the single line that
    matters, and hiding that would be the whole problem this script exists to
    avoid.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True
    warn(f"{what} failed:")
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
    for line in tail:
        print(f"         {line}")
    return False


# ---------------------------------------------------------------------------
# Steps.
# ---------------------------------------------------------------------------
def check_python() -> bool:
    step(1, "Python")
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        warn(f"Python {v.major}.{v.minor} is too old; need "
             f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.")
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()}")
    return True


def ensure_venv() -> bool:
    step(2, "Virtualenv and dependencies")
    if venv_python().exists():
        ok(f".venv already exists ({venv_python().relative_to(ROOT)})")
    else:
        print("     creating .venv …")
        if not run([sys.executable, "-m", "venv", str(VENV)], "Creating .venv"):
            return False
        did("created .venv")

    # Always run, never skipped on the presence of the venv: a pull that changes
    # requirements.txt leaves an existing venv stale, and that failure surfaces
    # much later as a confusing ImportError.
    print("     installing requirements (this is the slow part) …")
    if not run([str(venv_python()), "-m", "pip", "install", "-q",
                "-r", str(ROOT / "requirements.txt")], "Installing requirements"):
        return False
    did("dependencies installed")
    return True


def seed_data(force: bool) -> bool:
    step(3, "Data")
    seed = ROOT / "scripts" / "seed_dev_data.py"
    if not seed.exists():
        warn("scripts/seed_dev_data.py is missing; pull the latest and re-run.")
        return False
    cmd = [str(venv_python()), str(seed)] + (["--force"] if force else [])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        did("seeded data/processed from the repository")
        return True
    if "Refusing to overwrite" in out:
        ok("data/processed already populated, left alone")
        return True
    warn("Seeding failed:")
    for line in out.strip().splitlines()[-12:]:
        print(f"         {line}")
    return False


def parse_pg_url(url: str) -> dict[str, str] | None:
    """Split a postgres URL into the five variables this repo wants.

    Tolerant of the prefixes that show up in practice: postgres://,
    postgresql://, and SQLAlchemy's postgresql+driver://. Query parameters such
    as ?sslmode= or Prisma's ?schema= are ignored, since the connection here
    sets sslmode itself.
    """
    url = url.strip().strip('"').strip("'")
    if not url or "://" not in url:
        return None
    scheme, rest = url.split("://", 1)
    if not scheme.split("+")[0].startswith("postgres"):
        return None
    parsed = urllib.parse.urlparse(f"scheme://{rest}")
    if not (parsed.hostname and parsed.username):
        return None
    return {
        "HOST": parsed.hostname,
        "PORT": str(parsed.port or 5432),
        "NAME": (parsed.path or "/").lstrip("/") or "postgres",
        "USER": urllib.parse.unquote(parsed.username),
        "PASSWORD": urllib.parse.unquote(parsed.password or ""),
    }


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal .env reader. Not python-dotenv, which is not installed yet."""
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def find_commerce_env(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            p = p / ".env"
        return p if p.is_file() else None
    for guess in COMMERCE_GUESSES:
        p = (ROOT / guess / ".env").resolve()
        if p.is_file():
            return p
    return None


def write_env(commerce_env: Path | None, force: bool) -> bool:
    step(4, "Environment")
    dest = ROOT / ".env"
    if dest.exists() and not force:
        ok(".env already exists, left alone (use --force to rewrite)")
        return True

    derived: dict[str, str] = {}
    if commerce_env:
        src = read_env_file(commerce_env)
        for ours, theirs in (("DB", "DATABASE_URL"),
                             ("COMMERCE_DB", "SUPABASE_LOOKUP_DATABASE_URL")):
            parts = parse_pg_url(src.get(theirs, ""))
            if parts:
                for field, value in parts.items():
                    derived[f"{ours}_{field}"] = value
                ok(f"{ours}_* derived from {theirs}")
            else:
                warn(f"{ours}_* not derived: {theirs} absent or unparseable")
        for shared in ("FORECAST_API_TOKEN", "VELOCITY_SYNC_TOKEN"):
            if src.get(shared):
                derived[shared] = src[shared]
                ok(f"{shared} copied across")
    else:
        warn("No Commerce_Integration .env found. Writing a blank template.")
        print("       Pass one with --commerce-env <path to that repo or its .env>.")
        print("       Not fatal: the seeded data needs no database.")

    def val(k: str) -> str:
        return derived.get(k, "")

    body = f"""# Generated by scripts/setup_local.py. Safe to edit by hand.
#
# Both database blocks describe the SAME two databases the Commerce app uses.
# The names read backwards, so: DB_* is the primary shipcore database, and
# COMMERCE_DB_* is the Supabase lookup one.

# Primary (shipcore). Commerce calls this DATABASE_URL.
DB_HOST={val("DB_HOST")}
DB_PORT={val("DB_PORT")}
DB_NAME={val("DB_NAME")}
DB_USER={val("DB_USER")}
DB_PASSWORD={val("DB_PASSWORD")}

# Supabase lookup. Commerce calls this SUPABASE_LOOKUP_DATABASE_URL.
COMMERCE_DB_HOST={val("COMMERCE_DB_HOST")}
COMMERCE_DB_PORT={val("COMMERCE_DB_PORT")}
COMMERCE_DB_NAME={val("COMMERCE_DB_NAME")}
COMMERCE_DB_USER={val("COMMERCE_DB_USER")}
COMMERCE_DB_PASSWORD={val("COMMERCE_DB_PASSWORD")}

# Must match the same variable in the Commerce app, or every request except
# /health returns 401.
FORECAST_API_TOKEN={val("FORECAST_API_TOKEN")}

# Used by the weekly run's velocity sync step. Optional for local work.
VELOCITY_SYNC_URL=
VELOCITY_SYNC_TOKEN={val("VELOCITY_SYNC_TOKEN")}

# Optional. The /chat assistant returns 503 without these, and nothing else
# is affected.
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
"""
    dest.write_text(body, encoding="utf-8")
    did(f"wrote {dest.name}" + (" with derived settings" if derived else " as a blank template"))
    return True


#: Run inside the venv, because that is where the dependencies are.
#:
#: The database part actually connects rather than asking whether an engine
#: object could be built. `_engine` returns None for three unrelated reasons,
#: missing variables, a missing driver, and an unusable URL, and a check that
#: cannot tell them apart reports the wrong fix two times in three. This is the
#: one script whose entire job is saying what is wrong.
_PROBE = r"""
import os, sys
sys.path.insert(0, '.')
from src.planning.data import readiness
r = readiness()
print('READY', r['ready'])
print('MISSING', ','.join(r['missing_required']) or '-')

from src.planning import inventory as I
I._load_env()

for prefix in ('DB', 'COMMERCE_DB'):
    missing = [k for k in ('HOST','PORT','NAME','USER','PASSWORD')
               if not os.getenv(f'{prefix}_{k}')]
    if missing:
        print(prefix, 'UNSET', ','.join(missing)); continue
    try:
        from sqlalchemy import create_engine, text  # noqa: F401
    except Exception as exc:
        print(prefix, 'NODRIVER', type(exc).__name__); continue
    eng = I._engine(prefix)
    if eng is None:
        print(prefix, 'BADURL', '-'); continue
    try:
        with eng.connect() as c:
            c.exec_driver_sql('SELECT 1')
        print(prefix, 'OK', '-')
    except Exception as exc:
        print(prefix, 'REFUSED', str(exc).strip().splitlines()[0][:120])
"""


def verify() -> bool:
    step(5, "Verify")
    proc = subprocess.run([str(venv_python()), "-c", _PROBE],
                          capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        warn("Verification could not run:")
        for line in (proc.stderr or "").strip().splitlines()[-10:]:
            print(f"         {line}")
        return False

    res: dict[str, tuple[str, str]] = {}
    for line in proc.stdout.strip().splitlines():
        m = re.match(r"(\S+) (\S+) ?(.*)", line)
        if m:
            res[m.group(1)] = (m.group(2), m.group(3))

    ready = res.get("READY", ("", ""))[0] == "True"
    if ready:
        ok("data files present; the service can serve")
    else:
        warn(f"missing required data: {res.get('MISSING', ('?', ''))[0]}")

    for name, label in (("DB", "primary (shipcore)"),
                        ("COMMERCE_DB", "Supabase lookup")):
        state, detail = res.get(name, ("UNSET", ""))
        if state == "OK":
            ok(f"{label} database answered")
        elif state == "UNSET":
            warn(f"{label} database: not set in .env ({detail})")
        elif state == "NODRIVER":
            warn(f"{label} database: sqlalchemy/psycopg2 not installed ({detail})")
        elif state == "BADURL":
            warn(f"{label} database: settings present but unusable")
        else:
            warn(f"{label} database: did not answer — {detail}")
        if state != "OK":
            print("         (live inventory unavailable; seeded figures still work)")
    return ready


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commerce-env", metavar="PATH",
                    help="Commerce_Integration checkout, or its .env, to derive settings from")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing .env and re-seed data/processed")
    args = ap.parse_args()

    print(f"Setting up {ROOT}")

    if not check_python():
        return 1
    if not ensure_venv():
        return 1
    seeded = seed_data(args.force)

    commerce_env = find_commerce_env(args.commerce_env)
    if commerce_env:
        print(f"\n     found Commerce .env at {commerce_env}")
    write_env(commerce_env, args.force)

    ready = verify() if seeded else False

    print("\n" + "-" * 70)
    if ready:
        print("Done. Nothing here needs running again.")
        print("\nStart Demand Pilot and open a Planning page: it launches this")
        print("service itself. To watch it start by hand instead:")
        launch = (r"  .venv\Scripts\python.exe -m uvicorn api.main:app --port 8000"
                  if os.name == "nt"
                  else "  .venv/bin/uvicorn api.main:app --port 8000")
        print(launch)
        print("\nIn the Commerce app's .env, check FORECAST_SERVER_DIR: unset it, or")
        print(f"point it at {ROOT}")
    else:
        print("Not finished. The steps marked [!] above say what is left.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
