@echo off
REM One-time local setup on Windows.
REM
REM     setup.cmd
REM
REM Exists so nobody has to guess which interpreter this machine calls Python.
REM There are three plausible names and only one of them works on any given
REM Windows install:
REM
REM   python3  is a Unix convention and does not exist here at all. It is the
REM            name in every macOS instruction, which is how people arrive at it.
REM   python   may be the real thing, or may be the Microsoft Store's App
REM            Execution Alias: a stub that opens the Store and exits 0 without
REM            running anything. That is the worst case, because it looks like
REM            it worked.
REM   py       the Python launcher, installed by the python.org installer, and
REM            the reliable one when it is present.
REM
REM So this tries py first, then verifies that python is genuinely an
REM interpreter by asking it to print something, rather than trusting that the
REM command resolved.
REM
REM The mirror of `python3 scripts/setup_local.py` on macOS and Linux. Everything
REM it does lives in that script; this only finds the interpreter to run it with.

setlocal

cd /d "%~dp0"

set "PY="

py -3 -c "import sys" >nul 2>&1
if %errorlevel% equ 0 (
    set "PY=py -3"
    goto :found
)

REM Not `where python`: the Store alias resolves and would pass. Running real
REM code is the only check that distinguishes a stub from an interpreter.
python -c "import sys; sys.exit(0)" >nul 2>&1
if %errorlevel% equ 0 (
    set "PY=python"
    goto :found
)

echo.
echo Could not find a working Python on this machine.
echo.
echo Tried "py -3" and "python". If you have just installed Python, close this
echo terminal and open a new one: PATH is read at startup, so a fresh install is
echo invisible to a window that was already open.
echo.
echo Otherwise install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH" in the installer. The Microsoft Store build also
echo works but is the one that leaves a stub named python.exe which opens the
echo Store instead of running, so the python.org installer is the safer choice.
echo.
exit /b 1

:found
echo Using %PY%
echo.
%PY% scripts\setup_local.py %*
exit /b %errorlevel%
