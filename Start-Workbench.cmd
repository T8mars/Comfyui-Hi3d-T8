@echo off
cd /d "%~dp0"
if exist "runtime\python.exe" (
  "runtime\python.exe" -m hi3d_workbench --open
) else if exist ".appenv\Scripts\python.exe" (
  ".appenv\Scripts\python.exe" -m hi3d_workbench --open
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m hi3d_workbench --open
) else (
  echo Run scripts\Setup.ps1 with Python 3.10 first.
  pause
)
