@echo off
setlocal
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set "HI3D_WORKBENCH_ROOT="
if exist "runtime\python.exe" (
  "runtime\python.exe" -u -m hi3d_workbench --stop %*
) else if exist ".appenv\Scripts\python.exe" (
  ".appenv\Scripts\python.exe" -u -m hi3d_workbench --stop %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -u -m hi3d_workbench --stop %*
) else (
  echo Python runtime not found. Close the original launcher window.
)
pause
