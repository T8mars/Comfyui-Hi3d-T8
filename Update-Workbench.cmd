@echo off
setlocal
cd /d "%~dp0"
echo Close the Hi3D workbench before installing a code update.
if exist "runtime\python.exe" (
  set "PYTHONHOME="
  set "PYTHONPATH="
  "runtime\python.exe" -m hi3d_workbench.updater --install
) else if exist ".appenv\Scripts\python.exe" (
  ".appenv\Scripts\python.exe" -m hi3d_workbench.updater --install
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m hi3d_workbench.updater --install
) else (
  echo Python runtime not found. Obtain the complete bundle from T8star.
)
pause
