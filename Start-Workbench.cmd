@echo off
setlocal
cd /d "%~dp0"
if exist "runtime\python.exe" (
  set "PYTHONHOME="
  set "PYTHONPATH="
  set "CUDA_PATH="
  set "CUDA_HOME="
  if exist "runtime\Lib\site-packages\triton\runtime\tcc\tcc.exe" set "CC=%~dp0runtime\Lib\site-packages\triton\runtime\tcc\tcc.exe"
  set "TRITON_CACHE_DIR=%~dp0data\kernel-cache"
  "runtime\python.exe" -m hi3d_workbench --open
) else if exist ".appenv\Scripts\python.exe" (
  ".appenv\Scripts\python.exe" -m hi3d_workbench --open
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m hi3d_workbench --open
) else (
  echo Run scripts\Setup.ps1 with Python 3.10 first.
  pause
)
