@echo off
setlocal
cd /d "%~dp0"
if exist "Hi3D-Workbench.exe" goto launch
echo Building the Windows console launcher...
powershell -NoProfile -File "%~dp0scripts\Build-Launcher.ps1" -OutputDirectory "%~dp0."
if exist "Hi3D-Workbench.exe" goto launch
echo Launcher could not be built. See the error above or use the complete package.
pause
exit /b 1
:launch
"%~dp0Hi3D-Workbench.exe" %*
