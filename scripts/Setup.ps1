param([string]$Python = 'python', [switch]$DownloadModels)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "命令失败: $Exe" }
}
$version = & $Python -c 'import sys; print("%d.%d" % sys.version_info[:2])'
if ($version -ne '3.10') { throw '需要 Python 3.10。请使用 -Python 指定解释器。' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { Invoke-Checked $Python @('-m','venv','.venv') }
$workerPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
Invoke-Checked $workerPython @('-m','pip','install','--upgrade','pip')
Invoke-Checked $workerPython @('-m','pip','install','torch==2.11.0+cu128','torchvision==0.26.0+cu128','--index-url','https://download.pytorch.org/whl/cu128')
Invoke-Checked $workerPython @('-m','pip','install','-r','requirements-worker-windows.txt')
Invoke-Checked $workerPython @('-m','pip','install','.[workbench]')
if (-not (Test-Path -LiteralPath 'upstream\Hi3Dv1.0\sparc3d')) {
    Invoke-Checked 'git' @('clone','https://github.com/hi3d-official/Hi3Dv1.0.git','upstream/Hi3Dv1.0')
}
if (Test-Path -LiteralPath 'upstream\Hi3Dv1.0\.git') {
    Invoke-Checked 'git' @('-C','upstream/Hi3Dv1.0','checkout','--detach','23dcc684bc6e1fab86ecafb99ec5270edad122f9')
}
if ($DownloadModels) { Invoke-Checked $workerPython @('-m','hi3d_workbench.model_cli','download') }
Invoke-Checked $workerPython @('-m','hi3d_workbench.worker','--diagnose','--probe')
Write-Host '安装完成。双击 Start-Workbench.cmd 启动。原生 Windows 使用 SDPA 兼容模式。'
