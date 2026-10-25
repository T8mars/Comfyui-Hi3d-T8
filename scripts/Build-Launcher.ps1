param([string]$OutputDirectory = '.', [switch]$CreateArchive)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) { throw 'The Windows .NET Framework 4 compiler was not found.' }
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($outputRoot) | Out-Null
$launcher = Join-Path $outputRoot 'Hi3D-Workbench.exe'
& $compiler /nologo /target:exe /platform:x64 /optimize+ "/out:$launcher" (Join-Path $PSScriptRoot 'launcher\WorkbenchLauncher.cs')
if ($LASTEXITCODE -ne 0) { throw 'Launcher compilation failed.' }
$stop = Join-Path $outputRoot 'Stop-Workbench.cmd'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launcher\Stop-Workbench.cmd') -Destination $stop -Force
$start = Join-Path $outputRoot 'Start-Workbench.cmd'
if ($outputRoot.TrimEnd('\') -ne $projectRoot.TrimEnd('\')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot 'Start-Workbench.cmd') -Destination $start -Force
}
if ($CreateArchive) {
    $versionMatch = [regex]::Match([System.IO.File]::ReadAllText((Join-Path $projectRoot 'hi3d_workbench\common.py')), 'VERSION = "([0-9.]+)"')
    if (-not $versionMatch.Success) { throw 'Program version was not found.' }
    $archive = Join-Path $outputRoot ('Hi3D-Workbench-launcher-' + $versionMatch.Groups[1].Value + '.zip')
    Compress-Archive -LiteralPath $launcher,$start,$stop -DestinationPath $archive -Force
}
Write-Output $launcher
