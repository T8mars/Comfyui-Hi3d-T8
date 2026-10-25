param([Parameter(Mandatory=$true)][string]$ComfyUI)
$ErrorActionPreference='Stop'
$sourceRoot=Split-Path -Parent $PSScriptRoot
$targetRoot=(Resolve-Path -LiteralPath $ComfyUI).Path
if (-not (Test-Path -LiteralPath (Join-Path $targetRoot 'main.py'))) { throw '指定目录不是 ComfyUI 根目录' }
$destination=Join-Path $targetRoot 'custom_nodes\ComfyUI-Hi3D'
if (Test-Path -LiteralPath $destination) { throw '节点目录已存在，请先检查现有版本；不会覆盖。' }
New-Item -ItemType Directory -Path $destination | Out-Null
foreach ($name in @('__init__.py','comfy_nodes.py','requirements.txt','pyproject.toml','README.md','README.zh-CN.md','NOTICE.md','LICENSE')) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $name) -Destination $destination
}
New-Item -ItemType Directory -Path (Join-Path $destination 'docs') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $sourceRoot 'docs\COMFYUI.md') -Destination (Join-Path $destination 'docs\COMFYUI.md')
foreach ($name in @('examples','hi3d_native','web')) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $name) -Destination $destination -Recurse
}
Write-Host "节点已安装到 $destination。用 ComfyUI 自身 Python 安装 requirements.txt，然后重新启动 ComfyUI。无需启动工作台；在 Hi3D Load Mesh 节点上传粗模。"
