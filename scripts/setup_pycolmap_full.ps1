param(
  [Parameter(Mandatory = $true)]
  [string]$WheelPath,

  [string]$Python = ".\.venv\Scripts\python.exe",

  [switch]$NoDeps
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonInput = if ([System.IO.Path]::IsPathRooted($Python)) { $Python } else { Join-Path $ProjectRoot $Python }
$PythonPath = Resolve-Path $PythonInput
$Wheel = Resolve-Path $WheelPath
$UvCache = Join-Path $ProjectRoot ".uv-cache"

$env:UV_CACHE_DIR = $UvCache

$installArgs = @("pip", "install", "--python", $PythonPath.Path, "--force-reinstall")
if ($NoDeps) {
  $installArgs += "--no-deps"
}
$installArgs += $Wheel.Path

Write-Host "Installing PyCOLMAP wheel:"
Write-Host "  $($Wheel.Path)"
uv @installArgs

$sitePackages = & $PythonPath.Path -c "import site; print(site.getsitepackages()[0])"
$pycolmapLibs = Join-Path $sitePackages "pycolmap.libs"
$torchLib = Join-Path $sitePackages "torch\lib"

if ((Test-Path $torchLib) -and (Test-Path $pycolmapLibs)) {
  Write-Host "Copying cuDNN 9 runtime DLLs into pycolmap.libs"
  Get-ChildItem -Path $torchLib -Filter "cudnn*64_9.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $pycolmapLibs $_.Name) -Force
    Write-Host "  $($_.Name)"
  }
} else {
  Write-Host "Skipping cuDNN copy; torch/lib or pycolmap.libs not found."
}

Write-Host "Verifying PyCOLMAP capabilities"
& $PythonPath.Path -c @"
import pycolmap
print("pycolmap", pycolmap.__version__)
print("cuda_devices", pycolmap.get_num_cuda_devices())
print("extractors", sorted(pycolmap.FeatureExtractorType.__members__))
print("matchers", sorted(pycolmap.FeatureMatcherType.__members__))
print("ba_backends", sorted(pycolmap.BundleAdjustmentBackend.__members__))
missing = []
for name in ("ALIKED_N16ROT", "ALIKED_N32"):
    if name not in pycolmap.FeatureExtractorType.__members__:
        missing.append(name)
for name in ("ALIKED_BRUTEFORCE", "ALIKED_LIGHTGLUE"):
    if name not in pycolmap.FeatureMatcherType.__members__:
        missing.append(name)
if "CASPAR" not in pycolmap.BundleAdjustmentBackend.__members__:
    missing.append("CASPAR")
if pycolmap.get_num_cuda_devices() < 1:
    missing.append("CUDA_DEVICE")
if missing:
    raise SystemExit("Missing capabilities: " + ", ".join(missing))
"@
