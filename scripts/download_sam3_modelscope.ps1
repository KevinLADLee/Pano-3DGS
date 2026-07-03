param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
if ($env:SAM3_MODEL_DIR) {
    $Sam3ModelDir = $env:SAM3_MODEL_DIR
} else {
    $Sam3ModelDir = Join-Path $RootDir "models\facebook\sam3"
}

New-Item -ItemType Directory -Force -Path $Sam3ModelDir | Out-Null
modelscope download --model facebook/sam3 --local_dir $Sam3ModelDir @ExtraArgs
