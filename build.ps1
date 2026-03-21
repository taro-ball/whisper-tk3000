$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    python -m venv .venv
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
& $venvPython -m PyInstaller --noconfirm whisper_transcriber.spec

$sourceDir = Join-Path $repoRoot "dist\whisper_transcriber"
if (-not (Test-Path $sourceDir)) {
    throw "Expected build output was not created: $sourceDir"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputDir = Join-Path $repoRoot "dist\${timestamp}_whisper_transcriber"
if (Test-Path $outputDir) {
    Remove-Item $outputDir -Recurse -Force
}

Move-Item -Path $sourceDir -Destination $outputDir -Force
Write-Output "Build complete: $outputDir"
Start-Process explorer.exe $outputDir
