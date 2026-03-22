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

$exePath = Join-Path $sourceDir "whisper_transcriber.exe"
if (-not (Test-Path $exePath)) {
    throw "Expected executable was not created: $exePath"
}

$warnPath = Join-Path $repoRoot "build\whisper_transcriber\warn-whisper_transcriber.txt"
if (-not (Test-Path $warnPath)) {
    throw "Expected PyInstaller warning report was not created: $warnPath"
}

$requiredTopLevelModules = @(
    "tkinter",
    "tkinter.ttk",
    "tkinter.font",
    "tkinter.filedialog",
    "tkinter.constants"
)
$warnLines = Get-Content $warnPath
$blockingWarnings = foreach ($line in $warnLines) {
    foreach ($module in $requiredTopLevelModules) {
        $modulePattern = [regex]::Escape($module)
        if ($line -match "missing module named '?$modulePattern'?\s+- imported by") {
            $line
            break
        }
    }
}
if ($blockingWarnings) {
    throw "PyInstaller reported startup-critical missing modules:`n$($blockingWarnings -join "`n")"
}

$smokeCommand = """$exePath"" --smoke-startup"
$smokeOutput = & cmd.exe /c $smokeCommand 2>&1
$smokeExitCode = $LASTEXITCODE
if ($smokeExitCode -ne 0) {
    throw "Packaged startup smoke failed with exit code ${smokeExitCode}:`n$($smokeOutput -join "`n")"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputDir = Join-Path $repoRoot "dist\${timestamp}_whisper_transcriber"
if (Test-Path $outputDir) {
    Remove-Item $outputDir -Recurse -Force
}

Move-Item -Path $sourceDir -Destination $outputDir -Force
Write-Output "Build complete: $outputDir"
