$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$portableProductName = "Whisper-TK3000"

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    python -m venv .venv
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
& $venvPython -m PyInstaller --noconfirm whisper_transcriber.spec

$appModulePath = Join-Path $repoRoot "whisper_tk3000\app.py"
if (-not (Test-Path $appModulePath)) {
    throw "Expected app module was not found: $appModulePath"
}

$appVersionMatch = Select-String -Path $appModulePath -Pattern '^\s*APP_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $appVersionMatch -or -not $appVersionMatch.Matches[0].Success) {
    throw "Could not determine APP_VERSION from $appModulePath"
}
$appVersion = $appVersionMatch.Matches[0].Groups[1].Value

$pythonArchBits = (& $venvPython -c "import struct; print(struct.calcsize('P') * 8)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $pythonArchBits) {
    throw "Could not determine Python architecture for portable archive naming."
}
$portableArchLabel = switch ($pythonArchBits) {
    "64" { "win64" }
    "32" { "win32" }
    default { throw "Unsupported Python architecture reported for build naming: $pythonArchBits" }
}

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

$portableArchiveName = "${portableProductName}-v${appVersion}-${portableArchLabel}-portable-${timestamp}.zip"
$portableArchivePath = Join-Path $repoRoot "dist\$portableArchiveName"
$zipStagingRoot = Join-Path $repoRoot "dist\_zip_stage\$timestamp"
$zipFolderPath = Join-Path $zipStagingRoot $portableProductName

if (Test-Path $portableArchivePath) {
    Remove-Item $portableArchivePath -Force
}

if (Test-Path $zipStagingRoot) {
    Remove-Item $zipStagingRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $zipFolderPath -Force | Out-Null
Get-ChildItem -Path $outputDir -Force | Copy-Item -Destination $zipFolderPath -Recurse -Force

try {
    Compress-Archive -Path $zipFolderPath -DestinationPath $portableArchivePath -Force
}
finally {
    if (Test-Path $zipStagingRoot) {
        Remove-Item $zipStagingRoot -Recurse -Force
    }
}

Write-Output "Build complete: $outputDir"
Write-Output "Portable archive: $portableArchivePath"
