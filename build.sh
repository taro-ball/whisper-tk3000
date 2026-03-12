#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm ipconfig_runner.spec

timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="./dist/${timestamp}_ipconfig_runner"

rm -rf "$output_dir"
mv ./dist/ipconfig_runner "$output_dir"

echo "Build complete: $output_dir"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process explorer.exe (Resolve-Path '${output_dir//\//\\}')"
