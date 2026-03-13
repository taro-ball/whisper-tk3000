#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm whisper_transcriber.spec

timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="./dist/${timestamp}_whisper_transcriber"

rm -rf "$output_dir"
mv ./dist/whisper_transcriber "$output_dir"

echo "Build complete: $output_dir"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process explorer.exe (Resolve-Path '${output_dir//\//\\}')"
