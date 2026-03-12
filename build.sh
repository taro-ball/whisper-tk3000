#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm ipconfig_runner.spec

# rm -f ./dist/ipconfig_runner.zip
# powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path '.\\dist\\ipconfig_runner\\*' -DestinationPath '.\\dist\\ipconfig_runner.zip'"

echo "Build complete."
