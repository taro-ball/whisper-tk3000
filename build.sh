#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v cygpath >/dev/null 2>&1; then
  echo "build.sh requires Git Bash with cygpath available." >&2
  exit 1
fi

powershell.exe -ExecutionPolicy Bypass -File "$(cygpath -w "$script_dir/build.ps1")"
