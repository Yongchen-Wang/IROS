#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required for this helper. Alternatively run: pip install -e '.[training]'"
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "iros-training"; then
  conda env create -f "${SCRIPT_DIR}/environment.yml"
fi

echo "Environment ready. Activate it with: conda activate iros-training"
echo "Then run: python ${PROJECT_ROOT}/training/scripts/smoke_test.py"
