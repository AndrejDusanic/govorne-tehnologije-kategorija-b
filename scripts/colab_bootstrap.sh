#!/usr/bin/env bash
set -euo pipefail

WITH_ANALYSIS=0
SKIP_PREPARE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-analysis)
      WITH_ANALYSIS=1
      shift
      ;;
    --skip-prepare)
      SKIP_PREPARE=1
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/colab_bootstrap.sh [--with-analysis] [--skip-prepare]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if ! command -v git-lfs >/dev/null 2>&1; then
  apt-get -qq update
  apt-get -qq install -y git-lfs
fi

git lfs install
git lfs pull
python -m pip install -q -r requirements.txt

if [[ "$SKIP_PREPARE" -eq 0 ]]; then
  python scripts/prepare_data.py
fi

if [[ "$WITH_ANALYSIS" -eq 1 ]]; then
  python scripts/analyze_data.py
fi

echo "Colab bootstrap completed."
