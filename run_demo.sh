#!/bin/bash
# Presentation demo. Usage: ./run_demo.sh [photo_or_folder ...]
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

echo "### 1. Score on the held-out test set"
$PY repro/demo.py --score

echo; echo "### 2. Random unseen test images"
$PY repro/demo.py --show

echo; echo "### 3. The model's worst mistakes"
$PY repro/demo.py --errors --show

if [ "$#" -gt 0 ]; then
  echo; echo "### 4. Your own photos"
  $PY classify.py "$@" --show
fi
