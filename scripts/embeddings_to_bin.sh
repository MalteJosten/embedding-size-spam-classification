#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

# Packs every embeddings/*.json split into embeddings_bin/<model>.npz (train/test tensors).
python3 src/embeds/embeddings_to_bin.py
