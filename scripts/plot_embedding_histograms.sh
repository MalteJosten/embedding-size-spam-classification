#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

# Distance histograms over embeddings_bin/ -> output/embeddings/.
# Needs scripts/embeddings_to_bin.sh to have run first.
python3 src/statistics/plot_embedding_histograms.py
