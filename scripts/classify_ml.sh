#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

# Trains every ML classifier / hyperparameter / seed combination over embeddings_bin/
# and writes output/ml_classification/{results,summary}.csv. Requires the "ml" extra.
python3 src/classification/classify_ml.py
