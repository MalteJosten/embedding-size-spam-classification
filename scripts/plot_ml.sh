#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare reg="$1"

# Plots output/ml_classification/summary.csv. Pass a regressor name to plot only that one.
python3 src/eval/plot_ml_candles.py ${reg:+--reg "$reg"}
python3 src/eval/plot_ml_scatter.py ${reg:+--reg "$reg"}

# Per-embedding/regressor bars over results.csv (all regressors in one figure).
python3 src/eval/plot_ml_bars.py --metric accuracy --aggr mean
