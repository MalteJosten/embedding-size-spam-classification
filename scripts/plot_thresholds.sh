#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a models=("lml6" "lml12" "mpnet" "mxbai" "nomic64" "nomic128" "nomic256" "nomic512" "nomic768" "para")
declare -a tests=("test")

for m in "${models[@]}"
do
    echo "Working on: $m"
    python3 src/eval/plot_threshold.py --model $m
done
