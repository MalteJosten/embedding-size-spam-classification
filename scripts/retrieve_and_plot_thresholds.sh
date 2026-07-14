#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a models=("mxbai")
declare -a tests=("test")

for m in "${models[@]}"
do
    for t in "${tests[@]}"
    do
        echo "Working on: $m-$t"
        python3 src/statistics/retrieve_threshold_performance.py --model $m --test $t
    done
done

for m in "${models[@]}"
do
    echo "Working on: $m"
    python3 src/eval/plot_threshold.py --model $m
done
