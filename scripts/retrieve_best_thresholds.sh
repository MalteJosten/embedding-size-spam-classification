#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a models=("lml6" "lml12" "mpnet" "nomic64" "nomic128" "nomic256" "nomic512" "nomic768" "para")
declare -a tests=("test")
declare -a classifiers=("cosine" "euclidean")
declare -a metrics=("acc" "g-mean")

for mo in "${models[@]}"
do
    for c in "${classifiers[@]}"
    do
        for t in "${tests[@]}"
        do
            for me in "${metrics[@]}"
            do
                echo "Working on: $mo-$c-$t-$me"
                python3 src/statistics/get_best_threshold.py --model $mo --test $t --classifier $c --metric $me
            done
        done
    done
done
