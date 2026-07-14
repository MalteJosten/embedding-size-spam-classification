#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a models=("mxbai")
declare -a sizes=(2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 125 150 175 200 250 500 1000 2000 5000)

for model in "${models[@]}"
do
        for k in "${sizes[@]}"
        do
                echo "Doing: ${model} and ${k}"
                python3 src/classification/cluster_kmeans.py --model $model --k $k
        done
done
