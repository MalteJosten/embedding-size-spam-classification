#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a models=("lml6" "lml12" "mpnet" "nomic64" "nomic128" "nomic256" "nomic512" "nomic768")
declare -a classifiers=("cosine" "euclidean")

for m in "${models[@]}"
do
    for c in "${classifiers[@]}"
    do
        echo "Working on: $m-$c"
        python3 src/statistics/embedding_distances.py --train embeddings/${m}_spam_train.json --model $m --classifier $c
    done
done
