#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare model="$1"

declare -a classifiers=("euclidean")
declare -a tests=("ham_test" "spam_test")

for classifier in "${classifiers[@]}"
do
    for test in "${tests[@]}"
    do
        echo "Doing: ${model}-${classifier}"
        echo "Test: ${test}"
        python3 src/classification/knn.py --model $model --test embeddings/${model}_${test}.json --classifier $classifier
    done
done
