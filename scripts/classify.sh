#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare model="$1"

declare -a classifiers=("euclidean")
declare -a tests=("ham_test" "spam_test")
declare -a thresholds=(1.6 1.7 1.8 1.9 2.0)

for classifier in "${classifiers[@]}"
do
    for test in "${tests[@]}"
    do
        for threshold in "${thresholds[@]}"
        do
            echo "Doing: ${model}-${classifier} (Threshold: ${threshold})"
            echo "Test: ${test}"
            python3 src/classification/classify.py --train embeddings/${model}_spam_train.json --test embeddings/${model}_${test}.json --classifier $classifier --threshold $threshold --output res/${model}/out_${model}_${test}_${classifier}_${threshold}.json
        done
    done
done
