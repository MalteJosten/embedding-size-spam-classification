#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a metrics=("acc" "g-mean")
declare -a classifiers=("cosine" "euclidean")

for metric in "${metrics[@]}"
do
    for classifier in "${classifiers[@]}"
    do
        echo "Plotting kNN results: metric=${metric} classifier=${classifier}"
        python3 src/eval/plot_knn.py --metric "$metric" --test test --classifier "$classifier"
    done

    echo "Plotting kNN results: metric=${metric} (both classifiers)"
    python3 src/eval/plot_knn.py --metric "$metric" --test test
done
