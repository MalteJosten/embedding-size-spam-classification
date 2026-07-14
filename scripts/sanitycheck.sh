#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a models=("lml6" "nomic768" "nomic512" "nomic256" "nomic128" "nomic64" "lml12" "mpnet")
declare -a class=("cosine" "euclidean")

for m in "${models[@]}"
do
    for c in "${class[@]}"
    do
        echo "Starting with $m and $c"
        python3 src/eval/sanity_check.py "$m" "$c"
        echo "Done with $m and $c"
    done
done
