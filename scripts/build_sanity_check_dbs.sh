#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs

declare -a models=("nomic768" "nomic512" "nomic256" "nomic128" "nomic64" "mpnet" "lml6" "lml12")
declare -a class=("cosine" "euclidean")

for m in "${models[@]}"
do
    for c in "${class[@]}"
    do
        echo "Starting with $m and $c"
        python3 src/eval/build_sanity_check_db.py "$m" "$c" > "logs/sanity_db_${m}_${c}.log"
        echo "Done with $m and $c"
    done
done
