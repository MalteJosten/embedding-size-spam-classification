#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a datasets=("ham_test" "ham_train" "spam_test" "spam_train" "ham_cleaned" "spam_cleaned")
declare -a scales=("lin" "log")

for d in "${datasets[@]}"
do
    for s in "${scales[@]}"
    do
        python3 src/statistics/plot_mail_lengths.py --label $d --scale $s
    done
done
