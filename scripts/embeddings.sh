#!/bin/bash
set -o pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

declare -a labels=("mails/spam/train" "mails/ham/train" "mails/ham/test" "mails/spam/test")

for j in "${labels[@]}"
do
    python3 src/embeds/create_embeddings.py --model $1 --source $j --cuda $2
done
