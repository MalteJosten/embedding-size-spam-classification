#!/bin/bash
set -o pipefail
set -e

# Builds the GitHub Release assets from the working data directories.
#
# Produces, in dist/:
#   embeddings-<model>.tar.gz   one per model, the four splits as JSON
#   res-classification.tar.gz   threshold results (res/<model>/) and kNN (res/knn/)
#   res-clustering.tar.gz       k-Means results (res/cluster/kmeans/)
#   SHA256SUMS.txt
#
# Every asset unpacks from the repository root, so `tar -xzf <asset>` restores
# the directory layout the scripts expect.

cd "$(dirname "$0")/.."

declare dist="dist"
declare stage
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

declare -a models=("lml6" "lml12" "mpnet" "mxbai" "nomic64" "nomic128" "nomic256" "nomic512" "nomic768" "para")
declare -a splits=("spam_train" "spam_test" "ham_train" "ham_test")

# pigz is a drop-in parallel gzip; fall back to gzip when it is missing.
declare zip="gzip"
command -v pigz >/dev/null && zip="pigz"

mkdir -p "$dist"

# --- embeddings: one asset per model -----------------------------------------
# The .json files are the source of truth. The .tar.gz files sitting next to
# them in embeddings/ are per-file archives from an earlier workflow and are
# deliberately not shipped.

for model in "${models[@]}"
do
    declare -a files=()
    for split in "${splits[@]}"
    do
        declare json="embeddings/${model}_${split}.json"
        if [ ! -f "$json" ]; then
            echo "Missing $json - run embeddings/decompress.sh first." >&2
            exit 1
        fi
        files+=("$json")
    done

    echo "Packing embeddings-${model}.tar.gz"
    tar -cf - "${files[@]}" | $zip -6 > "${dist}/embeddings-${model}.tar.gz"
done

# --- res: classification ------------------------------------------------------
# Threshold results per model plus the kNN outputs. These are plain JSON and
# PNG, with no archives mixed in.

declare -a classification=()
for model in "${models[@]}"
do
    classification+=("res/${model}")
done
classification+=("res/knn")

echo "Packing res-classification.tar.gz"
tar -cf - "${classification[@]}" | $zip -6 > "${dist}/res-classification.tar.gz"

# --- res: clustering ----------------------------------------------------------
# cluster_kmeans.py and classify_kmeans.py write plain .json here, but part of
# the existing tree only survives as per-file <name>.json.tar.gz archives from an
# earlier workflow. Take the .json when it exists and fall back to the archive
# otherwise, so this works both on a regenerated tree and on the current one.
# Either way the release ships plain JSON, never archives inside an archive.

echo "Staging k-Means results"
declare kmeans="${stage}/res/cluster/kmeans"

for model in "${models[@]}"
do
    declare src="res/cluster/kmeans/${model}"
    mkdir -p "${kmeans}/${model}"

    for json in "${src}"/*.json
    do
        [ -e "$json" ] || continue
        cp "$json" "${kmeans}/${model}/"
    done

    for archive in "${src}"/*.tar.gz
    do
        [ -e "$archive" ] || continue
        declare json="${archive%.tar.gz}"
        [ -e "$json" ] && continue
        # Each archive stores its payload as <model>/<name>.json, so unpacking
        # into the kmeans root is what lands it in the per-model directory.
        tar -xzf "$archive" -C "$kmeans"
    done
done

# The aggregate k-Means plots sit directly in res/cluster/kmeans/, alongside the
# per-model directories rather than inside them.
cp res/cluster/kmeans/*.png "$kmeans"/

echo "Packing res-clustering.tar.gz"
tar -C "$stage" -cf - res | $zip -6 > "${dist}/res-clustering.tar.gz"

# --- checksums ----------------------------------------------------------------

echo "Writing SHA256SUMS.txt"
(cd "$dist" && sha256sum ./*.tar.gz > SHA256SUMS.txt)

echo
echo "Assets in ${dist}/:"
du -h "${dist}"/*.tar.gz | sort -k2
du -ch "${dist}"/*.tar.gz | tail -1
