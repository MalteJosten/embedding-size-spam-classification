# Data assets

This project uses three categories of data:

| Data        | Location      | Size            | Where it lives                         |
|-------------|---------------|-----------------|----------------------------------------|
| Mail corpus | `mails/`      | 142 MB          | **In this repository**                 |
| Embeddings  | `embeddings/` | 1.2 GB (packed) | **GitHub Release asset** (not in repo) |
| Results     | `res/`        | 1.5 GB (packed) | **GitHub Release asset** (not in repo) |

The embeddings and results are too large to keep under version control, so they
are attached to the corresponding [GitHub Release][releases]. You can either
**download** them or **regenerate** them from the mail corpus.

[releases]: https://github.com/maltejosten/embedding-spam-classification/releases

## Mail corpus (`mails/`)

Included in the repository. Anonymized emails (`.eml`), one file per message,
organized into splits:

```
mails/
├── spam/train/            # 10,000 spam training messages
├── spam/test/             #  5,000 spam test messages
├── ham/train/             # 10,000 ham training messages
├── ham/test/              #  5,000 ham test messages
└── *.txt                  # message-ID lists for each split
```

## Release assets

Every asset unpacks from the repository root, so `tar -xzf <asset>` restores the
layout the scripts expect. `SHA256SUMS.txt` on the release page covers all of
them.

### Embeddings — one asset per model

`embeddings-<model>.tar.gz` contains the four splits of one model as JSON, named
`{model}_{split}.json` (splits: `spam_train`, `spam_test`, `ham_train`,
`ham_test`). Each file maps a message ID to its embedding vector. Download only
the models you need:

| Asset                        | Size   | Asset                         | Size   |
|------------------------------|--------|-------------------------------|--------|
| `embeddings-mxbai.tar.gz`    |  19 MB | `embeddings-nomic512.tar.gz`  | 141 MB |
| `embeddings-nomic64.tar.gz`  |  19 MB | `embeddings-nomic768.tar.gz`  | 211 MB |
| `embeddings-nomic128.tar.gz` |  36 MB | `embeddings-mpnet.tar.gz`     | 212 MB |
| `embeddings-nomic256.tar.gz` |  71 MB | `embeddings-para.tar.gz`      | 281 MB |
| `embeddings-lml12.tar.gz`    | 106 MB | `embeddings-lml6.tar.gz`      | 107 MB |

### Results — two assets

| Asset                       | Size    | Contents                                                      |
|-----------------------------|---------|---------------------------------------------------------------|
| `res-classification.tar.gz` |  474 MB | Threshold results (`res/<model>/`) and kNN results (`res/knn/`) |
| `res-clustering.tar.gz`     |  1.1 GB | k-Means assignments and classifications (`res/cluster/kmeans/`) |

Clustering is by far the largest asset and is only needed for the k-Means
analysis, so it can be skipped if you only care about the threshold and kNN
results.

## Download workflow

```bash
# From the repository root. Adjust the patterns to fetch only what you need.
gh release download v1.0.0 --pattern 'embeddings-*.tar.gz' --dir dist
gh release download v1.0.0 --pattern 'res-*.tar.gz'        --dir dist

for asset in dist/*.tar.gz; do tar -xzf "$asset"; done
```

This restores `embeddings/*.json` and the `res/` tree in place.

## Packed embeddings (`embeddings_bin/`)

The supervised ML stage does not read the JSON embeddings directly. It uses
`embeddings_bin/<model>.npz`, which holds the shuffled, labelled train/test
tensors (`train_x`, `train_y`, `test_x`, `test_y`) for one model. These are
**derived** from `embeddings/*.json` and are neither committed nor shipped as a
release asset — build them once with:

```bash
scripts/embeddings_to_bin.sh
```

## Regenerating instead of downloading

```bash
scripts/embeddings.sh <model> [cuda-device]
```

See the model keys in the [README](README.md#embedding-models). Once the
embeddings are in place, regenerate the results with `scripts/classify.sh`,
`scripts/knn.sh`, `scripts/kmeans_create.sh`, `scripts/kmeans_classify.sh` and
`scripts/classify_ml.sh`.

## Building (the release) assets

`scripts/build_release.sh` rebuilds every asset above into `dist/`, along with
`SHA256SUMS.txt`:

```bash
scripts/build_release.sh
```

It reads `embeddings/*.json` and the `res/` tree. Under `res/cluster/kmeans/`,
some results may only exist as per-file `.json.tar.gz` archives left over from an
earlier workflow; the script prefers the plain `.json` and falls back to the
archive when there is none, so it works whether the results were downloaded or
regenerated. The release always ships plain JSON.
