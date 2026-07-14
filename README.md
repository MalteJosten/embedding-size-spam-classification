# (ARTIFACT) Size Does Matter: The Impact of Embedding Models and Sizes on Spam Email Classification
This repository accompanies a paper (see [Citation](#citation) down below).

## Repository layout

```
.
├── src/                     # All Python source (importable via PYTHONPATH=src)
│   ├── pre_processing/      # Parse, anonymize and sample the raw mail corpus
│   ├── embeds/              # Embedding models + embedding generation
│   │   └── para/            # Vendored ParaNMT encoder
│   ├── classification/      # Distance-threshold, kNN and k-Means classifiers
│   ├── statistics/          # Metric aggregation / results → tables & CSV
│   ├── eval/                # Plotting and sanity-check utilities
│   ├── util/                # Small shared helpers (progress bars, counters)
│   └── tests/               # Sanity tests over the embeddings
├── scripts/                 # Shell orchestration for each pipeline stage
├── mails/                   # Anonymized email corpus (tracked, see below)
├── embeddings/              # Generated embeddings (Release asset, see DATA.md)
├── res/                     # Classification/clustering results (Release asset)
├── output/                  # Generated figures, tables and summaries
├── requirements.txt         # Runtime dependencies
├── pyproject.toml           # Packaging metadata (pip install -e .)
└── DATA.md                  # How to obtain the large data assets
```

## Embedding models

Models are referenced throughout the code and scripts by a short key:

| Key        | Model                                                   | Dim. |
|------------|---------------------------------------------------------|------|
| `lml6`     | `sentence-transformers/all-MiniLM-L6-v2`                | 384  |
| `lml12`    | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 |
| `mpnet`    | `sentence-transformers/all-mpnet-base-v2`               | 768  |
| `mxbai`    | `mixedbread-ai/mxbai-embed-large-v1`       | 64   |
| `nomic64`  | `nomic-ai/nomic-embed-text-v1.5`           | 64   |
| `nomic128` | `nomic-ai/nomic-embed-text-v1.5`           | 128  |
| `nomic256` | `nomic-ai/nomic-embed-text-v1.5`           | 256  |
| `nomic512` | `nomic-ai/nomic-embed-text-v1.5`           | 512  |
| `nomic768` | `nomic-ai/nomic-embed-text-v1.5`           | 768  |
| `para`     | ParaNMT sentence embedder (at `src/embeds/para/`) | 1024 |

## Dataset

The anonymized mail corpus lives under `mails/` and **is included in this
repository**. Email addresses are replaced with placeholders; each message is
stored as an individual `.eml` file.

| Split                      | Messages |
|----------------------------|----------|
| `mails/spam/train`         | 10,000   |
| `mails/spam/test`          | 5,000    |
| `mails/ham/train`          | 10,000   |
| `mails/ham/test`           | 5,000    |

The `mails/*.txt` files list the message IDs assigned to each split.

The **embeddings** (`embeddings/`) and **result files** (`res/`) are large and
are distributed as **GitHub Release assets** rather than committed to the
repository — see [`DATA.md`](DATA.md). You can also regenerate them from the
mail corpus with the scripts below.

## Installation

Requires **Python 3.12+**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e .
```

The models are downloaded from the Hugging Face Hub on first use. A CUDA-capable
GPU is recommended for embedding generation but not required.

## Usage

All shell scripts in `scripts/` are self-contained: they resolve the repository
root, activate `.venv` if present, and put `src/` on `PYTHONPATH`, so they can be
run from anywhere. Each stage can also be invoked directly, e.g.
`PYTHONPATH=src python3 src/classification/classify.py --help`.

**1. Generate embeddings** for every split with a given model (optionally pass a
CUDA device as the second argument):

```bash
scripts/embeddings.sh mxbai            # CPU
scripts/embeddings.sh mxbai cuda:0     # GPU
```

**2. Distance-threshold classification:**

```bash
scripts/classify.sh mxbai
```

**3. kNN classification:**

```bash
scripts/knn.sh mxbai
```

**4. k-Means clustering** (create clusters, then classify):

```bash
scripts/kmeans_create.sh
scripts/kmeans_classify.sh
```

**5. Aggregate metrics and plot results**, e.g.:

```bash
scripts/retrieve_and_plot_thresholds.sh
scripts/plot_thresholds.sh
scripts/plot_knn.sh
```

> Several orchestration scripts contain a hard-coded list of models / parameters
> at the top (e.g. `kmeans_create.sh`, `plot_thresholds.sh`); edit those lists to
> cover the models you want to run.

## Output

Running the pipeline populates:

- `res/` — per-message classification results and per-model summaries (JSON).
- `output/` — figures (`.png`), aggregated tables and CSVs used in the paper.

## Citation

If you use this code or data, please cite the accompanying paper.
```bibtex
@conference{josten2026,
  author       = {Malte Josten and Gérald Kämmerer and Arne Kummerow and Torben Weis},
  title        = {Size Does Matter: The Impact of Embedding Models and Sizes on Spam Email Classification},
  booktitle    = {Proceedings of the 23rd International Conference on Security and Cryptography - Volume 1},
  year         = {2026},
  pages        = {841--851},
  publisher    = {SciTePress},
  organization = {INSTICC},
  doi          = {},
  isbn         = {978-989-758-858-7},
  issn         = {2184--7711},
}

```

## License

Released under the [MIT License](LICENSE). © 2026 Malte Josten.