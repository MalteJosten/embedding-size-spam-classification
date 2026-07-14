import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("output", "ml_classification")

MODELS = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]


def plot_graph(summary, regressor):

    # average the per-hyperparameter-config rows of one regressor into one point per model
    mean = (
        summary[summary["regressor"] == regressor]
            .groupby("embedding")["g_mean_mean"]
            .mean()
            .reindex(MODELS)
    )

    x = np.arange(len(MODELS))
    y = mean.to_numpy()

    fig, ax = plt.subplots(figsize=(7,5))

    plt.scatter(x, y, color='black', s=50, marker='o', alpha=0.7)
    ax.set_ylabel("G-Mean")
    ax.set_xticks(x)
    ax.set_ylim(0.87, 1)
    ax.set_xticklabels(MODELS, rotation=45, ha="right")
    ax.grid(True, axis="y", color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{regressor}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(args):
    summary = pd.read_csv(os.path.join(OUTPUT_DIR, "summary.csv"))

    regressors = [args.reg] if args.reg else sorted(summary["regressor"].unique())

    for regressor in regressors:
        plot_graph(summary, regressor)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--reg", type=str)

    args = argparser.parse_args()
    main(args)
