import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("output", "ml_classification")

MODELS = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]


def plot_graph(summary, regressor):

    # aggregate the per-hyperparameter-config rows of one regressor into one candle per model
    stats = (
        summary[summary["regressor"] == regressor]
            .groupby("embedding")
            .agg(
                mean=("g_mean_mean", "mean"),
                std=("g_mean_std", "mean"),
                min=("g_mean_min", "min"),
                max=("g_mean_max", "max"),
            )
            .reindex(MODELS)
    )

    # a single-seed config has no standard deviation
    stats["std"] = stats["std"].fillna(0)

    x = np.arange(len(MODELS))
    mean, std = stats["mean"].to_numpy(), stats["std"].to_numpy()
    min_v, max_v = stats["min"].to_numpy(), stats["max"].to_numpy()

    fig, ax = plt.subplots(figsize=(7,5))
    tick_width = 0.15
    body_width = 0.4

    for i in range(len(x)):

        # mean
        ax.plot(
            [x[i] - body_width/2, x[i] + body_width/2],
            [mean[i], mean[i]],
            color="black",
            linewidth=2
        )

        # vertical line
        ax.plot([x[i], x[i]], [min_v[i], max_v[i]],
                color="black",
                linewidth=2,
                zorder=-2)

        # min
        ax.plot([x[i] - tick_width, x[i] + tick_width],
                [min_v[i], min_v[i]],
                color="black", linewidth=2)

        # max
        ax.plot([x[i] - tick_width, x[i] + tick_width],
                [max_v[i], max_v[i]],
                color="black", linewidth=2)

        # body
        lower = mean[i] - std[i]
        upper = mean[i] + std[i]
        ax.add_patch(
            plt.Rectangle(
                (x[i] - body_width/2, lower),
                body_width,
                upper - lower,
                facecolor="lightgray",
                edgecolor="black"
            )
        )

    ax.set_ylabel("G-Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=45, ha="right")
    ax.set_ylim(0.865, 1)
    ax.grid(True, axis="y", color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"candle_{regressor}.png"), dpi=300, bbox_inches="tight")
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
