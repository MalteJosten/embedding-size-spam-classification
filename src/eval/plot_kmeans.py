import os
import json
import math
import argparse
import matplotlib.pyplot as plt

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--metric", type=str, required=True)
    arg_parser.add_argument("--bounds", type=int, nargs=2)
    arg_parser.add_argument("--single", type=str)
    args = arg_parser.parse_args()

    models = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]
    tests = [args.single] if args.single else ["test"]

    metric = args.metric
    metrics = {}

    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 500, 1000, 2000, 5000]
    if args.bounds:
        ks = [k for k in ks if k >= args.bounds[0] and k <= args.bounds[1]]

    for test in tests:
        for model in models:
            data = load_data(model, test, ks)

            metrics_int = []
            for k in ks:
                tp = data[k]["counts"]["tp"]
                tn = data[k]["counts"]["tn"]
                fp = data[k]["counts"]["fp"]
                fn = data[k]["counts"]["fn"]

                if metric == "acc":
                    metrics_int.append((tp + tn) / (tp+tn+fp+fn))
                elif metric == "g-mean":
                    tpr = (tp/(tp+fn))
                    tnr = (tn/(tn+fp))
                    g_mean = math.sqrt(tpr * tnr)
                    metrics_int.append(g_mean)

            metrics[f"{model}_{test}"] = metrics_int

    if args.single:
        fig, ax = plt.subplots(figsize=(7, 5))

        for model in models:
            ax.plot(ks, metrics[f"{model}_{test}"], marker='x', label=f"{model}")
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel("G-Mean")
        ax.grid(True, linestyle='--', alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1.001))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.show() 
        plt.savefig(f"res/cluster/kmeans/kmeans_{metric}_{ks[0]}-{ks[-1]}_{test}.png", dpi=300, bbox_inches="tight")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        axes = axes.flatten()

        labels = ["(a)", "(b)"]

        for i, ax1 in enumerate(axes):
            test = tests[i]
            label = f"{labels[i]} Test: {test}"

            for model in models:
                ax1.plot(ks, metrics[f"{model}_{test}"], marker='x', label=f"{model}")
            ax1.set_xlabel('k')
            ax1.set_ylabel(f"{metric}")
            ax1.grid(True, linestyle='--', alpha=0.6)

            ax1.text(0.5, -0.1, label, transform=ax1.transAxes, va='top', ha='center')

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.subplots_adjust(bottom=0.2)
        plt.suptitle(f"kMeans ({metric})", fontsize=16)
        plt.show() 
        plt.savefig(f"res/cluster/kmeans/kmeans_{metric}_{ks[0]}-{ks[-1]}.png")


def load_data(model, test_set, ks):
    kmeans_data = {}
    for k in ks:
        with open(os.path.join("res", "cluster", "kmeans", model, f"out_{model}_{k}_{test_set}.json"), "r") as res_file:
            data = json.loads(res_file.read())
            kmeans_data[k] = data

    return kmeans_data


if __name__ == "__main__":
    main()