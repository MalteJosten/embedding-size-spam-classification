import os
import json
import math
import argparse
import matplotlib.pyplot as plt


def main(args):
    if args.classifier:
        plot_1x1(args)
    else:
        plot_1x2(args)


def metric_label(metric):
    return "Accuracy" if metric == "acc" else "G-Mean"


def plot_1x1(args):
    models = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]
    classifier = args.classifier

    metric = args.metric
    metrics = {}
    ks = []

    for model in models:
        ham_data, spam_data, n_range = load_data(model, args.test, classifier)
        ks = n_range

        metrics_int = []
        for k in n_range:
            tp = spam_data[k]["spam"]
            tn = ham_data[k]["ham"]
            fp = ham_data[k]["spam"]
            fn = spam_data[k]["ham"]

            if metric == "acc":
                metrics_int.append((tp + tn) / (tp+tn+fp+fn))
            elif metric == "g-mean":
                tpr = (tp/(tp+fn))
                tnr = (tn/(tn+fp))
                g_mean = math.sqrt(tpr * tnr)
                metrics_int.append(g_mean)

        metrics[f"{model}_{args.test}_{classifier}"] = metrics_int

    fig, ax = plt.subplots(figsize=(7, 5))
    x = ks

    for model in models:
        ax.plot(x, metrics[f"{model}_{args.test}_{classifier}"], marker='x', label=f"{model}")

    ax.set_xlabel('Number of nearest neighbors (k)')
    ax.set_xticks(x)
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, linestyle='--', alpha=0.6)

    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1.01))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.21)
    plt.savefig(f"res/knn/knn_{args.test}_{classifier}_{metric}_1x1.png", dpi=300, bbox_inches='tight')


def plot_1x2(args):
    models = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]
    classifiers = ["cosine", "euclidean"]

    metric = args.metric
    metrics = {}
    ks = []

    for classifier in classifiers:
        for model in models:
            ham_data, spam_data, n_range = load_data(model, args.test, classifier)
            ks = n_range

            metrics_int = []
            for k in n_range:
                tp = spam_data[k]["spam"]
                tn = ham_data[k]["ham"]
                fp = ham_data[k]["spam"]
                fn = spam_data[k]["ham"]

                if metric == "acc":
                    metrics_int.append((tp + tn) / (tp+tn+fp+fn))
                elif metric == "g-mean":
                    tpr = (tp/(tp+fn))
                    tnr = (tn/(tn+fp))
                    g_mean = math.sqrt(tpr * tnr)
                    metrics_int.append(g_mean)

            metrics[f"{model}_{args.test}_{classifier}"] = metrics_int

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.5))
    axes = axes.flatten()

    panel_labels = ["(a)", "(b)"]

    for i, ax1 in enumerate(axes):
        classifier = classifiers[i % 2]
        label = f"{panel_labels[i]} Classifier: {classifier}"

        x = ks

        for model in models:
            ax1.plot(x, metrics[f"{model}_{args.test}_{classifier}"], marker='x', label=f"{model}")
        ax1.set_xlabel('# nearest neighbors')
        ax1.set_xticks(x)
        ax1.set_xticklabels(x)
        ax1.set_ylabel(metric_label(metric))
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.text(0.5, -0.15, label, transform=ax1.transAxes, va='top', ha='center')

    handles, lbls = ax1.get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.275)
    plt.savefig(f"res/knn/knn_{args.test}_{metric}_1x2.png", dpi=300, bbox_inches='tight')


def load_data(model, label, classifier):
    ham_data = {}
    with open(os.path.join("res", "knn", f"{model}_ham_test_{classifier}.json"), "r") as res_file:
        data = json.loads(res_file.read())
        ham_data = data["neighbors"]

    spam_data = {}
    with open(os.path.join("res", "knn", f"{model}_spam_test_{classifier}.json"), "r") as res_file:
        data = json.loads(res_file.read())
        spam_data = data["neighbors"]

    n_range = list(ham_data.keys())

    return ham_data, spam_data, n_range


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--metric", type=str, required=True, choices=["acc", "g-mean"])
    arg_parser.add_argument("--test", type=str, required=True)
    arg_parser.add_argument("--classifier", type=str, choices=["cosine", "euclidean"],
                            help="Plot a single classifier. If omitted, both are plotted side by side.")

    args = arg_parser.parse_args()
    main(args)