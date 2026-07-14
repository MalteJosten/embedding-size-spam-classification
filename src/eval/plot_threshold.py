import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    if args["mode"] == "1x2":
        plot_1_by_2(args)
    elif args["mode"] == "2x2":
        plot_2_by_2(args)
    else:
        plot_single(args)

def plot_1_by_2(args):
    model = args.model
    tests = ["test"]
    classifiers = ["cosine", "euclidean"]

    thresholds = []
    accuracies = []
    g_means = []

    for test in tests:
        for classifier in classifiers:
            summaries = {}
            with open(os.path.join("output", "results", f"{model}_{test}_summary.json"), "r") as sum_file:
                data = json.loads(sum_file.read())

                for trial in data.keys():
                    if trial.startswith(classifier):
                        summaries[trial] = data[trial]

    
            trials = list(summaries.keys())
            trials.sort()

            thresholds_int = []
            accuracies_int = []
            g_means_int = []

            for trial in trials:
                th = trial.split("_")[1]

                thresholds_int.append(float(th))
                accuracies_int.append(summaries[trial]["acc"])
                g_means_int.append(summaries[trial]["g-mean"])

            thresholds.append(thresholds_int)
            accuracies.append(accuracies_int)
            g_means.append(g_means_int)

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes = axes.flatten()

    labels = ["(a)", "(b)"]

    for i, ax in enumerate(axes):
        classifier = classifiers[i % 2]
        if "cosine" in classifier:
            label = f"{labels[i]} Metric: Cosine Similarity"
        else:
            label = f"{labels[i]} Metric: Euclidean (L2) Distance"

        x = thresholds[i]
        y1 = accuracies[i]
        y2 = g_means[i]
        
        ax.plot(x, y1, marker='x', color='tab:blue', label='Accuracy')
        ax.set_xlabel('Thresholds')
        if i == 0:
            ax.set_ylabel('Classification Performance')
        ticks = [round(x, 1) for x in np.arange(0, 2.05, 0.1)]
        ax.set_xticks(ticks)
        if classifier == "euclidean":
            ax.set_xticklabels(ticks, rotation=45, ha="right")
            ax.tick_params(axis='y')

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1)
        ax.plot(x, y2, marker='x', color='tab:red', label='G-Mean')
        ax.text(0.5, -0.15, label, transform=ax.transAxes, va='top', ha='center')
        
        leg_lines, leg_labels = ax.get_legend_handles_labels()
        ax.legend(leg_lines, leg_labels, loc='upper left')

    plt.tight_layout()
    plt.show() 
    plt.savefig(f"output/results/{model}_1x2.png", dpi=300, bbox_inches="tight")

def plot_2_by_2(args):
    model = args.model
    tests = ["test"]
    classifiers = ["cosine", "euclidean"]

    thresholds = []
    accuracies = []
    g_means = []

    for test in tests:
        for classifier in classifiers:
            summaries = {}
            with open(os.path.join("output", "results", f"{model}_{test}_summary.json"), "r") as sum_file:
                data = json.loads(sum_file.read())

                for trial in data.keys():
                    if trial.startswith(classifier):
                        summaries[trial] = data[trial]

    
            trials = list(summaries.keys())
            trials.sort()

            thresholds_int = []
            accuracies_int = []
            g_means_int = []

            for trial in trials:
                th = trial.split("_")[1]

                thresholds_int.append(float(th))
                accuracies_int.append(summaries[trial]["acc"])
                g_means_int.append(summaries[trial]["g-mean"])

            thresholds.append(thresholds_int)
            accuracies.append(accuracies_int)
            g_means.append(g_means_int)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, ax in enumerate(axes):
        classifier = classifiers[i % 2]
        label = f"{labels[i]} Test: {tests[int(i/2)]}, Classifier: {classifier}"

        x = thresholds[i]
        y1 = accuracies[i]
        y2 = g_means[i]
        
        ax.plot(x, y1, marker='x', color='tab:blue', label='Accuracy')
        ax.set_xlabel('Thresholds')
        ticks = [round(x, 1) for x in np.arange(0, 2.05, 0.1)]
        ax.set_xticks(ticks)
        if classifier == "euclidean":
            ax.set_xticklabels(ticks, rotation=45, ha="right")
            ax.tick_params(axis='y')

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1)
        ax.plot(x, y2, marker='x', color='tab:red', label='G-Mean')
        ax.text(0.5, -0.25, label, transform=ax.transAxes, va='top', ha='center')
        
        leg_lines, leg_labels = ax.get_legend_handles_labels()
        ax.legend(leg_lines, leg_labels, loc='upper left')

    plt.tight_layout()
    plt.show() 
    plt.savefig(f"output/results/{model}_2x2.png")


def plot_single(args):
    model = args.model
    test = args.test
    classifier = args.classifier 
    summaries = {}

    with open(os.path.join("output", "results", f"{model}_{test}_summary.json"), "r") as sum_file:
        data = json.loads(sum_file.read())

        for trial in data.keys():
            if trial.startswith(classifier):
                summaries[trial] = data[trial]

    thresholds = []
    accuracies = []
    g_means = []
    
    trials = list(summaries.keys())
    trials.sort()

    for trial in trials:
        th = trial.split("_")[1]

        thresholds.append(float(th))
        accuracies.append(summaries[trial]["acc"])
        g_means.append(summaries[trial]["g-mean"])

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(thresholds, accuracies, marker='x', color='tab:blue', label='Accuracy')
    ax.plot(thresholds, g_means, marker='x', color='tab:red', label='G-Mean')
    ax.set_xlabel('Thresholds')
    ax.set_ylabel('Classification Performance')

    if classifier == "cosine":
        ticks = [round(x, 1) for x in np.arange(0, 1.05, 0.1)]
        ax.set_xticklabels(ticks)

    if classifier == "euclidean":
        ticks = [round(x, 1) for x in np.arange(0, 2.05, 0.1)]
        ax.set_xticklabels(ticks, rotation=45, ha="right")


    ax.tick_params(axis='y')
    ax.set_xticks(ticks)

    ax.tick_params(axis='y')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.6)

    leg_lines, leg_labels = ax.get_legend_handles_labels()
    ax.legend(leg_lines, leg_labels, loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig(f"output/results/{model}_{test}_{classifier}_acc_g-mean.png", dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--classifier", type=str)
    argparser.add_argument("--test", type=str)
    argparser.add_argument("--mode", type=str, choices=["single", "1x2", "2x2"], default="single")

    args = argparser.parse_args()

    main(args)