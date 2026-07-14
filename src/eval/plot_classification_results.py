import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--classifier", type=str, required=True)
    arg_parser.add_argument("--mode", type=str, required=True, help="{absolute|normalized}")
    arg_parser.add_argument("--label", type=str)

    args = arg_parser.parse_args()

    results = load_results(args.classifier, args.mode)
    draw(args.classifier, results, args.mode, args.label)


def load_results(classifier, mode):
    results = {}
    for file_name in os.listdir(os.path.join("res", classifier)):
        if file_name.endswith(".json"):
            label = file_name.split(".")[0]
            results[label] = {}

            with open(os.path.join("res", classifier, file_name), "r") as out_file:
                out_data = json.loads(out_file.read())

                if mode == "normalized":
                    results[label]["spam"] = out_data["counts"]["spam"]/len(out_data["classifications"])
                    results[label]["ham"]  = out_data["counts"]["ham"]/len(out_data["classifications"])
                elif mode == "absolute":
                    results[label]["spam"] = out_data["counts"]["spam"]
                    results[label]["ham"]  = out_data["counts"]["ham"]

    return results

def draw(classifier, results, mode, graph_label):
    labels = ["train", "test"]

    ham_correct  = []
    ham_wrong    = []
    spam_correct = []
    spam_wrong   = []

    ham_correct  = [results["ham_train"]["ham"], results["ham_test"]["ham"]]
    ham_wrong    = [results["ham_train"]["spam"], results["ham_test"]["spam"]]
    spam_correct = [results["spam_train"]["spam"], results["spam_test"]["spam"]]
    spam_wrong   = [results["spam_train"]["ham"], results["spam_test"]["ham"]]

    group_tp = np.array(spam_correct)
    group_fn = np.array(spam_wrong)

    group_tn = np.array(ham_correct)
    group_fp = np.array(ham_wrong)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(x - width/2, group_tp, width, label="TP")
    ax.bar(x - width/2, group_fn, width, bottom=group_tp, label="FN")

    ax.bar(x + width/2, group_tn, width, label="TN")
    ax.bar(x + width/2, group_fp, width, bottom=group_tn, label="FP")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Classifications")

    if graph_label:
        ax.set_title(f"{graph_label} - {mode}")
    else:
        ax.set_title(f"{classifier} - {mode}")
    ax.legend()

    plt.savefig(os.path.join("res", classifier, f"classification_{mode}.png"))
    plt.show()

main()