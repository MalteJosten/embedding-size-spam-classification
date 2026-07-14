import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str, required=True)
    arg_parser.add_argument("--classifier", type=str, required=True)
    arg_parser.add_argument("--single", type=str, help="set name")
    arg_parser.add_argument("--combined", type=str, help="{test}")
    arg_parser.add_argument("--range", nargs="+", type=float, required=True, help="Threshold range")

    args = arg_parser.parse_args()

    if args.single:
        draw_single(args)
    elif args.combined:
        draw_combined(args)
    else:
        print("[ERROR] Use either --single or --combined")
        exit()


def load_results(model, classifier, label):
    results = []
    for res_json in os.listdir(os.path.join("res", model)):
        if res_json.startswith(f"out_{model}"):
            result = {}
            out_split = res_json.split("_")

            if out_split[-2] != classifier or "_".join(out_split[2:-2]) != label :
                continue

            result["model"]      = model
            result["classifier"] = classifier
            result["label"]      = label
            result["threshold"]  = ".".join(out_split[-1].split(".")[:-1])

            with open(os.path.join("res", model, res_json), "r") as out_file:
                out_data = json.loads(out_file.read())

                result["spam"] = out_data["counts"]["spam"]
                result["ham"]  = out_data["counts"]["ham"]

                results.append(result)

    return results


def draw_single(args):
    results = load_results(args.model, args.classifier, args.single)
    thresholds = args.range
    thresholds.sort()

    group_correct = []
    group_wrong   = []

    for th in thresholds:
        for res in results:
            if float(res["threshold"]) == float(th):
                if "spam" in res["label"]:
                    group_correct.append(res["spam"])
                    group_wrong.append(res["ham"])
                else:
                    group_correct.append(res["ham"])
                    group_wrong.append(res["spam"])

    x = np.arange(len(thresholds))

    plt.bar(x, group_correct, label="Correct")
    plt.bar(x, group_wrong, bottom=group_correct, label="Wrong") 

    plt.xticks(x, thresholds, rotation=45, ha="right")
    plt.xlabel("Thresholds")
    plt.ylabel("Classifications")
    plt.title(f"{args.model} - {args.classifier} - {args.single}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"res/{args.model}/{args.model}-{args.classifier}-{args.single}.png")
    plt.show()


def draw_combined(args):
    mode = args.combined
    ham  = load_results(args.model, args.classifier, "ham_test")
    spam = load_results(args.model, args.classifier, f"spam_test_{mode}")

    thresholds = args.range
    thresholds.sort()

    ham_correct = []
    ham_wrong = []
    spam_correct = []
    spam_wrong = []

    for th in thresholds:
        for res in ham:
            if float(res["threshold"]) == float(th):
                ham_correct.append(res["ham"])
                ham_wrong.append(res["spam"])

        for res in spam:
            if float(res["threshold"]) == float(th):
                spam_correct.append(res["spam"])
                spam_wrong.append(res["ham"])

    labels = thresholds
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
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel("Classifications")
    ax.set_xlabel("Thresholds")
    ax.set_title(f"{args.model} - {args.classifier} - combined_{mode}")
    ax.legend()
    
    plt.savefig(os.path.join("res", args.model, f"{args.model}-{args.classifier}-combined_{mode}.png"))
    plt.show()

if __name__=="__main__":
    main()