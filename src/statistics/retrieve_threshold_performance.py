import argparse
import os
import math
import json
from sklearn.metrics import auc, precision_recall_curve


def main(args):
    model = args.model
    test = args.test
    classifiers = ["cosine", "euclidean"]

    performance_data = {}

    for classifier in classifiers:
        relevant_files = []
        ham_thresholds = []
        spam_thresholds = []

        for f in os.listdir(os.path.join("res", model)):
            if f.startswith(f"out_{model}_ham_test_{classifier}"):
                ham_thresholds.append(float(f.split("_")[-1:][0][:-5]))
                relevant_files.append(f)
            elif f.startswith(f"out_{model}_spam_test_{classifier}"):
                spam_thresholds.append(float(f.split("_")[-1:][0][:-5]))
                relevant_files.append(f)

        relevant_files.sort()
        ham_thresholds.sort()
        spam_thresholds.sort()

        if ham_thresholds != spam_thresholds:
            print("[ERROR] Ham thresholds do not match spam thresholds! Aborting...")
            exit()

        for th in ham_thresholds:
            ham_data = {}
            with open(os.path.join("res", model, f"out_{model}_ham_test_{classifier}_{th}.json"), "r") as ham_file:
                ham_data = json.loads(ham_file.read())

            spam_data = {}
            with open(os.path.join("res", model, f"out_{model}_spam_test_{classifier}_{th}.json"), "r") as ham_file:
                spam_data = json.loads(ham_file.read())

            tp = spam_data["counts"]["spam"]
            tn = ham_data["counts"]["ham"]
            fp = ham_data["counts"]["spam"]
            fn = spam_data["counts"]["ham"]

            classifications = {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            }


            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)

            precision = tp / (tp + fp)
            accuracy = round((tp + tn) / (tp + tn + fp + fn), 4)
            balanced_accurary = round((tpr + tnr) / 2, 4)
            f1 = round((2 * tp) / (2 * tp + fp + fn), 4)
            g_mean = math.sqrt(tpr * tnr)

            ground_truth = []
            prediction = []
            ground_truth.extend([1] * tp)
            prediction.extend([1] * tp)
            ground_truth.extend([1] * fn)
            prediction.extend([0] * fn)
            ground_truth.extend([0] * tn)
            prediction.extend([0] * tn)
            ground_truth.extend([0] * fp)
            prediction.extend([1] * fp)
            auc_precision, auc_recall, _ = precision_recall_curve(ground_truth, prediction)
            auc_prc = round(auc(auc_recall, auc_precision), 4)


            performance_data[f"{classifier}_{th}"] = {
                "classifications": classifications,
                "tpr": round(tpr, 4),
                "tnr": round(tnr, 4),
                "precision": round(precision, 4),
                "recall": round(tnr, 4),
                "f1_score": f1,
                "g-mean": g_mean,
                "acc": accuracy,
                "ba_acc": balanced_accurary,
                "auc_prc": auc_prc
            }

    with open(os.path.join("output", "results", f"{model}_{test}_summary.json"), "w") as res_file:
        res_file.write(json.dumps(performance_data, indent=2))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--test", type=str, required=True)

    args = argparser.parse_args()

    main(args)