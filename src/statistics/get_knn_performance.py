import os
import json
import math
import argparse

def main(k):
    models = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]

    results = {}

    for model in models:
        perf = {
            "tp": 0,
            "fn": 0,
            "tn": 0,
            "fp": 0,
            "acc": 0,
            "g_mean": 0
        }

        results[model] = {}

        with open(os.path.join("res", "knn", f"{model}_ham_test_cosine.json"), "r") as res_file:
            res_data = json.loads(res_file.read())
            p = res_data["neighbors"][str(k)]
            perf["tn"] = p["ham"]
            perf["fp"] = p["spam"]

        with open(os.path.join("res", "knn", f"{model}_spam_test_cosine.json"), "r") as res_file:
            res_data = json.loads(res_file.read())
            p = res_data["neighbors"][str(k)]
            perf["tp"] = p["spam"]
            perf["fn"] = p["ham"]
                
        perf["acc"]    = (perf["tp"] + perf["tn"])/(perf["tp"] + perf["fn"] + perf["tn"] + perf["fp"])
        perf["g_mean"] = math.sqrt((perf["tp"] / (perf["tp"] + perf["fn"])) * (perf["tn"] / (perf["tn"] + perf["fp"])))

        print(f"[k={k}] {model}: {round(perf["g_mean"],4)}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--k", type=int, required=True)
    args = argparser.parse_args()

    main(args.k)

