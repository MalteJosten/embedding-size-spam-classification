import os
import json
import math
import numpy as np
import argparse

def main(k):
    if k:
        ks = [k]
    else:
        ks = [x + 1 for x in range(1, 10)]

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

        for k in ks:
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
                    
            perf["acc"] =  (perf["tp"] + perf["tn"])/(perf["tp"] + perf["fn"] + perf["tn"] + perf["fp"])
            perf["g_mean"] = math.sqrt((perf["tp"] / (perf["tp"] + perf["fn"])) * (perf["tn"] / (perf["tn"] + perf["fp"])))

            results[model][k] = perf

    diffs = {}
    all_stats = {}

    for k in ks:
        stats = {}

        diffs[k] = {"min": math.inf, "max": -math.inf, "diff": -math.inf}

        for model in models:
            stats[model] = results[model][k]["g_mean"]

        vals = list(stats.values())
        vals.sort()

        local_diff = vals[-1] - vals[0]
        cur_diff = diffs[k]

        if local_diff > cur_diff["diff"]:
            diffs[k] = {
                "min": vals[0],
                "max": vals[-1],
                "min_model": get_model_by_value(stats, vals[0]),
                "max_model": get_model_by_value(stats, vals[-1]),
                "diff": local_diff,
            }

        print("avg", np.average(vals))
        all_stats[k] = stats


    best = {"k": -1, "diff": -math.inf}
    for k, stats in diffs.items():
        if stats["diff"] > best["diff"]:
            best = {"k": k, "min": stats["min"], "max": stats["max"], "diff": stats["diff"], "min_model": stats["min_model"], "max_model": stats["max_model"]}

    print(best)


def get_model_by_value(stats, val):
    for model, g_mean in stats.items():
        if val == g_mean:
            return model

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--k", type=int)
    args = argparser.parse_args()

    main(args.k)

