import os
import json
import math
import numpy as np
import argparse

def main(k):
    if k:
        ks = [k]
    else:
        ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 500, 1000, 2000, 5000]

    models = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]

    diffs = {}
    all_stats = {}

    for k in ks:
        stats = {}

        diffs[k] = {"min": math.inf, "max": -math.inf, "diff": -math.inf}

        for model in models:
            with open(os.path.join("res", "cluster", "kmeans", model, f"out_{model}_{k}_test.json"), "r") as out_file:
                out_data = json.loads(out_file.read())
                stats[model] = out_data["counts"]["g_mean"]

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
                "diff": local_diff
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

