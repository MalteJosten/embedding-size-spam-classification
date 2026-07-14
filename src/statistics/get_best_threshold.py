import argparse
import os
import json
import math


def main(args):
    model = args.model
    test = args.test
    classifier = args.classifier
    metric = args.metric

    data = {}

    with open(os.path.join("output", "results", f"{model}_{test}_summary.json"), "r") as res_file:
        data = json.loads(res_file.read())

    best = {}    
    best_th = -1 

    for trial, summary in data.items():
        if classifier in trial:
            current_best_metric = best.get(metric, -math.inf)
            if summary[metric] > current_best_metric:
                best = summary
                best_th = trial.split("_")[1]

    print(best_th)
    print(f"{round(best["tpr"]*100, 2)} & {round(best["tnr"]*100, 2)} & {round(best["acc"]*100, 2)} & {round(best["g-mean"]*100,2)}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--test", type=str, required=True)
    argparser.add_argument("--classifier", type=str, required=True)
    argparser.add_argument("--metric", type=str, required=True)

    args = argparser.parse_args()

    main(args)