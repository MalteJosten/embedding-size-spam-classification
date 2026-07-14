import json
import os
import argparse

def get_best_for_model(model, metric, test, bounds):
    result_data = {}

    for file_name in os.listdir(os.path.join("res", "cluster", "kmeans", model)):
        if file_name.startswith("out_") and file_name.endswith(f"{test}.json"):
            if bounds:
                k = file_name.split("_")[2]

                if k < bounds[0] or k > bounds[1]:
                    continue

            with open(os.path.join("res", "cluster", "kmeans", model, file_name), "r") as result_file:
                file_data = json.loads(result_file.read())
                k = file_name.split("_")[2]
                result_data[k] = file_data["counts"][metric]

    best_k = { "k": -1, metric: -1 }

    for k, m in result_data.items():
        if m > best_k[metric]:
            best_k = { "k": k, metric: m }

    return best_k


def get_best_all(metric, test, bounds):
    models = ["lml6", "lml12", "mpnet", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]

    for model in models:
        best = get_best_for_model(model, metric, test, bounds)
        print(model, best)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--metric", type=str, required=True)
    argparser.add_argument("--test", type=str, required=True)
    argparser.add_argument("--bounds", type=int, nargs=2)
    args = argparser.parse_args()

    if args.model:
        get_best_for_model(args.model, args.metric, args.test, args.bounds)
    else:
        get_best_all(args.metric, args.test, args.bounds)