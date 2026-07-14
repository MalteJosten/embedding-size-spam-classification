import os
import json
import argparse

def main(k):
    models = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]
    stats = {}

    for model in models:
        with open(os.path.join("res", "cluster", "kmeans", model, f"out_{model}_{k}_test.json"), "r") as out_file:
            out_data = json.loads(out_file.read())
            stats[model] = out_data["counts"]["g_mean"]

    print(stats)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--k", type=int, required=True)
    args = argparser.parse_args()

    main(args.k)

