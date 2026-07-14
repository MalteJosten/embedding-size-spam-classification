import os
import json
import argparse
import numpy as np

def main(args):
    model = args.model

    for embed_file_name in os.listdir("embeddings"):
        if embed_file_name.startswith(model):
            with open(os.path.join("embeddings", embed_file_name), "r") as embed_file:
                embeddings = json.loads(embed_file.read())
                results = test_for_normalized(embeddings)

                print(f"{embed_file_name}\n---------\nNormalized: {results['norm']}\nNot normalized: {results['not_norm']}\n")


def test_for_normalized(embeddings):
    results = {}
    results["norm"] = 0
    results["not_norm"] = 0

    for embed in embeddings.values():
        norm_value = round(np.linalg.norm(embed), 0)
        if norm_value == 1:
            results["norm"]  = results["norm"] + 1
        else:
            print(norm_value)
            results["not_norm"]  = results["not_norm"] + 1

    return results

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)

    args = argparser.parse_args()
    main(args)