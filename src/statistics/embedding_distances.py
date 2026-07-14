import os
import json
import time
import numpy as np
import argparse
from classification.my_classifiers import get_classifier
from util.progress_bar import progressBarTime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Relative path to train embeddings JSON")
    parser.add_argument("--model", type=str, required=True, help="Embedder used.")
    parser.add_argument("--classifier", type=str, required=True, help="Type of classifier")

    args = parser.parse_args()

    tests = ["spam_test.json", "ham_test.json"]

    start_time = time.time()

    train_data = {}
    with open(args.train, "r") as train_embeds:
        train_data = json.loads(train_embeds.read())

    results = {}
    for test in tests:
        test_data = {}
        with open(os.path.join("embeddings", f"{args.model}_{test}"), "r") as test_file:
            test_data = json.loads(test_file.read())

        classifier = get_classifier(args.classifier)
        result = {}
        result["best_vals"] = []

        for _, test_embed in progressBarTime(test_data.items(), prefix = 'Progress:', start_time=start_time, length = 50):
            for train_id, train_embed in train_data.items():
                classifier.compare(test_embed, train_id, train_embed)

            summary = classifier.get_summary()
            result["best_vals"].append(summary[0])
            classifier.reset()

        result["min"] = np.min(result.get("best_vals", []))
        result["max"] = np.max(result.get("best_vals", []))
        result["avg"] = np.average(result.get("best_vals", []))
        result["med"] = np.median(result.get("best_vals", []))

        results[test] = result

    os.makedirs(os.path.join("output/embeddings"), exist_ok=True)    
    with open(f"output/embeddings/distances_{args.model}_{args.classifier}_{args.train.split("/")[1].split("_")[1]}.json", "w") as out_file:
        out_file.write(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()