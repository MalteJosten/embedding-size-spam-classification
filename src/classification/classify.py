import os
import json
import time
import argparse
from classification.my_classifiers import get_classifier
from util.progress_bar import progressBar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Relative path to train embeddings JSON")
    parser.add_argument("--test", type=str, required=True, help="Relative path to test embeddings JSON")
    parser.add_argument("--classifier", type=str, required=True, help="Type of classifier")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold used for classification")
    parser.add_argument("--output", type=str, required=True, help="Relative path to output file")

    args = parser.parse_args()

    start_time = time.time()

    train_data = {}
    with open(args.train, "r") as train_embeds:
        train_data = json.loads(train_embeds.read())

    test_data = {}
    with open(args.test, "r") as test_file:
        test_data = json.loads(test_file.read())

    classifier = get_classifier(args.classifier)
    threshold = args.threshold

    ham_cnt = spam_cnt = 0
    results = {}
    results["classifications"] = {}
    results["best_matches"] = {}

    for test_id, test_embed in progressBar(test_data.items(), prefix = 'Progress:', suffix = 'Complete', length = 50):
        for train_id, train_embed in train_data.items():
            classifier.compare(test_embed, train_id, train_embed)

        if classifier.is_better_than_threshold(classifier.best_val, threshold):
            results["classifications"][test_id] = 1
            spam_cnt += 1
        else:
            results["classifications"][test_id] = 0
            ham_cnt += 1

        results["best_matches"][test_id] = classifier.get_best_data()

        classifier.reset()

    print(f"Spam: {spam_cnt} // Ham: {ham_cnt}")
    results["counts"] = { "spam": spam_cnt, "ham": ham_cnt }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)    
    with open(args.output, "w") as out_file:
        out_file.write(json.dumps(results, indent=2))

    runtime = time.time() - start_time
    minutes = int(runtime // 60)
    seconds = int(runtime % 60)
    print(f"This took: {minutes}m{seconds}s")

main()