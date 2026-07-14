import os
import math
import json
import argparse
from classification.my_classifiers import CosineSimilarity

def main(model, test_set, k):
    ham_test_data = {}
    with open(os.path.join("embeddings", f"{model}_ham_test.json"), "r") as embed_file:
        ham_test_data = json.loads(embed_file.read())

    spam_test_data = {}
    with open(os.path.join("embeddings", f"{model}_spam_test.json"), "r") as embed_file:
        spam_test_data = json.loads(embed_file.read())

    ham_centroids = {}
    spam_centroids = {}
    with open(os.path.join("res", "cluster", "kmeans", model, f"{model}_{k}.json"), "r") as kmean_file:
        kmean = json.loads(kmean_file.read())
        ham_centroids = kmean["ham_centroids"]
        spam_centroids = kmean["spam_centroids"]


    ham_classifications, ham_counts = classify(ham_test_data, False, ham_centroids, spam_centroids)
    spam_classifications, spam_counts = classify(spam_test_data, True, ham_centroids, spam_centroids)

    tp = spam_counts["correct"]
    tn = ham_counts["correct"]
    fp = ham_counts["wrong"]
    fn = spam_counts["wrong"]
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    g_mean = round(math.sqrt((tp / (tp + fn)) * (tn / (tn + fp))), 4)

    results = {
        "ham_classifications": ham_classifications,
        "spam_classifications": spam_classifications,
        "test_set": test_set,
        "counts": {
            "spam": ham_counts["spam"] + spam_counts["spam"],
            "ham": ham_counts["ham"] + spam_counts["ham"],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "acc": acc,
            "g_mean": g_mean
        }
    }

    res_file_dir = os.path.join("res", "cluster", "kmeans", model)
    res_file_path = os.path.join("res", "cluster", "kmeans", model, f"out_{model}_{k}_{test_set}.json")
    if not os.path.exists(res_file_dir):
        os.makedirs(res_file_dir)

    with open(res_file_path, "w") as res_file:
        res_file.write(json.dumps(results, indent=2))
    print("Wrote results to", res_file_path)

def classify(test_data, is_spam, ham_centroids, spam_centroids):
    cos_sim = CosineSimilarity()
    classifications = {}
    ham_count = spam_count = 0
    correct = wrong = 0
    for m_id, embed in test_data.items():
        result = {}

        ham_match = {"cluster_id": -1, "score": math.inf * -1}
        for cluster_id, centroid in ham_centroids.items():
            similarity = cos_sim.get_distance(embed, centroid)
            if similarity > ham_match["score"]:
                ham_match["cluster_id"] = cluster_id
                ham_match["score"] = similarity

        spam_match = {"cluster_id": -1, "score": math.inf * -1}
        for cluster_id, centroid in spam_centroids.items():
            similarity = cos_sim.get_distance(embed, centroid)
            if similarity > spam_match["score"]:
                spam_match["cluster_id"] = cluster_id
                spam_match["score"] = similarity

        if spam_match["score"] >= ham_match["score"]:
            label = 1
            spam_count += 1
            if is_spam:
                correct += 1
            else:
                wrong += 1
        else:
            label = 0
            ham_count += 1
            if is_spam:
                wrong += 1
            else:
                correct += 1

        result = {
            "label": label,
            "ham_match": {
                "cluster_id": ham_match["cluster_id"],
                "score": ham_match["score"]
            },
            "spam_match": {
                "cluster_id": spam_match["cluster_id"],
                "score": spam_match["score"]
            }
        }

        classifications[m_id] = result
        counts = {
            "spam": spam_count,
            "ham": ham_count,
            "correct": correct,
            "wrong": wrong
        }

    return classifications, counts


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--test", type=str, required=True)
    argparser.add_argument("--k", type=int, required=True)
    args = argparser.parse_args()

    main(args.model, args.test, args.k)