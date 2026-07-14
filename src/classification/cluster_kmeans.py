import os
import json
import math
import argparse
import numpy as np
from sklearn.cluster import KMeans

SEED = 42

def main(model, k):
    ham_train_data = {}
    with open(os.path.join("embeddings", f"{model}_ham_train.json"), "r") as train_embeds:
        ham_train_data = json.loads(train_embeds.read())

    spam_train_data = {}
    with open(os.path.join("embeddings", f"{model}_spam_train.json"), "r") as train_embeds:
        spam_train_data = json.loads(train_embeds.read())

    ham_data = list(ham_train_data.values())
    spam_data = list(spam_train_data.values())

    ham_res, ham_centroids, ham_stats = create_clusters(k, ham_data, list(ham_train_data.keys()))
    spam_res, spam_centroids, spam_stats = create_clusters(k, spam_data, list(spam_train_data.keys()))

    results = {}
    results["model"] = model
    results["k"] = k
    results["ham_labels"] = ham_res
    results["spam_labels"] = spam_res
    results["ham_centroids"] = ham_centroids
    results["spam_centroids"] = spam_centroids
    results["stats"] = {
        "ham": ham_stats,
        "spam": spam_stats
    }

    res_file_dir = os.path.join("res", "cluster", "kmeans", model)
    save_path = os.path.join("res", "cluster", "kmeans", model, f"{model}_{k}.json")
    if not os.path.exists(res_file_dir):
        os.makedirs(res_file_dir)

    with open(save_path, "w") as label_file:
        label_file.write(json.dumps(results, indent=2))
        print("Wrote results to", save_path)


def create_clusters(k, data, data_ids):
    clusterer = KMeans(n_clusters=k, random_state=SEED)
    labels = clusterer.fit_predict(data)
    labels = list(labels)
    labels = [int(x) for x in labels]

    res = {}

    for i, k in enumerate(data_ids):
        res[k] = labels[i]

    counts = {}
    for label in res.values():
        counts[label] = counts.get(label, 0) + 1
        data_keys = list(counts.keys())
        data_keys.sort()

        data_labels = {i: counts[i] for i in data_keys}

    stats = {
        "clusters": data_labels,
        "cluster_cnt": len(list(counts.keys())),
        "avg_cluster_size": math.floor(np.average(list(counts.values()))),
    }

    centroids_raw = clusterer.cluster_centers_
    centroids = {i: list(center) for i, center in enumerate(centroids_raw)}

    return res, centroids, stats


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--k", type=int, required=True)
    args = argparser.parse_args()

    main(args.model, args.k)