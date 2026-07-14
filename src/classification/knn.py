import os
import json
import time
import argparse
from classification.my_classifiers import get_classifier
from util.progress_bar import progressBar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Embedder model under test")
    parser.add_argument("--test", type=str, required=True, help="Relative path to test embeddings JSON")
    parser.add_argument("--classifier", type=str, required=True, help="Type of classifier")

    args = parser.parse_args()
    neighbors = [x for x in range(2,11)]

    test_name = "_".join(args.test.split("_")[1:])
    test_name = test_name.split(".")[0]
    output_path = os.path.join("res", "knn", f"{args.model}_{test_name}_{args.classifier}.json")

    start_time = time.time()

    ham_train_data = {}
    with open(os.path.join("embeddings", f"{args.model}_ham_train.json"), "r") as train_embeds:
        ham_train_data = json.loads(train_embeds.read())

    spam_train_data = {}
    with open(os.path.join("embeddings", f"{args.model}_spam_train.json"), "r") as train_embeds:
        spam_train_data = json.loads(train_embeds.read())

    test_data = {}
    with open(args.test, "r") as test_file:
        test_data = json.loads(test_file.read())

    classifier = get_classifier(args.classifier)

    ham_cnt  = [0]*len(neighbors)
    spam_cnt = [0]*len(neighbors)

    results = {}
    results["classifications"] = {}

    for test_id, test_embed in progressBar(test_data.items(), prefix = 'Progress:', suffix = 'Complete', length = 50):
        distances = {}

        # "collect" distances
        for train_id, train_embed in ham_train_data.items():
            dist = classifier.get_distance(test_embed, train_embed)
            distances.setdefault(dist, []).append({
                "id": train_id,
                "is_spam": False,
                "distance": dist
            })

        for train_id, train_embed in spam_train_data.items():
            dist = classifier.get_distance(test_embed, train_embed)
            distances.setdefault(dist, []).append({
                "id": train_id,
                "is_spam": True,
                "distance": dist
            })

        # classify
        for i in range(len(neighbors)):    
            n = neighbors[i]
            dist_keys = list(distances.keys())
            dist_keys = classifier.sort_distances(dist_keys)

            nearest_neigbors = []

            for key in dist_keys:
                if len(nearest_neigbors) == n:
                    break

                for datum in distances[key]:
                    nearest_neigbors.append(datum)
                    if len(nearest_neigbors) == n:
                        break

            int_spam_cnt = int_ham_cnt = 0
            for neighbor in nearest_neigbors:
                if neighbor["is_spam"]:
                    int_spam_cnt += 1
                else:
                    int_ham_cnt += 1
            
            if int_spam_cnt >= int_ham_cnt:
                label = True
                spam_cnt[i] = spam_cnt[i] + 1
            else:
                label = False
                ham_cnt[i] = ham_cnt[i] + 1

            result = {
                "id": test_id,
                "n": n,
                "is_spam": label,
                "nearest_neighbors": nearest_neigbors
            }

            results["classifications"][test_id] = result

    results["neighbors"] = {}
    for i in range(len(neighbors)):
        results["neighbors"][neighbors[i]] = {"spam": spam_cnt[i], "ham": ham_cnt[i]}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)    
    with open(output_path, "w") as out_file:
        out_file.write(json.dumps(results, indent=2))

    runtime = time.time() - start_time
    minutes = int(runtime // 60)
    seconds = int(runtime % 60)
    print(f"This took: {minutes}m{seconds}s")

main()