import os
import json
import argparse
import numpy as np

def main(args):
    model = args.model
    test_file_names = ["ham_train", "ham_test", "spam_train", "spam_test"]

    for test_file_name in test_file_names:
        with open(os.path.join("embeddings", f"{model}_{test_file_name}.json"), "r") as embed_file:
            embeddings = json.loads(embed_file.read())
            new_embeddings = normalize(embeddings)

            with open(os.path.join("embeddings", f"{model}_{test_file_name}_normalized.json"), "w") as embed_file:
                embed_file.write(json.dumps(new_embeddings, indent=2))

        print(f"Normalized: {model}-{test_file_name}")

    for test_file_name in test_file_names:
        os.remove(os.path.join("embeddings", f"{model}_{test_file_name}.json"))
        os.rename(os.path.join("embeddings", f"{model}_{test_file_name}_normalized.json"), os.path.join("embeddings", f"{model}_{test_file_name}.json"))

        print(f"Renamed: {model}-{test_file_name}")

def normalize(embeddings):
    new_embeddings = {}
    for mail_id, embed in embeddings.items():
        new_embed = []
        old_sum_norm = np.linalg.norm(embed)
        for part in embed:
            new_embed.append(part / old_sum_norm)

        new_embeddings[mail_id] = new_embed

    return new_embeddings


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)

    args = argparser.parse_args()
    main(args)