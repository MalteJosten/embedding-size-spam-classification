import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.manifold import TSNE


def main(model, mode):
    ham_train_data = {}
    with open(os.path.join("embeddings", f"{model}_ham_train.json"), "r") as train_embeds:
        ham_train_data = json.loads(train_embeds.read())

    spam_train_data = {}
    with open(os.path.join("embeddings", f"{model}_spam_train.json"), "r") as train_embeds:
        spam_train_data = json.loads(train_embeds.read())

    ham_data = np.array(list(ham_train_data.values()))
    spam_data = np.array(list(spam_train_data.values()))

    components = {
        "2d": 2,
        "3d": 3
    }

    print("Starting with ham")
    reduced_ham = TSNE(n_components=components[mode], learning_rate='auto', init='random', perplexity=5).fit_transform(ham_data)
    print("Starting with spam")
    reduced_spam = TSNE(n_components=components[mode], learning_rate='auto', init='random', perplexity=5).fit_transform(spam_data)

    print("Plotting")

    if mode == "2d":
        plt.figure(figsize=(6, 5))
        plt.scatter(reduced_ham[:, 0], reduced_ham[:, 1], c='blue', s=1, alpha=0.2)
        plt.scatter(reduced_spam[:, 0], reduced_spam[:, 1], c='orange', s=1, alpha=0.2)

        plt.title("t-SNE embedding of ham and spam data") 
        plt.grid(True)
        plt.savefig(os.path.join("output/embeddings", f"tsne_{model}.png"))
    else:
        views = [
            (20, 30),
            (20, 120),
            (20, 210),
            (20, 300)
        ]

        # Create subplots
        fig = plt.figure(figsize=(12, 10))

        for i, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            ax.scatter(reduced_ham[:, 0], reduced_ham[:, 1], reduced_ham[:, 2], c='blue', s=1, alpha=0.2)
            ax.scatter(reduced_spam[:, 0], reduced_spam[:, 1], reduced_spam[:, 2], c='orange', s=1, alpha=0.2)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"View: elev={elev}, azim={azim}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        plt.savefig(os.path.join("output/embeddings", f"tsne_{model}_3d.png"))
    print("Saved plot")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--mode", type=str, required=True)
    args = argparser.parse_args()

    main(args.model, args.mode)