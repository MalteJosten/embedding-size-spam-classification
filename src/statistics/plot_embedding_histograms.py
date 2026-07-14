import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

OUTPUT_DIR = os.path.join("output", "embeddings")

NUM_SAMPLES = 10000


def plot_combined(models):
    fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for model_i, model in enumerate(models):
        data = load_test_x(model)
        cos_sim = 1 - pdist(data, metric='cosine')

        ax = axes[model_i]
        ax.hist(cos_sim, bins=50, color='skyblue', edgecolor='black', density=True)
        ax.set_title(model)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')

        if model_i == 0 or model_i == 5:
            ax.set_ylabel('Density')

        ax.set_xlim(-0.4, 1.0)
        ax.set_ylim(0, 6.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distance_histograms_combined.png"))
    plt.close(fig)


def plot_sidebyside(models):
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True)

    for model_i, model in enumerate(models):
        data = load_test_x(model)
        euk_dist = pdist(data, metric='euclidean')

        ax = axes[0, model_i]
        ax.hist(euk_dist, bins=50, density=True)
        ax.set_title(model)
        ax.set_xlabel('Euclidean Distance')

        if model_i == 0:
            ax.set_ylabel('Density')

        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0, 6.0)

    for model_i, model in enumerate(models):
        data = load_test_x(model)
        cos_sim = 1 - pdist(data, metric='cosine')

        ax = axes[1, model_i]
        ax.hist(cos_sim, bins=50, density=True)
        ax.set_xlabel('Cosine Similarity')

        if model_i == 0:
            ax.set_ylabel('Density')

        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(0, 6.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distance_histograms_sidebyside.png"))
    plt.close(fig)


def load_test_x(model):
    data = np.load(os.path.join("embeddings_bin", f"{model}.npz"))
    return data["test_x"][:NUM_SAMPLES]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = sorted([e.split('.')[0] for e in os.listdir("embeddings_bin") if e.endswith("npz")])

    plot_combined(models)
    plot_sidebyside(["nomic768", "para"])


if __name__ == "__main__":
    main()
