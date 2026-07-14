import itertools
import os
import pandas as pd
import numpy as np
import sklearn.metrics

import ml_classifiers
from util.seeds import set_global_seed
from util.formatting import lstr, pstr

OUTPUT_DIR = os.path.join("output", "ml_classification")


def test_model(model_predict, x, true_y):

    pred_y = model_predict(x)

    if not np.all(np.isin(pred_y, [0, 1])):
        raise ValueError(f"Prediction returned non zero/one values: {np.unique(pred_y)}")

    if pred_y.dtype != np.float32:
        raise ValueError(f"Prediction returned non float32 dtype: {pred_y.dtype}")

    precision = sklearn.metrics.precision_score(true_y, pred_y)
    recall    = sklearn.metrics.recall_score(true_y, pred_y)

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true_y, pred_y).ravel()

    accuracy    = (tp + tn) / (tn + fp + fn + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    g_mean = np.sqrt(recall * tnr)

    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "tnr": tnr,
        "precision": precision,
        "recall": recall,
        "g_mean": g_mean,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def summarize(results):

    summary = results.drop(columns=['seed'])
    summary = (
        summary.groupby(['embedding', 'embedding_size', 'regressor', 'hyperparams'])
            .agg(['mean', 'min', 'max', 'std'])
            .reset_index()
        )
    summary.columns = [
        '_'.join(col).strip('_') for col in summary.columns.to_flat_index()
    ]

    return summary


def main():

    # configuration
    embeddings = sorted([e.split('.')[0] for e in os.listdir("embeddings_bin") if e.endswith("npz")])

    regressors = ["svm", "logistic_regression", "random_forest", "naive_bayes", "mlp"]

    regressors_seed_dependent = ["random_forest", "mlp"]

    hyper_params = {
        "svm": {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        },
        "logistic_regression": {
            "max_iter": [50, 100, 500],
        },
        "random_forest": {
            "n_estimators": [80, 100, 200],
        },
        "naive_bayes": {
            "method": ["gaussian", "multinomial", "bernoulli", "categorical"],
        },
        "mlp": {
            "num_layers": [2, 3],
            "dropout": [0.1, 0.2, 0.3],
        }
    }

    hp_configs = {
        regressor: [
            dict(zip(hyper_params[regressor].keys(), values))
            for values in itertools.product(*hyper_params[regressor].values())
        ] for regressor in regressors
    }

    seeds = [823, 950, 686, 319, 84] # optained by [random.randint(0, 1000) for _ in range(5)]

    results = pd.DataFrame(columns=[
        "embedding", "embedding_size", "regressor", "hyperparams", "seed",
        "accuracy", "tpr", "tnr","precision", "recall", "g_mean",
        "tn", "fp", "fn", "tp"
    ])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(OUTPUT_DIR, "results.csv")

    # embedding loop
    for embedding_i, embedding in enumerate(embeddings):

        data = np.load(os.path.join("embeddings_bin", f"{embedding}.npz"))
        embedding_size = data["train_x"].shape[1]

        info = { "embedding": embedding, "embedding_size": embedding_size }
        print(f"[{embedding_i+1}/{len(embeddings)}] {embedding}")

        # regressor loop
        for regressor_i, regressor in enumerate(regressors):

            info["regressor"] = regressor
            print(f"  [{regressor_i+1}/{len(regressors)}] {regressor}")

            # hyperparams loop
            for hp_config_i, hp_config in enumerate(hp_configs[regressor]):

                info["hyperparams"] = "_".join([f"{p}-{v}" for p, v in hp_config.items()])

                seeds_to_use = seeds if regressor in regressors_seed_dependent else seeds[:1]

                # seed loop
                for seed_i, seed in enumerate(seeds_to_use):

                    info["seed"] = seed
                    set_global_seed(seed)
                    ml_classifiers.set_tf_seed(seed)

                    print(f"\r    [{hp_config_i+1}/{len(hp_configs[regressor])}] {lstr(info['hyperparams'])} {pstr(seed_i, len(seeds_to_use))}", end="", flush=True)

                    model = ml_classifiers.train_model(regressor, hp_config, seed, embedding_size,
                                                       data["train_x"], data["train_y"], data["test_x"], data["test_y"])

                    results.loc[len(results)] = info | test_model(model, data["test_x"], data["test_y"])

                    results.to_csv(results_file, index=False, float_format="%.4f")

                print(f"\r    [{hp_config_i+1}/{len(hp_configs[regressor])}] {lstr(info['hyperparams'])} {pstr(len(seeds_to_use), len(seeds_to_use))}", flush=True)

    summarize(results).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
