import os
import argparse
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join("output", "ml_classification")

MODELS = ["lml6", "lml12", "mpnet", "mxbai", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]


def table_row(summary, regressor, model):

    rows = summary[(summary["regressor"] == regressor) & (summary["embedding"] == model)]

    tpr    = np.average(rows["tpr_mean"])
    tnr    = np.average(rows["tnr_mean"])
    acc    = np.average(rows["accuracy_mean"])
    g_mean = np.average(rows["g_mean_mean"])

    # a regressor whose configs are all single-seed has no spread to report
    min_gmean = np.min(rows["g_mean_min"])
    max_gmean = np.max(rows["g_mean_max"])

    min_gmean = "—" if min_gmean == g_mean else round(min_gmean * 100, 2)
    max_gmean = "—" if max_gmean == g_mean else round(max_gmean * 100, 2)

    return (f"{model} & {round(tpr*100, 2)} & {round(tnr*100, 2)} & {round(acc*100, 2)} "
            f"& {round(g_mean*100, 2)} & {min_gmean} & {max_gmean} \\\\\\midrule")


def main(args):
    summary = pd.read_csv(os.path.join(OUTPUT_DIR, "summary.csv"))

    for model in MODELS:
        print(table_row(summary, args.reg, model))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--reg", type=str, required=True)

    args = argparser.parse_args()
    main(args)
