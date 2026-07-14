import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = os.path.join("output", "ml_classification")


# ATTENTION: mostly by AI
def bar_plot(data, metric, aggr_func_name):

    aggr_metric = getattr(data.groupby(['embedding', 'regressor'])[metric], aggr_func_name)().reset_index()
    aggr_metric = aggr_metric.sort_values(['embedding', 'regressor'])

    regressors = aggr_metric['regressor'].unique()
    palette = plt.get_cmap("tab10").colors[:len(regressors)]
    color_map = dict(zip(regressors, palette))

    y_positions = []
    embedding_centers = {}
    current_pos = 0

    for emb in aggr_metric['embedding'].unique():
        group = aggr_metric[aggr_metric['embedding'] == emb]
        start_pos = current_pos
        for _, row in group.iterrows():
            y_positions.append(current_pos)
            current_pos += 1
        end_pos = current_pos - 1
        embedding_centers[emb] = (start_pos + end_pos) / 2
        current_pos += 1

    aggr_metric['y_pos'] = y_positions

    plt.figure(figsize=(6, 10))
    for _, row in aggr_metric.iterrows():
        plt.barh(row['y_pos'], row[metric], color=color_map[row['regressor']])

    plt.yticks(list(embedding_centers.values()), list(embedding_centers.keys()))
    plt.gca().invert_yaxis()

    plt.xlim(aggr_metric[metric].min()-0.01, aggr_metric[metric].max()+0.01)
    plt.ylim(-1.5, current_pos - 0.5)

    handles = [plt.Rectangle((0,0),1,1, color=color_map[r]) for r in regressors]
    plt.legend(handles, regressors, title='Regressor', loc='upper left', frameon=True, fontsize=10)

    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"bars_{metric}_{aggr_func_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main(args):
    results = pd.read_csv(os.path.join(OUTPUT_DIR, "results.csv"))
    results = results.drop(columns=['seed'])

    bar_plot(results, args.metric, args.aggr)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--metric", type=str, default="accuracy", help="Column of results.csv to plot")
    argparser.add_argument("--aggr", type=str, default="mean", help="How to aggregate over hyperparams/seeds")

    args = argparser.parse_args()
    main(args)
