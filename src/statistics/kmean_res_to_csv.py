import csv
import json
import os


def main():
    csv_headers = ["model", "k", "TPR", "TNR", "ACC", "GMEAN"]
    models = ["lml6", "lml12", "mpnet", "nomic64", "nomic128", "nomic256", "nomic512", "nomic768", "para"]

    entries = []

    for model in models:
        ks = set([])
        for file_name in os.listdir(os.path.join("res", "cluster", "kmeans", model)):
            if file_name.startswith("out_") and file_name.endswith(".json"):
                k = file_name.split("_")[2]
                ks.add(int(k))


        ks = list(ks)
        ks.sort()
        for k in ks:
            with open(os.path.join("res", "cluster", "kmeans", model, f"out_{model}_{k}_test.json"), "r") as result_file:
                file_data = json.loads(result_file.read())
                x = file_data["counts"]

                tpr = round((x["tp"] / (x["tp"] + x["fn"]))*100, 2)
                tnr = round((x["tn"] / (x["tn"] + x["fp"]))*100, 2)
                acc = round(x["acc"]*100, 2)
                gmean = round(x["g_mean"]*100, 2)

            entry = [model, k, tpr, tnr, acc, gmean]
            entries.append(entry)

    with open(os.path.join("output", "cluster", "kmeans_res.csv"), "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)

        for entry in entries:
            csv_writer.writerow(entry)


if __name__ == "__main__":
    main()
