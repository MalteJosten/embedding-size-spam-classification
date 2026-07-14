import sys
import os
import json
import sqlite3
import numpy as np


def set_up_results():
    try:
        with open(os.path.join("res", "sanity_check.json"), "r") as sanity_file:
            results = json.load(sanity_file)
    except:
        results = []

    return results


def set_up_database(db_name: str) -> sqlite3.Cursor:
    sql_conn = sqlite3.connect(os.path.join("dbs", f"{db_name}.db"))
    sql_conn.row_factory = sqlite3.Row
    cursor = sql_conn.cursor()

    print(f"Connected to DB '{db_name}'")

    return cursor


def retrieve_entries(cursor: sqlite3.Cursor, classifier: str):
    cursor.execute(f"SELECT * from eval WHERE classifier LIKE '{classifier}'")
    matches = cursor.fetchall()
    return matches


def analyse_entries(db_name, entries):
    if len(entries) < 1:
        return {}

    summary = {}
    summary["model_name"] = db_name
    summary["classifier"] = entries[0]["classifier"]

    matches_exact = 0
    matches_not_exact = 0
    original_distances = []
    original_in_top_5_cnt  = 0
    original_in_top_10_cnt = 0
    
    for entry in entries:
        if entry["mail_id"] == entry["best_id"]:
            matches_exact += 1
        else:
            matches_not_exact += 1

        if entry["orig_in_top_5"] == 1:
            original_in_top_5_cnt += 1

        if entry["orig_in_top_10"] == 1:
            original_in_top_10_cnt += 1

        original_distances.append(entry["orig_dist"])

    recall = matches_exact / len(entries)

    summary["matches_exact"]     = matches_exact
    summary["matches_not_exact"] = matches_not_exact
    summary["matches_recall"]    = recall

    summary["orig_dist_avg"]         = np.average(original_distances)
    summary["orig_dist_med"]         = np.median(original_distances)
    summary["orig_in_top_5"]         = original_in_top_5_cnt
    summary["orig_in_top_5_recall"]  = original_in_top_5_cnt / len(entries)
    summary["orig_in_top_10"]        = (original_in_top_5_cnt + original_in_top_10_cnt)
    summary["orig_in_top_10_recall"] = (original_in_top_5_cnt + original_in_top_10_cnt) / len(entries)

    return summary


def write_summary_to_file(results, summary):
    for i in range(len(results)):
        if results[i]["model_name"] == summary["model_name"] and results[i]["classifier"] == summary["classifier"]:
            del results[i]
    
    results.append(summary)

    with open(os.path.join("res", "sanity_check.json"), "w") as sanity_file:
        sanity_file.write(json.dumps(results, indent=2))


def main():
    db_name = sys.argv[1]
    classifier = sys.argv[2]

    cursor = set_up_database(db_name)
    results = set_up_results()
    entries = retrieve_entries(cursor, classifier)
    summary = analyse_entries(db_name, entries)

    print(summary)

    write_summary_to_file(results, summary)


if __name__ == "__main__":
    main()