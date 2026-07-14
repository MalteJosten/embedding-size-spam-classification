import os
import json
import sys
import sqlite3
import time
from embeds.my_sentence_transformers import get_model
from classification.my_classifiers import get_classifier


def set_up_database(name):
    sql_conn = sqlite3.connect(os.path.join("dbs", f"{name}.db"))
    cursor = sql_conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS eval (
                    mail_id TEXT NOT NULL,
                    best_val FLOAT NOT NULL,
                    best_id TEXT NOT NULL,
                    worst_val FLOAT NOT NULL,
                    worst_id TEXT NOT NULL,
                    orig_dist FLOAT NOT NULL,
                    orig_in_top_5 INTEGER NOT NULL,
                    orig_in_top_10 INTEGER NOT NULL,
                    avg FLOAT,
                    med FLOAT,
                    classifier TEXT NOT NULL,
                   PRIMARY KEY (mail_id, classifier)
                   )
                   ''')

    return sql_conn, cursor


def compare_embeddings(classifier, mail_id, mod_embed, embeddings):
    classifier.reset()
    classifier.set_mail_id(mail_id)

    for k_embed, v_embed in embeddings.items():
        if len(mod_embed) != len(v_embed):
            continue

        classifier.compare(mod_embed, k_embed, v_embed)

    return classifier.get_summary()


def main():
    start_time = time.time()
    model_name = sys.argv[1] if len(sys.argv) >= 2 else "lml6"
    model = get_model(model_name)
    
    classifier_name = sys.argv[2] if len(sys.argv) >= 3 else "euclidean"
    classifier = get_classifier(classifier_name)


    sql_conn, cursor = set_up_database(model_name)

    mod_mails = list(os.listdir("mod_mails"))
    mod_mails.sort()

    mail_cnt = 0

    with open(os.path.join("embeddings", f"embeddings_{model_name}.json"), "r") as embed_file:
        embeddings = json.load(embed_file)

    for mod_mail in mod_mails:
        content = ""
        with open(os.path.join("mod_mails", mod_mail), "r") as mail_file:
            content = "\n\n".join(mail_file.read().split("\n\n")[1:])

        if content == "":
            continue

        mod_embedding = model.embed(content)
        
        summary = compare_embeddings(classifier, mod_mail, mod_embedding, embeddings)

        try:
            cursor.execute('INSERT INTO eval (mail_id, best_val, best_id, worst_val, worst_id, orig_dist, orig_in_top_5, orig_in_top_10, avg, med, classifier) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (mod_mail, *summary, classifier_name))
            sql_conn.commit()

            print(f"Progress: {mail_cnt} of {len(mod_mails)}")
            mail_cnt += 1
        except sqlite3.IntegrityError as e:
            print(e)
            continue

    sql_conn.close()
    model.clean()

    print(f"This took {time.time() - start_time} seconds.")

if __name__ == "__main__":
    main()