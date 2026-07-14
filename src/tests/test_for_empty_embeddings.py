import sys
import os
import json

def main():
    model_name = sys.argv[1]

    with open(os.path.join("embeddings", f"embeddings_{model_name}.json"), "r") as embed_file:
        embeddings = json.load(embed_file)

    counter = 0
    for k, v in embeddings.items():
        if len(v) == 0:
            print(k)

            counter += 1

            with open(os.path.join("mails", k), "r") as mail_file:
                content = mail_file.read()
                print(content + "\n\n----------------------------------\n\n")

    print(counter)

if __name__ == "__main__":
    main()