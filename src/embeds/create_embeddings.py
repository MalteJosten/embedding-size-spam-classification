import os
import json
import time
import argparse
from util.progress_bar import progressBarTime
from embeds.my_sentence_transformers import *


def embed(sen_trans: SenTransformer, text: str):
    return sen_trans.embed(text)


def read_file(file):
    with open(file, "r") as read_file:
        file_content = read_file.read()
        return file_content


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True, help="model_name")
    argparser.add_argument("--source", type=str, required=True, help="Directory that contains mails to be embedded")
    argparser.add_argument("--cuda", type=str)

    args = argparser.parse_args()

    model_name = args.model
    src_dir = args.source
    src_split = src_dir.split("/")
    embed_name = f"embeddings/{model_name}_{"_".join(src_split[1:])}.json"

    model = get_model(model_name, args.cuda)

    print(f"Generating embeddings for mails in '{src_dir}' with '{model_name}'")
    
    embeddings = {}
    embed_counter = 0

    start_embedding = time.time()

    for mail in progressBarTime(os.listdir(src_dir), start_embedding, prefix=" ", length=50):
        content = "\n\n".join(read_file(os.path.join(src_dir, mail)).split("\n\n")[1:])
        embedding = embed(model, content)

        embeddings[mail] = embedding
        embed_counter += 1

    model.clean()

    print(f"Created {embed_counter} embeddings in {time.time() - start_embedding} seconds.")
    
    with open(embed_name, "w") as json_embeds:
        json_embeds.write(json.dumps(embeddings, indent=2))

    print(f"Saved embeddings at {embed_name}")


if __name__ == "__main__":
    main()