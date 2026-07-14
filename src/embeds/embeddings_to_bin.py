import os
import json
import numpy as np

from util.seeds import set_global_seed


def load_json_data(model):

    def load_json(split):
        file_name = os.path.join("embeddings", f"{model}_{split}.json")

        with open(file_name, "r", encoding="utf-8") as embed_file:
            embeddings = json.loads(embed_file.read())

        return np.array(list(embeddings.values()), dtype=np.float32)

    ham_train, spam_train, ham_test, spam_test = (
        load_json("ham_train"),
        load_json("spam_train"),
        load_json("ham_test"),
        load_json("spam_test"),
    )

    # ham has label 0, spam has label 1
    def ham(arr):
        return np.zeros((arr.shape[0],), dtype=np.float32)

    def spam(arr):
        return np.ones((arr.shape[0],), dtype=np.float32)

    train_x = np.concatenate([ham_train, spam_train])
    train_y = np.concatenate([ham(ham_train), spam(spam_train)])
    test_x = np.concatenate([ham_test, spam_test])
    test_y = np.concatenate([ham(ham_test), spam(spam_test)])

    def shuffle(x, y):
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        return x[indices], y[indices]

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y


def main():

    set_global_seed(42)  # load_json_data shuffles, so seed to always get the same dataset

    os.makedirs("embeddings_bin", exist_ok=True)

    models = sorted(set(e.split('_')[0] for e in os.listdir("embeddings") if e.endswith("json")))

    for model in models:

        print(model)

        train_x, train_y, test_x, test_y = load_json_data(model)

        np.savez_compressed(os.path.join("embeddings_bin", model),
                            train_x=train_x,
                            train_y=train_y,
                            test_x=test_x,
                            test_y=test_y)


if __name__ == "__main__":
    main()
