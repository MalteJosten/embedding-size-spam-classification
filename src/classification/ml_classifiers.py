import tensorflow as tf
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
import keras
import numpy as np


def set_tf_seed(seed):
    tf.random.set_seed(seed)


def train_model(model_type, hp_config, seed, input_size, train_x, train_y, val_x, val_y):

    if model_type == "svm":
        return train_model_svm(hp_config, seed, input_size, train_x, train_y, val_x, val_y)
    elif model_type == "logistic_regression":
        return train_model_logistic_regression(hp_config, seed, input_size, train_x, train_y, val_x, val_y)
    elif model_type == "random_forest":
        return train_model_random_forest(hp_config, seed, input_size, train_x, train_y, val_x, val_y)
    elif model_type == "naive_bayes":
        return train_model_naive_bayes(hp_config, seed, input_size, train_x, train_y, val_x, val_y)
    elif model_type == "mlp":
        return train_model_mlp(hp_config, seed, input_size, train_x, train_y, val_x, val_y)
    else:
        raise ValueError(f"Unknown model type {model_type}")


def train_model_svm(hp_config, seed, input_size, train_x, train_y, val_x, val_y):

    model = SVC(
        kernel=hp_config["kernel"],
        C=1.0,
        probability=False,
        random_state=seed
    )

    model.fit(train_x, train_y)

    def predict(x):
        return model.predict(x)

    return predict


def train_model_logistic_regression(hp_config, seed, input_size, train_x, train_y, val_x, val_y):

    model = LogisticRegression(
        max_iter=hp_config["max_iter"],
        random_state=seed
    )

    model.fit(train_x, train_y)

    def predict(x):
        return model.predict(x)

    return predict


def train_model_random_forest(hp_config, seed, input_size, train_x, train_y, val_x, val_y):

    model = RandomForestClassifier(
        n_estimators=hp_config["n_estimators"],
        max_depth=None,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(train_x, train_y)

    def predict(x):
        return model.predict(x)

    return predict


def train_model_naive_bayes(hp_config, seed, input_size, train_x, train_y, val_x, val_y):

    if hp_config["method"] == "gaussian":
        model = GaussianNB()
    elif hp_config["method"] == "multinomial":
        model = MultinomialNB()
    elif hp_config["method"] == "bernoulli":
        model = BernoulliNB()
    elif hp_config["method"] == "categorical":
        model = CategoricalNB()
    else:
        raise ValueError("hp_config[\"method\"] not in supported")

    model.fit(train_x, train_y)

    def predict(x):
        return model.predict(x)

    return predict


def train_model_mlp(hp_config, seed, input_size, train_x, train_y, val_x, val_y):

    train_y_reshaped = train_y.reshape(-1, 1)
    val_y_reshaped   = val_y.reshape(-1, 1)

    if hp_config["num_layers"] == 3:
        layers = [256, 128]
    elif hp_config["num_layers"] == 2:
        layers = [128]
    else:
        raise ValueError("hp_config[\"layers\"] not in [2, 3]")

    model = keras.models.Sequential([
        keras.layers.Input(shape=(input_size,)) ] + [
            x for units in layers for x in (
            keras.layers.Dense(units, activation='relu'),
            keras.layers.Dropout(hp_config["dropout"]),
        )] + [
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid') # Binary output
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x=train_x, y=train_y_reshaped,
        validation_data=(val_x, val_y_reshaped),
        shuffle=True,
        epochs=20,
        batch_size=128,
        verbose=0
    )

    def predict(x):
        pred_y = model.predict(x, verbose=0)
        return (pred_y > 0.5).astype(np.float32).ravel()

    return predict
