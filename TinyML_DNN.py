# =============================================================================
# TinyML DNN ECG ARRHYTHMIA CLASSIFIER (USING PRE-SPLIT FEATURE DATA)
# =============================================================================

# =========================
# IMPORTS
# =========================
import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")

# =========================
# REPRODUCIBILITY
# =========================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("✅ TensorFlow:", tf.__version__)

# =========================
# CONFIG
# =========================
class Config:
    INPUT_DIM = None   # will set after loading

    HIDDEN_UNITS = [128, 64, 32]
    DROPOUT = 0.3
    L2 = 1e-4

    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-3

    PATIENCE = 10

CFG = Config()

# =========================
# LOAD DATA (FROM YOUR PIPELINE)
# =========================
def load_data():
    print("📂 Loading preprocessed feature data...")

    X_train = np.load("X_train.npy")
    X_val   = np.load("X_val.npy")
    X_test  = np.load("X_test.npy")

    y_train = np.load("y_train.npy")
    y_val   = np.load("y_val.npy")
    y_test  = np.load("y_test.npy")

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    return {
        "train": {"X": X_train, "y": y_train},
        "val":   {"X": X_val,   "y": y_val},
        "test":  {"X": X_test,  "y": y_test},
    }

# =========================
# CLASS WEIGHTS (FIXED)
# =========================
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced",
                                   classes=classes,
                                   y=y)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print("⚖️ Class weights:", cw)
    return cw

# =========================
# BUILD MODEL
# =========================
def build_model(input_dim):
    inp = keras.Input(shape=(input_dim,))

    x = inp
    for i, units in enumerate(CFG.HIDDEN_UNITS):
        x = layers.Dense(units,
                         kernel_regularizer=regularizers.l2(CFG.L2),
                         name=f"fc_{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(CFG.DROPOUT)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)

    model.compile(
        optimizer=keras.optimizers.Adam(CFG.LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model

# =========================
# TRAIN
# =========================
def train(model, splits, cw):
    cb = [
        callbacks.EarlyStopping(
            patience=CFG.PATIENCE,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            "best_tinyml.keras",
            monitor="val_auc",
            save_best_only=True
        )
    ]

    history = model.fit(
        splits["train"]["X"], splits["train"]["y"],
        validation_data=(splits["val"]["X"], splits["val"]["y"]),
        epochs=CFG.EPOCHS,
        batch_size=CFG.BATCH_SIZE,
        class_weight=cw,
        callbacks=cb
    )

    return history

# =========================
# EVALUATION
# =========================
def evaluate(model, X, y, name):
    y_prob = model.predict(X).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n{name} RESULTS")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUC:", roc_auc)

    print(classification_report(y, y_pred))

    return fpr, tpr, roc_auc, y_pred

# =========================
# PLOTS
# =========================
def plot_confusion(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc(fpr, tpr, auc_score):
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

# =========================
# TFLITE (TinyML)
# =========================
def convert_to_tflite(model, X_sample):
    def rep_data():
        for i in range(300):
            yield [X_sample[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data

    tflite_model = converter.convert()

    with open("tinyml_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ TFLite model saved")

# =========================
# MAIN
# =========================
def main():
    splits = load_data()

    # shuffle training
    idx = np.random.permutation(len(splits["train"]["X"]))
    splits["train"]["X"] = splits["train"]["X"][idx]
    splits["train"]["y"] = splits["train"]["y"][idx]

    CFG.INPUT_DIM = splits["train"]["X"].shape[1]

    cw = get_class_weights(splits["train"]["y"])

    model = build_model(CFG.INPUT_DIM)
    model.summary()

    history = train(model, splits, cw)

    # Evaluate
    evaluate(model, splits["train"]["X"], splits["train"]["y"], "TRAIN")
    evaluate(model, splits["val"]["X"], splits["val"]["y"], "VAL")

    fpr, tpr, auc_score, y_pred = evaluate(
        model,
        splits["test"]["X"],
        splits["test"]["y"],
        "TEST"
    )

    plot_confusion(splits["test"]["y"], y_pred)
    plot_roc(fpr, tpr, auc_score)

    convert_to_tflite(model, splits["train"]["X"])

    print("\n✅ DONE")

if __name__ == "__main__":
    main()