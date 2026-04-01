# =============================================================================
# TCN ECG ARRHYTHMIA CLASSIFIER (USING PRE-SPLIT DATA) — FIXED VERSION
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
from tensorflow.keras import layers, Model, callbacks

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

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
    WINDOW = 187

    # TCN
    TCN_FILTERS = 64
    TCN_KERNEL_SIZE = 5
    TCN_DILATIONS = [1, 2, 4, 8, 16, 32]
    TCN_DROPOUT = 0.2

    DENSE_UNITS = [128, 64]

    # Training
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-3

    ES_PATIENCE = 15
    LR_PATIENCE = 7
    LR_FACTOR = 0.5
    MIN_LR = 1e-6

CFG = Config()

# =========================
# LOAD DATA
# =========================
def load_data():
    print("📂 Loading preprocessed data...")

    X_train = np.load("X_train_raw.npy")
    X_val   = np.load("X_val_raw.npy")
    X_test  = np.load("X_test_raw.npy")

    y_train = np.load("y_train.npy")
    y_val   = np.load("y_val.npy")
    y_test  = np.load("y_test.npy")

    # reshape → (N, 187, 1)
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    return {
        "train": {"X": X_train, "y": y_train},
        "val":   {"X": X_val,   "y": y_val},
        "test":  {"X": X_test,  "y": y_test},
    }

# =========================
# CLASS WEIGHTS (FIXED)
# =========================
def get_class_weights(y):
    print("🔍 Unique labels:", np.unique(y))

    classes = np.unique(y)  # ✅ robust fix
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    cw = dict(zip(classes, weights))
    print("⚖️ Class weights:", cw)
    return cw

# =========================
# TCN BLOCK
# =========================
def tcn_block(x, filters, kernel, dilation, dropout, block_id):
    res = x

    for i in range(2):
        x = layers.Conv1D(filters, kernel,
                          padding='causal',
                          dilation_rate=dilation,
                          name=f"conv_{block_id}_{i}")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SpatialDropout1D(dropout)(x)

    if res.shape[-1] != filters:
        res = layers.Conv1D(filters, 1)(res)

    return layers.Add()([x, res])

# =========================
# BUILD MODEL
# =========================
def build_model():
    inp = keras.Input(shape=(CFG.WINDOW, 1))
    x = inp

    for i, d in enumerate(CFG.TCN_DILATIONS):
        x = tcn_block(x,
                      CFG.TCN_FILTERS,
                      CFG.TCN_KERNEL_SIZE,
                      d,
                      CFG.TCN_DROPOUT,
                      i)

    x = layers.GlobalAveragePooling1D()(x)

    for units in CFG.DENSE_UNITS:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    return Model(inp, out)

# =========================
# TRAIN
# =========================
def train(model, splits, cw):
    model.compile(
        optimizer=keras.optimizers.Adam(CFG.LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    model.summary()

    cb = [
        callbacks.EarlyStopping(
            patience=CFG.ES_PATIENCE,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            factor=CFG.LR_FACTOR,
            patience=CFG.LR_PATIENCE,
            min_lr=CFG.MIN_LR
        ),
        callbacks.ModelCheckpoint(
            "best_tcn.keras",
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
        callbacks=cb,
        verbose=1
    )

    return history

# =========================
# EVALUATION
# =========================
def evaluate(model, X, y, name):
    y_prob = model.predict(X, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n{name} RESULTS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {roc_auc:.4f}")

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
# MAIN
# =========================
def main():
    splits = load_data()

    # shuffle train
    idx = np.random.permutation(len(splits["train"]["X"]))
    splits["train"]["X"] = splits["train"]["X"][idx]
    splits["train"]["y"] = splits["train"]["y"][idx]

    cw = get_class_weights(splits["train"]["y"])

    model = build_model()

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

    print("\n✅ DONE")

if __name__ == "__main__":
    main()