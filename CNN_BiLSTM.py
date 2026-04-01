# =============================================================================
# CNN-BiLSTM USING FIXED PRE-SAVED SPLIT (IEEE-CORRECT)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)

# =============================================================================
# 1. LOAD PRE-SAVED DATA (IMPORTANT CHANGE)
# =============================================================================
print("Loading pre-saved dataset...")

X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")
X_test  = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_val   = np.load("y_val.npy")
y_test  = np.load("y_test.npy")

# =============================================================================
# 2. RESHAPE FOR CNN INPUT
# =============================================================================
# Your saved data is features → not raw signal
# So reshape as (samples, features, 1)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val   = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("Shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# =============================================================================
# 3. CLASS WEIGHTS (SAME DATA → FAIR COMPARISON)
# =============================================================================
classes = np.array([0, 1])
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {0: cw[0], 1: cw[1]}

print("Class weights:", class_weights)

# =============================================================================
# 4. CNN-BiLSTM MODEL (UNCHANGED ARCHITECTURE)
# =============================================================================
def build_cnn_bilstm(input_shape):
    inp = keras.Input(shape=input_shape)

    # CNN
    x = layers.Conv1D(64, 5, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)

    # Dense
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inp, out)

model = build_cnn_bilstm((X_train.shape[1], 1))
model.summary()

# =============================================================================
# 5. TRAINING
# =============================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# =============================================================================
# 6. EVALUATION
# =============================================================================
def evaluate(X, y, name):
    y_prob = model.predict(X).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n=== {name} ===")
    print(classification_report(y, y_pred))

    auc = roc_auc_score(y, y_prob)
    print("AUC:", auc)

    return y_prob, y_pred

prob_train, pred_train = evaluate(X_train, y_train, "TRAIN")
prob_val,   pred_val   = evaluate(X_val,   y_val,   "VAL")
prob_test,  pred_test  = evaluate(X_test,  y_test,  "TEST")

# =============================================================================
# 7. CONFUSION MATRIX
# =============================================================================
cm = confusion_matrix(y_test, pred_test)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =============================================================================
# 8. ROC CURVE
# =============================================================================
fpr, tpr, _ = roc_curve(y_test, prob_test)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, prob_test):.3f}")
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curve")
plt.show()