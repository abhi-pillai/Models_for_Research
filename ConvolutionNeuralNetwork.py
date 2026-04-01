import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# =========================
# CONFIG
# =========================
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/mitdb"
WINDOW_BEFORE = 93
WINDOW_AFTER = 93
WINDOW_SIZE = 187
FS = 360

EXCLUDED_RECORDS = ['102', '104']
REVERSED_RECORD = '114'

NORMAL_SYMBOLS = ['N', '·']
ARRHYTHMIA_SYMBOLS = ['L','R','A','a','J','S','V','F','e','j','E','Q','|','x']
VALID_SYMBOLS = set(NORMAL_SYMBOLS + ARRHYTHMIA_SYMBOLS)

# =========================
# LOAD SAME PATIENT SPLIT
# =========================
train_patients = np.load("train_patients.npy")
val_patients   = np.load("val_patients.npy")
test_patients  = np.load("test_patients.npy")

print("Using SAME patient split as RF/SVM")

# =========================
# LOAD RAW BEATS (IMPORTANT)
# =========================
def load_beats_by_patients(patient_list):
    beats, labels = [], []

    for rec in patient_list:
        sig, _ = wfdb.rdsamp(os.path.join(DATA_PATH, rec))
        ann = wfdb.rdann(os.path.join(DATA_PATH, rec), 'atr')

        ch = 1 if rec == REVERSED_RECORD else 0
        signal = sig[:, ch]

        for s, sym in zip(ann.sample, ann.symbol):
            if sym not in VALID_SYMBOLS:
                continue

            y = 0 if sym in NORMAL_SYMBOLS else 1

            start = s - WINDOW_BEFORE
            end = s + WINDOW_AFTER + 1

            beat = np.zeros(187)

            if start < 0:
                beat[-start:] = signal[0:end]
            elif end > len(signal):
                beat[:len(signal)-start] = signal[start:]
            else:
                beat = signal[start:end]

            # SAME normalization as before
            if np.std(beat) > 0:
                beat = (beat - np.mean(beat)) / np.std(beat)

            beats.append(beat)
            labels.append(y)

    return np.array(beats), np.array(labels)

# =========================
# LOAD DATA (SAME SAMPLES)
# =========================
print("Loading SAME samples for CNN...")

X_train_raw, y_train = load_beats_by_patients(train_patients)
X_val_raw, y_val     = load_beats_by_patients(val_patients)
X_test_raw, y_test   = load_beats_by_patients(test_patients)

print("Shapes:")
print("Train:", X_train_raw.shape)
print("Val:", X_val_raw.shape)
print("Test:", X_test_raw.shape)

# =========================
# RESHAPE FOR CNN
# =========================
X_train = X_train_raw.reshape(-1, WINDOW_SIZE, 1)
X_val   = X_val_raw.reshape(-1, WINDOW_SIZE, 1)
X_test  = X_test_raw.reshape(-1, WINDOW_SIZE, 1)

# =========================
# CLASS WEIGHTS
# =========================
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: weights[0], 1: weights[1]}

# =========================
# CNN MODEL
# =========================
def build_cnn():
    model = models.Sequential([
        layers.Conv1D(32, 11, padding='same', input_shape=(WINDOW_SIZE, 1)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(3),

        layers.Conv1D(64, 7, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(3),

        layers.Conv1D(128, 5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_cnn()

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.Precision(name='precision')]
)

model.summary()

# =========================
# TRAINING
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    class_weight=class_weight_dict,
    verbose=1
)

# =========================
# EVALUATION
# =========================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# PLOTS
# =========================

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss Curve")
plt.legend()
plt.show()

# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy Curve")
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Arrhythmia'],
            yticklabels=['Normal','Arrhythmia'])
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curve")
plt.show()