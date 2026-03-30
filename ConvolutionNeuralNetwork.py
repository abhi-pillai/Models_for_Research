import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# --- 1. SETTINGS & PATHS ---
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/mitdb/"
SAMPLING_RATE = 360
WINDOW_SIZE = 187
HALF_WINDOW = 93
SEED = 42

# Records to exclude (Paced beats)
excluded_records = ['102', '104']

# --- 2. DATA UTILITIES ---
def get_record_list(path):
    files = [f for f in os.listdir(path) if f.endswith('.dat')]
    records = [r.replace('.dat', '') for r in files if r.replace('.dat', '') not in excluded_records]
    return sorted(records)

def extract_beats_from_record(rid):
    """
    Extracts beats, applies Z-score normalization, and handles Record 114 logic.
    """
    X_rec, y_rec = [], []
    record_path = os.path.join(DATA_PATH, rid)

    # IEEE Rationale: Use Lead II (Ch 0) generally, Lead V1 (Ch 1) for Record 114
    channel = 1 if rid == '114' else 0

    try:
        record = wfdb.rdrecord(record_path, channels=[channel])
        annotation = wfdb.rdann(record_path, 'atr')
        signal = record.p_signal.flatten()

        # Z-score Normalization (Zero mean, unit variance)
        signal = (signal - np.mean(signal)) / np.std(signal)

        for i, ann_idx in enumerate(annotation.sample):
            symbol = annotation.symbol[i]

            # Label Mapping: Normal (0), Arrhythmia (1)
            if symbol in ['N', '.']:
                label = 0
            elif symbol in ['L', 'R', 'A', 'a', 'J', 'S', 'V', 'E', 'F', 'e', 'j']:
                label = 1
            else:
                continue # Skip non-beat symbols like '+' or '~'

            # Window extraction with boundary handling
            if ann_idx - HALF_WINDOW >= 0 and ann_idx + HALF_WINDOW + 1 <= len(signal):
                beat = signal[ann_idx - HALF_WINDOW : ann_idx + HALF_WINDOW + 1]
                X_rec.append(beat)
                y_rec.append(label)

    except Exception as e:
        print(f"Error processing record {rid}: {e}")

    return np.array(X_rec), np.array(y_rec)

# --- 3. PATIENT-WISE DATA SPLITTING ---
records = get_record_list(DATA_PATH)

# Split patients: 70% Train, 10% Val, 20% Test
train_ids, test_ids = train_test_split(records, test_size=0.20, random_state=SEED)
train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=SEED) # 0.125 * 0.8 = 0.1

def collect_data(id_list):
    X_list, y_list = [], []
    for rid in id_list:
        X_r, y_r = extract_beats_from_record(rid)
        if len(X_r) > 0:
            X_list.append(X_r)
            y_list.append(y_r)
    return np.vstack(X_list), np.hstack(y_list)

print("Loading data by patient groups...")
X_train_raw, y_train = collect_data(train_ids)
X_val_raw, y_val = collect_data(val_ids)
X_test_raw, y_test = collect_data(test_ids)

# Reshape for 1D CNN: (Samples, TimeSteps, Channels)
X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
X_val = X_val_raw.reshape(X_val_raw.shape[0], X_val_raw.shape[1], 1)
X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)

# Calculate Class Weights to handle imbalance (Normal beats vastly outnumber Arrhythmias)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}

# --- 4. 1D-CNN ARCHITECTURE ---
def build_cnn():
    model = models.Sequential([
        # Layer 1: Captures QRS sharp edges
        layers.Conv1D(32, kernel_size=11, padding='same', input_shape=(WINDOW_SIZE, 1)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=3),

        # Layer 2: Captures rhythm/morphology patterns
        layers.Conv1D(64, kernel_size=7, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=3),

        # Layer 3: Higher level feature integration
        layers.Conv1D(128, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling1D(),

        # Fully Connected
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_cnn()
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')])

# --- 5. TRAINING ---
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=128,
                    class_weight=class_weight_dict,
                    verbose=1)

# --- 6. EVALUATION & VISUALIZATION ---
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# 6a. Training Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# 6b. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Arrhythmia'], yticklabels=['Normal', 'Arrhythmia'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 6c. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

print("\nClassification Report:\n", classification_report(y_test, y_pred))