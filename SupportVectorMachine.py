import numpy as np
import pandas as pd
import wfdb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- 1. CONFIGURATION & DATA LOADING ---
PATH = "/content/drive/MyDrive/Colab Notebooks/mitdb/"
SAMPLING_RATE = 360
WINDOW_SIZE = 93 # 93 before + 1 (R) + 93 after = 187 samples
RANDOM_SEED = 42

# Exclude paced records
EXCLUDED_RECORDS = ['102', '104']
RECORDS = [f for f in os.listdir(PATH) if f.endswith('.dat')]
RECORDS = [r.replace('.dat', '') for r in RECORDS if r.replace('.dat', '') not in EXCLUDED_RECORDS]

def extract_features(signal):
    """Extracts Time, Morphological, and Frequency features."""
    # Time Domain
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)

    # Morphological
    r_amp = np.max(signal)
    qrs_energy = np.sum(signal**2)
    zcr = ((signal[:-1] * signal[1:]) < 0).sum()

    # Frequency Domain (FFT)
    freq_data = np.abs(fft(signal))[:len(signal)//2]
    dom_freq = np.argmax(freq_data)
    spec_entropy = -np.sum(freq_data * np.log2(freq_data + 1e-12))

    return [mean_val, std_val, rms, skew_val, kurt_val, r_amp, qrs_energy, zcr, dom_freq, spec_entropy]

# --- 2. DATA PROCESSING ---
def load_and_process():
    print("Loading and processing data...")
    all_features = []
    all_labels = []
    patient_ids = []

    for res in RECORDS:
        # Special case: Record 114 uses second channel (index 1)
        channel = 1 if res == '114' else 0
        record = wfdb.rdrecord(os.path.join(PATH, res), channels=[channel])
        ann = wfdb.rdann(os.path.join(PATH, res), 'atr')

        signal = record.p_signal.flatten()
        # Z-score normalization of full record
        signal = (signal - np.mean(signal)) / np.std(signal)

        for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            if sample < WINDOW_SIZE or sample > len(signal) - WINDOW_SIZE:
                continue

            # Extract Beat
            beat = signal[sample - WINDOW_SIZE : sample + WINDOW_SIZE + 1]

            # Label Mapping
            label = 0 if symbol in ['N', '.'] else 1

            # Feature Extraction
            features = extract_features(beat)

            all_features.append(features)
            all_labels.append(label)
            patient_ids.append(res)

    return np.array(all_features), np.array(all_labels), np.array(patient_ids)

X, y, groups = load_and_process()

# --- 3. PATIENT-WISE SPLIT ---
unique_patients = np.unique(groups)
train_pts, test_pts = train_test_split(unique_patients, test_size=0.20, random_state=RANDOM_SEED)
train_pts, val_pts = train_test_split(train_pts, test_size=0.125, random_state=RANDOM_SEED) # 0.125 * 0.8 = 0.1

train_mask = np.isin(groups, train_pts)
val_mask = np.isin(groups, val_pts)
test_mask = np.isin(groups, test_pts)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# --- 4. MODEL TRAINING (SVM) ---
# Note: In a full research workflow, use GridSearchCV here.
# We use RBF kernel as it handles non-linear morphological overlaps best.
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True)
svm_model.fit(X_train, y_train)

# --- 5. EVALUATION ---
y_pred = svm_model.predict(X_test)
y_prob = svm_model.predict_proba(X_test)[:, 1]

print("Test Classification Report:\n", classification_report(y_test, y_pred))

# Visualizations
plt.figure(figsize=(12, 5))

# Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend()
plt.show()