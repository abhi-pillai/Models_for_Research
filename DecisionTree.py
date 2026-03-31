import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
# 1. CONSTANTS AND CONFIGURATION
# =============================================================================
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/mitdb"
FS = 360
WINDOW_BEFORE = 93
WINDOW_AFTER = 93  # Total window = 187 (93 + 1 + 93 = 187)
RANDOM_SEED = 42

EXCLUDED_RECORDS = ['102', '104']
REVERSED_RECORD = '114'

NORMAL_SYMBOLS = ['N', '·']
ARRHYTHMIA_SYMBOLS = ['L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'E', 'Q', '|', 'x']
VALID_SYMBOLS = set(NORMAL_SYMBOLS + ARRHYTHMIA_SYMBOLS)

np.random.seed(RANDOM_SEED)

# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================
def extract_features(beat_signal):
    eps = 1e-10
    features = {}

    # Time-Domain
    features['Mean'] = np.mean(beat_signal)
    features['Std_Dev'] = np.std(beat_signal)
    features['RMS'] = np.sqrt(np.mean(beat_signal**2))
    features['Peak_to_Peak'] = np.max(beat_signal) - np.min(beat_signal)
    features['Skewness'] = stats.skew(beat_signal)
    features['Kurtosis'] = stats.kurtosis(beat_signal)

    # Morphological
    features['R_Peak_Amp'] = beat_signal[WINDOW_BEFORE]
    features['R_Peak_Idx'] = WINDOW_BEFORE
    features['Signal_Energy'] = np.sum(beat_signal**2)
    features['Zero_Crossing_Rate'] = np.sum(np.diff(np.sign(beat_signal)) != 0)
    threshold = 0.5 * features['R_Peak_Amp']
    features['QRS_Width'] = np.sum(beat_signal > threshold)

    # Frequency-Domain
    N = len(beat_signal)
    yf = fft(beat_signal)
    xf = fftfreq(N, 1/FS)[:N//2]
    power_spectrum = 2.0/N * np.abs(yf[0:N//2])**2

    features['Dominant_Freq'] = xf[np.argmax(power_spectrum)]
    features['Low_Freq_Power'] = np.sum(power_spectrum[xf < 10])
    features['Mid_Freq_Power'] = np.sum(power_spectrum[(xf >= 10) & (xf <= 30)])
    features['High_Freq_Power'] = np.sum(power_spectrum[xf > 30])
    psd_norm = power_spectrum / (np.sum(power_spectrum) + eps)
    features['Spectral_Entropy'] = -np.sum(psd_norm * np.log2(psd_norm + eps))

    return features

# =============================================================================
# 3. DATA LOADING & PREPROCESSING
# =============================================================================
def load_and_preprocess_data(db_path):
    features_list, labels, patient_ids = [], [], []

    records = [f.split('.')[0] for f in os.listdir(db_path) if f.endswith('.dat')]
    records = [r for r in records if r not in EXCLUDED_RECORDS]
    records = list(set(records))

    for record in records:
        record_path = os.path.join(db_path, record)
        try:
            signals, fields = wfdb.rdsamp(record_path)
            annotations = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print(f"Skipping {record} due to read error: {e}")
            continue

        channel_idx = 1 if record == REVERSED_RECORD else 0
        signal = signals[:, channel_idx]

        for sample, symbol in zip(annotations.sample, annotations.symbol):
            if symbol not in VALID_SYMBOLS:
                continue
            label = 0 if symbol in NORMAL_SYMBOLS else 1

            start = sample - WINDOW_BEFORE
            end = sample + WINDOW_AFTER + 1
            beat = np.zeros(WINDOW_BEFORE + WINDOW_AFTER + 1)

            if start < 0:
                beat[abs(start):] = signal[0:end]
            elif end > len(signal):
                beat[:len(signal)-start] = signal[start:len(signal)]
            else:
                beat = signal[start:end]

            beat_std = np.std(beat)
            beat_norm = (beat - np.mean(beat)) / beat_std if beat_std > 0 else beat - np.mean(beat)

            feat_dict = extract_features(beat_norm)
            features_list.append(feat_dict)
            labels.append(label)
            patient_ids.append(record)

    df_features = pd.DataFrame(features_list)
    return df_features, np.array(labels), np.array(patient_ids)

# =============================================================================
# 4. DECISION TREE TUNING
# =============================================================================
def tune_decision_tree(X_train, y_train, X_val, y_val):
    depths = [5, 10, 15, 20, 25]
    splits = [2, 5, 10]

    best_f1 = -1
    best_model = None
    best_params = {}

    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

    tuning_results = []

    for d in depths:
        for s in splits:
            clf = DecisionTreeClassifier(max_depth=d,
                                         min_samples_split=s,
                                         min_samples_leaf=5,
                                         class_weight=weight_dict,
                                         random_state=RANDOM_SEED)
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred)
            tuning_results.append({'max_depth': d, 'min_samples_split': s, 'val_f1': val_f1})

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = {'max_depth': d, 'min_samples_split': s}
                best_model = clf

    return best_model, best_params, pd.DataFrame(tuning_results)

# =============================================================================
# 5. EVALUATION
# =============================================================================
def evaluate_and_plot(model, X_train, y_train, X_val, y_val, X_test, y_test, df_tuning, feature_names):
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("📈 TEST SET METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_test_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=['Normal (0)', 'Arrhythmia (1)']))

    plt.figure(figsize=(18,12))

    # Confusion Matrix
    plt.subplot(2,3,1)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal','Arrhythmia'], yticklabels=['Normal','Arrhythmia'])
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")

    # ROC Curve
    plt.subplot(2,3,2)
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={roc_auc_score(y_test, y_test_prob):.2f})')
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Feature Importance
    plt.subplot(2,3,3)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
    plt.title("Feature Importances")
    plt.xlabel("Gini Importance")

    # Learning Analysis
    plt.subplot(2,3,4)
    sns.lineplot(data=df_tuning, x='max_depth', y='val_f1', hue='min_samples_split', marker='o')
    plt.title("Validation F1 vs Tree Depth")
    plt.xlabel("Maximum Tree Depth")
    plt.ylabel("Validation F1-Score")

    # Tree Visualization (Depth 3)
    plt.subplot(2,3,(5,6))
    plot_tree(model, max_depth=3, feature_names=feature_names,
              class_names=['Normal','Arrhythmia'], filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Structure (Truncated to Depth 3)")

    plt.tight_layout()
    plt.savefig("ecg_evaluation_results.png", dpi=300)
    plt.show()

# =============================================================================
# 6. MAIN (uses pre-split data)
# =============================================================================
def main():
    print("⏳ Loading pre-split dataset...")

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    print("\n" + "="*50)
    print("📊 SPLIT DATASET STATISTICS")
    print("="*50)
    print(f"Training Set:   {len(y_train)} beats")
    print(f"Validation Set: {len(y_val)} beats")
    print(f"Testing Set:    {len(y_test)} beats")

    print("\nTraining Class Distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = "Normal" if u == 0 else "Arrhythmia"
        print(f"{lbl}: {c}")

    # Feature names
    feature_names = [
        'Mean','Std_Dev','RMS','Peak_to_Peak','Skewness','Kurtosis',
        'R_Peak_Amp','R_Peak_Idx','Signal_Energy','Zero_Crossing_Rate',
        'QRS_Width','Dominant_Freq','Low_Freq_Power','Mid_Freq_Power',
        'High_Freq_Power','Spectral_Entropy'
    ]

    print("\n🌳 Training and Tuning Decision Tree Classifier...")
    best_model, best_params, df_tuning = tune_decision_tree(X_train, y_train, X_val, y_val)

    print("\n🏆 Best Hyperparameters Found:")
    print(f"max_depth: {best_params['max_depth']}")
    print(f"min_samples_split: {best_params['min_samples_split']}")
    print("min_samples_leaf: 5 (fixed)")

    print("\n🚀 Evaluating Model...")
    evaluate_and_plot(best_model, X_train, y_train, X_val, y_val, X_test, y_test, df_tuning, feature_names)

if __name__ == "__main__":
    main()