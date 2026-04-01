"""
=============================================================================
IEEE ECG ARRHYTHMIA CLASSIFICATION — SVM (PRE-SPLIT DATA)
=============================================================================
Dataset      : MIT-BIH Arrhythmia Database
Task         : Binary Classification (Normal vs Arrhythmia)
Model        : Support Vector Machine (RBF Kernel)

NOTE:
-----
Uses precomputed features + patient-wise split (.npy files)
Ensures reproducibility and fair comparison with other models
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)

# =============================================================================
# CONFIG
# =============================================================================
RANDOM_SEED = 42

FEATURE_NAMES = [
    'Mean','Std','RMS','Peak2Peak','Skew','Kurtosis',
    'R_amp','Energy','ZCR','QRS_width',
    'Dom_freq','LF','MF','HF','Spec_entropy'
]

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data():
    print("\nLoading pre-split data...")

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")

    X_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    print("Shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# DATASET STATS
# =============================================================================
def print_stats(y_train, y_val, y_test):
    print("\nDATASET DISTRIBUTION")
    print("="*50)

    for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        total = len(y)
        normal = np.sum(y == 0)
        arr = np.sum(y == 1)

        print(f"\n{name}:")
        print(f"Total       : {total}")
        print(f"Normal      : {normal} ({100*normal/total:.2f}%)")
        print(f"Arrhythmia  : {arr} ({100*arr/total:.2f}%)")


# =============================================================================
# TRAIN SVM
# =============================================================================
def train_svm(X_train, y_train, X_val, y_val):
    print("\nScaling features...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print("\nTraining SVM...")

    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=RANDOM_SEED
    )

    model.fit(X_train, y_train)

    # Validation check
    y_val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)

    print("Validation F1:", val_f1)

    return model, scaler


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(model, scaler, X, y, name):
    X = scaler.transform(X)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{name} RESULTS")
    print("="*40)
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("ROC AUC  :", auc)

    print("\nConfusion Matrix:\n", cm)
    print("\n", classification_report(y, y_pred))

    return {
        'acc': acc, 'prec': prec, 'rec': rec,
        'f1': f1, 'auc': auc, 'cm': cm,
        'y_prob': y_prob
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_results(y, prob, cm):
    plt.figure(figsize=(12,5))

    # Confusion Matrix
    plt.subplot(1,2,1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # ROC Curve
    plt.subplot(1,2,2)
    fpr, tpr, _ = roc_curve(y, prob)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*70)
    print("SVM ECG ARRHYTHMIA CLASSIFICATION (PRE-SPLIT)")
    print("="*70)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Stats
    print_stats(y_train, y_val, y_test)

    # Train
    model, scaler = train_svm(X_train, y_train, X_val, y_val)

    # Evaluate
    train_res = evaluate(model, scaler, X_train, y_train, "Train")
    val_res   = evaluate(model, scaler, X_val, y_val, "Validation")
    test_res  = evaluate(model, scaler, X_test, y_test, "Test")

    # Plot
    plot_results(y_test, test_res['y_prob'], test_res['cm'])

    print("\nFINAL RESULTS")
    print("=============")
    print("Test F1 :", test_res['f1'])
    print("Test AUC:", test_res['auc'])

    return model


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()