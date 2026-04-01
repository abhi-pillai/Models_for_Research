"""
=============================================================================
IEEE ECG ARRHYTHMIA CLASSIFICATION USING RANDOM FOREST (PRE-SPLIT DATA)
=============================================================================
Dataset      : MIT-BIH Arrhythmia Database (PhysioNet)
Task         : Binary Classification — Normal vs Arrhythmia
Classifier   : RandomForestClassifier (scikit-learn)

NOTE:
-----
This version uses PRECOMPUTED FEATURES and PATIENT-WISE SPLIT stored as .npy files.
This ensures:
✔ No data leakage
✔ Reproducibility
✔ Fair comparison across models
=============================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 2: CONFIG
# =============================================================================
RANDOM_SEED = 42

# Feature names (MUST match your extraction order)
FEATURE_NAMES = [
    'Mean','Std','RMS','Peak2Peak','Skew','Kurtosis',
    'R_amp','Energy','ZCR','QRS_width',
    'Dom_freq','LF','MF','HF','Spec_entropy'
]

# =============================================================================
# SECTION 3: LOAD DATA
# =============================================================================
def load_split_data():
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
# SECTION 4: DATASET STATS
# =============================================================================
def print_dataset_statistics(y_train, y_val, y_test):
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        n_total = len(y)
        n_normal = np.sum(y == 0)
        n_arr = np.sum(y == 1)

        print(f"\n{name}:")
        print(f"Total        : {n_total}")
        print(f"Normal       : {n_normal} ({100*n_normal/n_total:.2f}%)")
        print(f"Arrhythmia   : {n_arr} ({100*n_arr/n_total:.2f}%)")
        print(f"Imbalance    : {n_arr/max(n_normal,1):.4f}")


# =============================================================================
# SECTION 5: TRAINING + TUNING
# =============================================================================
def tune_and_train(X_train, y_train, X_val, y_val):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3]
    }

    best_f1 = -1
    best_model = None
    best_params = None
    results = []

    print("\nTuning model...")

    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for split in param_grid['min_samples_split']:
                for leaf in param_grid['min_samples_leaf']:

                    model = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        min_samples_split=split,
                        min_samples_leaf=leaf,
                        class_weight='balanced',
                        random_state=RANDOM_SEED,
                        n_jobs=-1
                    )

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    f1 = f1_score(y_val, y_pred)

                    results.append([n_est, depth, split, leaf, f1])

                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model
                        best_params = (n_est, depth, split, leaf)

    print("\nBest Params:", best_params)
    print("Best Val F1:", best_f1)

    results_df = pd.DataFrame(results, columns=[
        'n_estimators','max_depth','min_split','min_leaf','f1'
    ])

    return best_model, best_params, results_df


# =============================================================================
# SECTION 6: EVALUATION
# =============================================================================
def evaluate(model, X, y, name):
    y_prob = model.predict_proba(X)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{name} Results")
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
# SECTION 7: VISUALIZATION
# =============================================================================
def plot_confusion(cm):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc(y, prob):
    fpr, tpr, _ = roc_curve(y, prob)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.show()


def plot_feature_importance(model):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]

    plt.barh(range(len(idx)), imp[idx])
    plt.yticks(range(len(idx)), np.array(FEATURE_NAMES)[idx])
    plt.title("Feature Importance")
    plt.show()


# =============================================================================
# SECTION 8: MAIN
# =============================================================================
def main():
    print("\n" + "="*70)
    print("ECG ARRHYTHMIA CLASSIFICATION (PRE-SPLIT)")
    print("="*70)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()

    # Stats
    print_dataset_statistics(y_train, y_val, y_test)

    # Train
    model, params, results_df = tune_and_train(X_train, y_train, X_val, y_val)

    # Evaluate
    train_res = evaluate(model, X_train, y_train, "Train")
    val_res   = evaluate(model, X_val, y_val, "Validation")
    test_res  = evaluate(model, X_test, y_test, "Test")

    # Plots
    plot_confusion(test_res['cm'])
    plot_roc(y_test, test_res['y_prob'])
    plot_feature_importance(model)

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