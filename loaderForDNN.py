import os
import wfdb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq

# =========================
# CONFIG
# =========================
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/mitdb"
FS = 360
WINDOW_BEFORE = 93
WINDOW_AFTER = 93
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

EXCLUDED_RECORDS = ['102', '104']
REVERSED_RECORD = '114'

NORMAL_SYMBOLS = ['N', '·']
ARRHYTHMIA_SYMBOLS = ['L','R','A','a','J','S','V','F','e','j','E','Q','|','x']
VALID_SYMBOLS = set(NORMAL_SYMBOLS + ARRHYTHMIA_SYMBOLS)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(beat):
    eps = 1e-10
    f = {}

    f['Mean'] = np.mean(beat)
    f['Std'] = np.std(beat)
    f['RMS'] = np.sqrt(np.mean(beat**2))
    f['Peak2Peak'] = np.max(beat) - np.min(beat)
    f['Skew'] = stats.skew(beat)
    f['Kurtosis'] = stats.kurtosis(beat)

    f['R_amp'] = beat[WINDOW_BEFORE]
    f['Energy'] = np.sum(beat**2)
    f['ZCR'] = np.sum(np.diff(np.sign(beat)) != 0)

    thr = 0.5 * f['R_amp']
    f['QRS_width'] = np.sum(beat > thr)

    N = len(beat)
    yf = fft(beat)
    xf = fftfreq(N, 1/FS)[:N//2]
    ps = 2.0/N * np.abs(yf[:N//2])**2

    f['Dom_freq'] = xf[np.argmax(ps)]
    f['LF'] = np.sum(ps[xf < 10])
    f['MF'] = np.sum(ps[(xf >= 10) & (xf <= 30)])
    f['HF'] = np.sum(ps[xf > 30])

    ps_norm = ps / (np.sum(ps) + eps)
    f['Spec_entropy'] = -np.sum(ps_norm * np.log2(ps_norm + eps))

    return f

# =========================
# LOAD DATA (MODIFIED)
# =========================
def load_data(path):
    feats, raw_beats, labels, p_ids = [], [], [], []

    records = [f.split('.')[0] for f in os.listdir(path) if f.endswith('.dat')]
    records = [r for r in records if r not in EXCLUDED_RECORDS]

    for rec in records:
        sig, _ = wfdb.rdsamp(os.path.join(path, rec))
        ann = wfdb.rdann(os.path.join(path, rec), 'atr')

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

            # normalize
            if np.std(beat) > 0:
                beat = (beat - np.mean(beat)) / np.std(beat)

            # ✅ STORE BOTH
            raw_beats.append(beat)                  # For CNN
            feats.append(extract_features(beat))    # For RF/SVM
            labels.append(y)
            p_ids.append(rec)

    return np.array(raw_beats), pd.DataFrame(feats), np.array(labels), np.array(p_ids)

# =========================
# PATIENT-WISE SPLIT (MODIFIED)
# =========================
def split_data(X_raw, X_feat, y, p_ids):
    patients = np.unique(p_ids)
    np.random.shuffle(patients)

    n = len(patients)
    tr = int(0.7*n)
    va = int(0.8*n)

    train_p = patients[:tr]
    val_p = patients[tr:va]
    test_p = patients[va:]

    train_m = np.isin(p_ids, train_p)
    val_m = np.isin(p_ids, val_p)
    test_m = np.isin(p_ids, test_p)

    return (
        X_raw[train_m], X_feat[train_m], y[train_m],
        X_raw[val_m],   X_feat[val_m],   y[val_m],
        X_raw[test_m],  X_feat[test_m],  y[test_m],
        train_p, val_p, test_p
    )

# =========================
# MAIN
# =========================
print("Loading & processing...")

X_raw, X_feat_df, y, p_ids = load_data(DATA_PATH)

print("Splitting...")

(Xr_tr, Xf_tr, y_tr,
 Xr_va, Xf_va, y_va,
 Xr_te, Xf_te, y_te,
 tr_p, va_p, te_p) = split_data(X_raw, X_feat_df.values, y, p_ids)

print("Saving...")

# -------- FEATURES (RF, SVM) --------
np.save("X_train.npy", Xf_tr)
np.save("X_val.npy", Xf_va)
np.save("X_test.npy", Xf_te)

# -------- RAW (CNN) --------
np.save("X_train_raw.npy", Xr_tr)
np.save("X_val_raw.npy", Xr_va)
np.save("X_test_raw.npy", Xr_te)

# -------- LABELS --------
np.save("y_train.npy", y_tr)
np.save("y_val.npy", y_va)
np.save("y_test.npy", y_te)

# -------- PATIENT GROUPS --------
np.save("train_patients.npy", tr_p)
np.save("val_patients.npy", va_p)
np.save("test_patients.npy", te_p)

print("✅ Done! SAME samples saved for RF, SVM, and CNN.")