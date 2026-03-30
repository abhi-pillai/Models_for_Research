"""
=============================================================================
IEEE RESEARCH-GRADE ECG ARRHYTHMIA CLASSIFICATION USING RANDOM FOREST
=============================================================================
Dataset      : MIT-BIH Arrhythmia Database (PhysioNet)
Task         : Binary Classification — Normal vs Arrhythmia
Classifier   : RandomForestClassifier (scikit-learn)
Author       : [Your Name]
Institution  : [Your Institution]
=============================================================================

ABSTRACT
--------
This script implements a patient-wise split Random Forest classifier on the
MIT-BIH Arrhythmia Database for binary ECG beat classification (Normal vs
Arrhythmia). Comprehensive time-domain, morphological, and frequency-domain
features are extracted from each beat window. Class imbalance is addressed
exclusively through class_weight='balanced'. No synthetic data augmentation
is performed. Evaluation includes accuracy, precision, recall, F1-score,
ROC-AUC, confusion matrix, and feature importance analysis — all consistent
with IEEE publication standards.

DATASET REFERENCE
-----------------
Moody, G.B., Mark, R.G. (2001). The impact of the MIT-BIH Arrhythmia
Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45–50.
=============================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
import wfdb

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 2: GLOBAL CONFIGURATION
# =============================================================================

# Path to MIT-BIH dataset on Google Drive
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/mitdb"

# Sampling frequency of MIT-BIH (360 Hz — fixed hardware standard)
FS = 360

# Beat window: 187 samples centered on R-peak (93 before + 1 + 93 after)
# Rationale: At 360 Hz, 187 samples ≈ 519 ms, sufficient to capture
# P-wave onset through T-wave end for most heart rates.
WINDOW_BEFORE = 93
WINDOW_AFTER  = 93
WINDOW_SIZE   = WINDOW_BEFORE + 1 + WINDOW_AFTER  # = 187

# Records to exclude (known annotation/signal quality issues)
EXCLUDED_RECORDS = ['102', '104']

# Record requiring second channel (signal reversal in MLII for record 114)
RECORD_114_CHANNEL = 1   # index 1 = second channel
DEFAULT_CHANNEL    = 0   # MLII is always channel index 0

# Label mapping
# Normal: AAMI N class symbols
# Arrhythmia: everything else with a valid beat annotation
NORMAL_SYMBOLS    = {'N', '·', 'L', 'R', 'e', 'j'}
# NOTE: Strictly, only 'N' and '·' are Normal per AAMI EC57 standard.
# Symbols like L,R are left/right bundle branch beats — often Normal-class
# in prior literature. Adjust NORMAL_SYMBOLS to {'N','·'} for strict AAMI.
NORMAL_SYMBOLS    = {'N', '.'}   # Strict AAMI EC57

# Random seed for reproducibility
RANDOM_SEED = 42

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# =============================================================================
# SECTION 3: DATASET LOADING & RECORD DISCOVERY
# =============================================================================

def get_record_ids(data_path, excluded):
    """
    Discover all valid record IDs in the MIT-BIH dataset directory.

    MIT-BIH Arrhythmia Database contains 48 half-hour recordings from
    47 subjects (record 201 and 202 are from the same subject).
    Records are numbered: 100–124, 200–234 (not all numbers used).

    Parameters
    ----------
    data_path : str
        Path to directory containing .dat, .hea, .atr files.
    excluded : list of str
        Record IDs to exclude.

    Returns
    -------
    list of str
        Sorted list of valid record IDs.
    """
    # Collect all .hea (header) files — each corresponds to one record
    all_files = os.listdir(data_path)
    record_ids = sorted([
        f.replace('.hea', '')
        for f in all_files
        if f.endswith('.hea') and f.replace('.hea', '') not in excluded
    ])
    print(f"[INFO] Total records found    : {len(record_ids) + len(excluded)}")
    print(f"[INFO] Excluded records        : {excluded}")
    print(f"[INFO] Records used            : {len(record_ids)}")
    return record_ids


# =============================================================================
# SECTION 4: PATIENT-WISE DATA SPLITTING
# =============================================================================

def patient_wise_split(record_ids, train_ratio, val_ratio, seed):
    """
    Perform PATIENT-WISE (record-wise) train/val/test split.

    IEEE JUSTIFICATION
    ------------------
    Beat-wise splitting would allow beats from the same patient to appear in
    both training and test sets. Since consecutive beats from one patient share
    morphological and rhythmic characteristics (patient-specific ECG signature),
    this leads to DATA LEAKAGE — the model learns patient identity, not
    arrhythmia patterns, yielding artificially inflated test performance that
    does NOT generalize to unseen patients (a critical clinical requirement).

    Patient-wise splitting ensures that ALL beats from a given patient appear
    in exactly ONE partition, faithfully simulating real-world deployment where
    the model encounters entirely new patients.

    RANDOM SHUFFLING RATIONALE
    --------------------------
    MIT-BIH records are numbered roughly in order of acquisition. Without
    shuffling, a sequential split would assign records 100–117 to training
    and 118–124 + 200–234 to test, introducing DISTRIBUTION BIAS because
    records 200–234 are a more complex arrhythmia subset. Random shuffling
    ensures each split gets a representative mix of normal and pathological
    cases, preventing systematic bias in evaluation.

    Parameters
    ----------
    record_ids : list of str
    train_ratio : float (e.g., 0.70)
    val_ratio   : float (e.g., 0.10)
    seed        : int

    Returns
    -------
    train_ids, val_ids, test_ids : lists of str
    """
    rng = np.random.default_rng(seed)
    ids = np.array(record_ids)
    rng.shuffle(ids)                     # ← CRITICAL: random shuffle

    n = len(ids)
    n_train = int(np.floor(train_ratio * n))
    n_val   = int(np.floor(val_ratio   * n))
    # test gets the remainder to ensure no record is lost

    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val :].tolist()

    print(f"\n[SPLIT] Total patients         : {n}")
    print(f"[SPLIT] Training patients      : {len(train_ids)}  → {train_ids}")
    print(f"[SPLIT] Validation patients    : {len(val_ids)}  → {val_ids}")
    print(f"[SPLIT] Test patients          : {len(test_ids)}  → {test_ids}")

    return train_ids, val_ids, test_ids


# =============================================================================
# SECTION 5: BEAT EXTRACTION & LABEL ASSIGNMENT
# =============================================================================

def load_and_extract_beats(record_ids, data_path, normal_symbols,
                            window_before, window_after, fs):
    """
    Load ECG records, extract individual beats, assign binary labels.

    BEAT EXTRACTION METHODOLOGY
    ---------------------------
    R-peak locations are taken directly from expert-annotated beat annotations
    (*.atr files) provided by the MIT-BIH dataset. Each beat is extracted as
    a fixed-length window:
        [R_peak - window_before  :  R_peak + window_after + 1]
    This symmetric window (93+1+93 = 187 samples) captures:
        • P-wave, PR interval (≈50–200 ms before QRS)
        • QRS complex itself (≈60–120 ms)
        • ST segment and T-wave onset (≈100–250 ms after QRS)
    At 360 Hz, 187 samples ≈ 519 ms, adequate for most physiological RR intervals.

    BOUNDARY HANDLING
    -----------------
    Beats at the very start or end of a recording may have insufficient samples.
    Such beats are ZERO-PADDED symmetrically to maintain consistent window size.
    This avoids discarding edge beats (which may be clinically significant)
    while preserving array dimensionality for feature extraction.

    NOISE CONSIDERATIONS
    --------------------
    MIT-BIH signals contain baseline wander, motion artifact, and electrode
    noise. This implementation does NOT apply additional filtering because:
    (a) wfdb loads ADC-corrected physical signals (mV) after gain normalization;
    (b) the annotation-guided windowing already isolates heartbeat morphology;
    (c) per-beat z-score normalization further suppresses DC bias and amplitude
        drift. Advanced bandpass filtering (e.g., 0.5–40 Hz) can be added as
        a preprocessing step without changing the pipeline.

    LABEL ASSIGNMENT
    ----------------
    AAMI EC57 Standard (Association for the Advancement of Medical Instrumentation)
    defines two annotation classes:
        Normal (N class) : 'N', '.' — sinus rhythm beats
        Arrhythmia (S/V/F/Q) : premature contractions, flutter, paced beats, etc.
    All non-normal beat annotations are mapped to label=1 (Arrhythmia).
    Non-beat annotations (rhythm change markers, signal quality markers) are
    automatically excluded because wfdb's rdann returns only beat symbols by
    default, and we filter on known beat symbol lists.

    LEAD SELECTION (MLII)
    ---------------------
    Modified Lead II (MLII) is the standard lead used in the MIT-BIH database
    for beat morphology analysis. MLII produces prominent upright QRS complexes
    with clear P-waves, maximizing signal-to-noise ratio for feature extraction.
    The orthogonal lower lead often has inverted or attenuated morphology and is
    used only for Record 114, where the MLII signal quality is compromised.

    Parameters
    ----------
    record_ids : list of str
    data_path  : str
    normal_symbols : set of str
    window_before, window_after : int
    fs : int (sampling frequency)

    Returns
    -------
    X : np.ndarray, shape (n_beats, window_size)  — raw beat windows
    y : np.ndarray, shape (n_beats,)               — binary labels
    """
    # Valid beat annotation symbols in MIT-BIH
    # (excludes rhythm annotations, quality markers, etc.)
    VALID_BEAT_SYMBOLS = {
        'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
        'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?', '.'
    }

    all_beats  = []
    all_labels = []

    for rec_id in record_ids:
        rec_path = os.path.join(data_path, rec_id)

        try:
            # Load signal
            record = wfdb.rdrecord(rec_path)

            # Channel selection (IEEE-justified above)
            if rec_id == '114':
                channel = RECORD_114_CHANNEL
                # Record 114: channel 0 has low-amplitude / reversed QRS;
                # channel 1 (V5 surrogate) provides better morphology
            else:
                channel = DEFAULT_CHANNEL  # MLII

            ecg_signal = record.p_signal[:, channel]  # physical signal in mV

            # Load beat annotations
            annotation = wfdb.rdann(rec_path, 'atr')
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol

            for i, (sample, symbol) in enumerate(zip(ann_samples, ann_symbols)):
                # Keep only valid beat annotations
                if symbol not in VALID_BEAT_SYMBOLS:
                    continue

                # Define window boundaries
                start = sample - window_before
                end   = sample + window_after + 1

                # Extract beat with zero-padding for boundary cases
                if start < 0 or end > len(ecg_signal):
                    # Create zero-padded beat
                    beat = np.zeros(window_before + 1 + window_after)
                    sig_start = max(0, start)
                    sig_end   = min(len(ecg_signal), end)
                    beat_start = sig_start - start
                    beat_end   = beat_start + (sig_end - sig_start)
                    beat[beat_start:beat_end] = ecg_signal[sig_start:sig_end]
                else:
                    beat = ecg_signal[start:end]

                # Z-score normalization (per-beat)
                # Removes inter-patient amplitude differences, baseline wander
                std = np.std(beat)
                if std < 1e-6:
                    continue  # Skip flat/degenerate beats
                beat = (beat - np.mean(beat)) / std

                # Binary label assignment
                label = 0 if symbol in normal_symbols else 1

                all_beats.append(beat)
                all_labels.append(label)

        except Exception as e:
            print(f"[WARN] Could not load record {rec_id}: {e}")
            continue

    X = np.array(all_beats,  dtype=np.float32)
    y = np.array(all_labels, dtype=np.int8)

    n_normal = np.sum(y == 0)
    n_arrhy  = np.sum(y == 1)
    print(f"       Beats loaded: {len(y)} | Normal: {n_normal} | Arrhythmia: {n_arrhy} "
          f"| Ratio: {n_arrhy/max(n_normal,1):.3f}")

    return X, y


# =============================================================================
# SECTION 6: FEATURE EXTRACTION
# =============================================================================

def extract_features(beats, fs):
    """
    Extract time-domain, morphological, and frequency-domain features
    from each ECG beat window.

    FEATURE RELEVANCE IN ARRHYTHMIA DETECTION
    ------------------------------------------
    Raw ECG signals contain 187 data points per beat. Feeding raw signals
    directly into a Random Forest would:
        (a) ignore domain knowledge about ECG physiology;
        (b) make the model highly susceptible to noise;
        (c) prevent interpretable feature importance analysis.

    Feature engineering transforms each beat into a compact, physiologically
    meaningful descriptor vector. Each feature group targets a different
    aspect of arrhythmia pathophysiology:

    TIME-DOMAIN FEATURES
    --------------------
    Mean        : DC offset after normalization; residual drift indicator.
    Std/Var     : Beat amplitude variability; high in ectopic beats with
                  irregular morphology (PVCs, PACs).
    RMS         : Signal power, correlates with QRS amplitude.
    Peak-to-peak: Reflects depolarization amplitude; attenuated in bundle
                  branch blocks, elevated in ventricular hypertrophy.
    Skewness    : Asymmetry of QRS morphology; negative in left BBB,
                  positive in some SVT morphologies.
    Kurtosis    : Peakedness of distribution; high kurtosis indicates
                  sharp, well-defined QRS (Normal); low kurtosis in broad
                  beats (ventricular beats, BBB).

    MORPHOLOGICAL FEATURES
    ----------------------
    R-peak amplitude: Direct measure of ventricular depolarization height.
                      Abnormal R-peak position (not at center of window)
                      suggests annotation errors or irregular timing.
    R-peak index    : Temporal position of the maximum sample. In perfectly
                      extracted beats this is ~93; deviations indicate rate
                      irregularity or window shift.
    QRS width       : Approximated by counting samples > threshold × R-amp.
                      Wide QRS (>120 ms / ~43 samples at 360 Hz) indicates
                      ventricular origin (PVC) or bundle branch block.
    Signal energy   : Integral of squared amplitude over the window. Higher
                      in high-amplitude ventricular beats.
    Zero-crossing rate: Rate of sign changes. High ZCR indicates noise or
                      irregular oscillation (flutter, fibrillation). Low ZCR
                      indicates smooth, Normal morphology.

    FREQUENCY-DOMAIN FEATURES
    --------------------------
    Spectral analysis via FFT decomposes the beat into its frequency components:
    Low-freq power  (<10 Hz) : Captures P-wave and T-wave energy; reflects
                               atrial activity and repolarization.
    Mid-freq power  (10–30 Hz): QRS complex energy; primary discriminator
                               between Normal and ventricular morphologies.
    High-freq power (>30 Hz) : Noise and high-frequency notching; elevated
                               in bundle branch blocks and fragmented QRS.
    Dominant freq   : Frequency bin with maximum spectral power; Normal beats
                      peak around 10–20 Hz; arrhythmic beats may shift.
    Spectral entropy: Measures distribution of spectral power. Low entropy =
                      power concentrated at dominant frequency (Normal QRS).
                      High entropy = diffuse spectrum (irregular/noisy beats).

    Parameters
    ----------
    beats : np.ndarray, shape (n_beats, window_size)
    fs    : int

    Returns
    -------
    features : np.ndarray, shape (n_beats, n_features)
    feature_names : list of str
    """
    n_beats = beats.shape[0]
    feature_list = []
    feature_names = []

    # Frequency axis for FFT
    N = beats.shape[1]
    freqs = np.fft.rfftfreq(N, d=1.0/fs)  # frequency bins in Hz

    for beat in beats:
        feat = []

        # ------------------------------------------------------------------
        # A. TIME-DOMAIN FEATURES
        # ------------------------------------------------------------------
        mean_val    = np.mean(beat)
        std_val     = np.std(beat)
        var_val     = np.var(beat)
        rms_val     = np.sqrt(np.mean(beat**2))
        ptp_val     = np.ptp(beat)                      # peak-to-peak
        skew_val    = skew(beat)
        kurt_val    = kurtosis(beat)

        feat += [mean_val, std_val, var_val, rms_val,
                 ptp_val, skew_val, kurt_val]

        # ------------------------------------------------------------------
        # B. MORPHOLOGICAL FEATURES
        # ------------------------------------------------------------------
        # R-peak amplitude and position
        r_amp_idx   = np.argmax(np.abs(beat))           # index of max |amplitude|
        r_amp       = beat[r_amp_idx]                   # signed amplitude at R-peak
        r_idx_norm  = r_amp_idx / N                     # normalized position [0,1]

        # QRS width: samples where |beat| > 0.5 × |R_amp| (threshold-based)
        qrs_thresh  = 0.5 * abs(r_amp)
        qrs_mask    = np.abs(beat) > qrs_thresh
        qrs_width   = np.sum(qrs_mask)                  # in samples

        # Signal energy
        energy      = np.sum(beat**2)

        # Zero-crossing rate
        # Number of times signal crosses zero, normalized by window length
        signs       = np.sign(beat)
        signs[signs == 0] = 1  # treat zero as positive
        zcr         = np.sum(np.diff(signs) != 0) / (N - 1)

        feat += [r_amp, r_idx_norm, qrs_width, energy, zcr]

        # ------------------------------------------------------------------
        # C. FREQUENCY-DOMAIN FEATURES
        # ------------------------------------------------------------------
        fft_vals    = np.abs(np.fft.rfft(beat))         # magnitude spectrum
        fft_power   = fft_vals**2                        # power spectrum

        # Band power masks
        low_mask    = freqs < 10
        mid_mask    = (freqs >= 10) & (freqs < 30)
        high_mask   = freqs >= 30

        total_power  = np.sum(fft_power) + 1e-10        # avoid div-by-zero
        lf_power     = np.sum(fft_power[low_mask])  / total_power
        mf_power     = np.sum(fft_power[mid_mask])  / total_power
        hf_power     = np.sum(fft_power[high_mask]) / total_power

        # Dominant frequency
        dom_freq_idx = np.argmax(fft_power)
        dom_freq     = freqs[dom_freq_idx]

        # Spectral entropy (Shannon entropy of normalized power spectrum)
        p_norm       = fft_power / total_power
        p_norm       = p_norm[p_norm > 0]               # log(0) guard
        spec_entropy = -np.sum(p_norm * np.log2(p_norm))

        feat += [lf_power, mf_power, hf_power, dom_freq, spec_entropy]

        feature_list.append(feat)

    # Build feature name list (for interpretability / IEEE reporting)
    if not feature_names:
        feature_names = [
            # Time-domain
            'td_mean', 'td_std', 'td_var', 'td_rms',
            'td_ptp', 'td_skewness', 'td_kurtosis',
            # Morphological
            'morph_r_amplitude', 'morph_r_position',
            'morph_qrs_width', 'morph_energy', 'morph_zcr',
            # Frequency-domain
            'freq_lf_power', 'freq_mf_power', 'freq_hf_power',
            'freq_dominant_freq', 'freq_spectral_entropy'
        ]

    features = np.array(feature_list, dtype=np.float32)
    return features, feature_names


# =============================================================================
# SECTION 7: DATA PIPELINE — LOAD ALL SPLITS
# =============================================================================

def build_dataset(split_ids, data_path, normal_symbols,
                  window_before, window_after, fs, split_name=''):
    """
    Full pipeline: load beats → extract features → return (X_feat, y).
    """
    print(f"\n{'='*60}")
    print(f"  Loading {split_name} split ({len(split_ids)} patients)...")
    print(f"{'='*60}")

    X_raw, y = load_and_extract_beats(
        split_ids, data_path, normal_symbols,
        window_before, window_after, fs
    )

    print(f"  Extracting features from {len(y)} beats...")
    X_feat, feat_names = extract_features(X_raw, fs)

    print(f"  Feature matrix shape: {X_feat.shape}")
    return X_feat, y, feat_names


# =============================================================================
# SECTION 8: RANDOM FOREST TRAINING & HYPERPARAMETER TUNING
# =============================================================================

def tune_and_train(X_train, y_train, X_val, y_val, feature_names):
    """
    Hyperparameter search on validation set using manual grid search.

    WHY RANDOM FOREST OVER DECISION TREE
    -------------------------------------
    A single Decision Tree is a high-variance model: small perturbations in
    training data produce structurally different trees, leading to OVERFITTING.
    It memorizes noise in training data rather than learning generalizable
    patterns.

    Random Forest (Breiman, 2001) addresses this via:

    1. BOOTSTRAP AGGREGATION (BAGGING):
       Each tree is trained on a bootstrap sample (random sampling WITH
       replacement) of the training data. This introduces diversity among
       trees — each sees a slightly different version of the dataset, preventing
       any single tree from memorizing outliers.

    2. RANDOM FEATURE SUBSAMPLING:
       At each node split, only a random subset (√n_features for classification)
       of features is considered. This de-correlates the trees: even if one
       feature strongly predicts the outcome, not all trees will use it at the
       root, forcing the ensemble to discover complementary features (e.g., both
       QRS width AND spectral entropy).

    3. ENSEMBLE AVERAGING (MAJORITY VOTING):
       The final prediction is the majority class vote across all n_estimators
       trees. By the Central Limit Theorem, the variance of the ensemble mean
       decreases as O(1/n_estimators), systematically reducing overfitting risk.
       For classification: Var(ensemble) ≈ ρσ² + (1-ρ)σ²/B
       where ρ = inter-tree correlation, σ² = single tree variance, B = n trees.
       Random feature subsampling reduces ρ, amplifying variance reduction.

    HYPERPARAMETER JUSTIFICATION
    ----------------------------
    n_estimators    : More trees → lower variance, diminishing returns >200.
                      100 and 200 tested as practical trade-off.
    max_depth       : Limits tree complexity. None = full growth (high variance);
                      10/20 = regularization to prevent overfitting on minority class.
    min_samples_split: Minimum samples required to split a node. Higher values
                      prevent overfitting on rare arrhythmia patterns.
    min_samples_leaf : Minimum samples in leaf node. Smooths decision boundaries.
    class_weight    : 'balanced' automatically computes per-class weights as
                      n_samples / (n_classes × np.bincount(y)), upweighting the
                      minority arrhythmia class proportionally. This is equivalent
                      to cost-sensitive learning without any data augmentation.

    Parameters
    ----------
    X_train, y_train : training data and labels
    X_val, y_val     : validation data and labels
    feature_names    : list of str

    Returns
    -------
    best_model : fitted RandomForestClassifier
    best_params : dict
    results_df  : pd.DataFrame of all hyperparameter combinations
    """
    # Hyperparameter grid (manually defined for transparency)
    param_grid = {
        'n_estimators'    : [100, 200],
        'max_depth'       : [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf' : [1, 3, 5],
    }

    results = []
    best_f1 = -1
    best_model = None
    best_params = None

    total_combinations = (len(param_grid['n_estimators']) *
                          len(param_grid['max_depth']) *
                          len(param_grid['min_samples_split']) *
                          len(param_grid['min_samples_leaf']))
    print(f"\n[TUNING] Grid search over {total_combinations} hyperparameter combinations...")
    print(f"[TUNING] Scoring metric: F1-score (macro) on validation set\n")

    combo_idx = 0
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    combo_idx += 1

                    clf = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        class_weight='balanced',   # handle imbalance
                        random_state=RANDOM_SEED,
                        n_jobs=-1                  # use all CPU cores
                    )
                    clf.fit(X_train, y_train)

                    # Evaluate on validation set
                    y_val_pred = clf.predict(X_val)
                    val_f1  = f1_score(y_val, y_val_pred, average='macro')
                    val_acc = accuracy_score(y_val, y_val_pred)
                    val_rec = recall_score(y_val, y_val_pred, average='macro')

                    results.append({
                        'n_estimators'     : n_est,
                        'max_depth'        : str(max_d),  # None → string for display
                        'min_samples_split': min_split,
                        'min_samples_leaf' : min_leaf,
                        'val_f1'           : val_f1,
                        'val_accuracy'     : val_acc,
                        'val_recall'       : val_rec
                    })

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_model = clf
                        best_params = {
                            'n_estimators'     : n_est,
                            'max_depth'        : max_d,
                            'min_samples_split': min_split,
                            'min_samples_leaf' : min_leaf
                        }

                    if combo_idx % 9 == 0:
                        print(f"  [{combo_idx}/{total_combinations}] "
                              f"n_est={n_est}, depth={max_d}, "
                              f"split={min_split}, leaf={min_leaf} "
                              f"→ val_F1={val_f1:.4f}")

    results_df = pd.DataFrame(results)
    print(f"\n[TUNING] Best hyperparameters: {best_params}")
    print(f"[TUNING] Best validation F1   : {best_f1:.4f}")

    return best_model, best_params, results_df


# =============================================================================
# SECTION 9: EVALUATION
# =============================================================================

def evaluate_model(model, X, y, split_name='Test', threshold=0.5):
    """
    Comprehensive evaluation of classifier on a given split.

    METRICS JUSTIFICATION (IEEE STYLE)
    -----------------------------------
    For clinical ECG classification with class imbalance, accuracy alone is
    misleading (a classifier predicting all-Normal achieves high accuracy on
    an 80% Normal dataset). The following metrics provide a complete picture:

    Accuracy    : (TP+TN)/(TP+TN+FP+FN) — overall correctness.
    Precision   : TP/(TP+FP) — of detected arrhythmias, fraction truly
                  arrhythmic. Low precision → many false alarms (clinician burden).
    Recall/Sens.: TP/(TP+FN) — of true arrhythmias, fraction detected.
                  Low recall → missed arrhythmias (life-threatening in clinical
                  context). Recall is the PRIMARY clinical metric.
    F1-score    : Harmonic mean of Precision & Recall. Preferred single-number
                  summary for imbalanced datasets (balances FP vs FN costs).
    ROC-AUC     : Area under the Receiver Operating Characteristic curve.
                  Measures discriminative ability across ALL thresholds;
                  threshold-independent metric essential for comparing classifiers.
                  AUC = 1.0 → perfect separation; AUC = 0.5 → random chance.

    Parameters
    ----------
    model      : fitted RandomForestClassifier
    X, y       : feature matrix and true labels
    split_name : str (for printing)
    threshold  : float, decision threshold for positive class

    Returns
    -------
    metrics : dict
    y_prob  : np.ndarray, predicted probabilities for class 1
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    auc  = roc_auc_score(y, y_prob)
    cm   = confusion_matrix(y, y_pred)

    print(f"\n{'='*60}")
    print(f"  EVALUATION — {split_name.upper()} SET")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y, y_pred, target_names=['Normal', 'Arrhythmia'],
                                zero_division=0))

    return {
        'split': split_name, 'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'auc': auc,
        'cm': cm, 'y_pred': y_pred, 'y_prob': y_prob
    }, y_prob


# =============================================================================
# SECTION 10: VISUALIZATIONS
# =============================================================================

def plot_confusion_matrix(cm, split_name, save_path=None):
    """Plot and optionally save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Arrhythmia'],
                yticklabels=['Normal', 'Arrhythmia'],
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {split_name} Set', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_prob, split_name, auc_score, save_path=None):
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(f'ROC Curve — {split_name} Set', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=17, save_path=None):
    """
    Plot and explain Gini-based feature importances.

    GINI IMPORTANCE (MEAN DECREASE IN IMPURITY)
    -------------------------------------------
    At each split node, Random Forest selects the feature that maximally reduces
    Gini impurity: Gini(t) = 1 − Σ p_k². The importance of feature j is the
    weighted sum of Gini reduction across ALL nodes where j is used, averaged
    over all trees. Higher importance → feature is more frequently chosen as
    a split criterion AND produces purer child nodes.

    CLINICAL INTERPRETATION
    -----------------------
    Features ranked highest reveal which ECG characteristics are most
    discriminative between Normal and Arrhythmia:
    • QRS width (morph_qrs_width): Wide QRS is the hallmark of ventricular
      ectopy (PVCs) and bundle branch blocks.
    • Spectral entropy (freq_spectral_entropy): Arrhythmic beats have more
      diffuse frequency content than the well-concentrated QRS spectrum.
    • Kurtosis (td_kurtosis): Normal beats have highly peaked, symmetric QRS;
      arrhythmic morphologies are broader and asymmetric.
    • R-peak amplitude (morph_r_amplitude): Altered in hypertrophy, ischemia,
      and ventricular beats.
    This ranking provides model transparency and supports clinical validation.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(indices)))
    bars = ax.barh(range(len(indices)), top_vals[::-1], color=colors[::-1],
                   edgecolor='white', height=0.7)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel('Gini Importance (Mean Decrease in Impurity)', fontsize=11)
    ax.set_title('Random Forest Feature Importance\n(Top Features by Gini Index)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{top_vals[::-1][i]:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()

    print("\n[FEATURE IMPORTANCE RANKING]")
    for rank, idx in enumerate(indices):
        print(f"  {rank+1:2d}. {feature_names[idx]:<30s} : {importances[idx]:.4f}")


def plot_hyperparameter_analysis(results_df, save_path=None):
    """
    Visualize validation F1-score across hyperparameter combinations.

    Since Random Forest has no epoch-based loss curve (it is a non-iterative
    ensemble model), learning behavior is analyzed through the hyperparameter
    sensitivity landscape. This serves as the equivalent of a 'learning curve'
    — showing how model performance evolves with increasing model complexity
    (n_estimators, max_depth) and regularization (min_samples_split/leaf).

    This analysis is essential for IEEE papers to demonstrate:
        (a) Robustness of the chosen hyperparameters
        (b) Absence of severe overfitting across the search space
        (c) Marginal returns from increasing complexity beyond a threshold
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: n_estimators vs max_depth heatmap (avg F1) ---
    pivot1 = results_df.groupby(['n_estimators', 'max_depth'])['val_f1'].mean().unstack()
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Val F1-Score: n_estimators × max_depth\n(averaged over min_samples)',
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Max Depth', fontsize=10)
    axes[0].set_ylabel('n_estimators', fontsize=10)

    # --- Plot 2: n_estimators effect on F1 (line plot per max_depth) ---
    for depth, grp in results_df.groupby('max_depth'):
        mean_f1 = grp.groupby('n_estimators')['val_f1'].mean()
        axes[1].plot(mean_f1.index, mean_f1.values, marker='o', label=f'depth={depth}')
    axes[1].set_xlabel('n_estimators', fontsize=10)
    axes[1].set_ylabel('Mean Validation F1-Score', fontsize=10)
    axes[1].set_title('Effect of n_estimators on Validation F1\n(Learning Behavior Analysis)',
                      fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_metrics_summary(train_metrics, val_metrics, test_metrics, save_path=None):
    """Bar chart comparing accuracy, precision, recall, F1, AUC across splits."""
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metrics, split) in enumerate(
        zip([train_metrics, val_metrics, test_metrics], ['Train', 'Val', 'Test'])
    ):
        vals = [metrics[k] for k in metrics_keys]
        ax.bar(x + i*width, vals, width=width, label=split,
               alpha=0.85, edgecolor='white')

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Performance Metrics Across Splits', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


# =============================================================================
# SECTION 11: DATASET STATISTICS
# =============================================================================

def print_dataset_statistics(y_train, y_val, y_test):
    """Print detailed class distribution across all splits."""
    print("\n" + "="*60)
    print("  DATASET STATISTICS")
    print("="*60)
    for name, y in [('Training', y_train), ('Validation', y_val), ('Test', y_test)]:
        n_total  = len(y)
        n_normal = np.sum(y == 0)
        n_arrhy  = np.sum(y == 1)
        print(f"\n  {name} Set:")
        print(f"    Total beats  : {n_total:,}")
        print(f"    Normal       : {n_normal:,} ({100*n_normal/n_total:.1f}%)")
        print(f"    Arrhythmia   : {n_arrhy:,} ({100*n_arrhy/n_total:.1f}%)")
        print(f"    Imbalance ratio : {n_arrhy/max(n_normal,1):.3f}")


# =============================================================================
# SECTION 12: MAIN PIPELINE
# =============================================================================

def main():
    """
    Main pipeline orchestrating the full ECG classification experiment.
    """
    print("\n" + "="*70)
    print("  IEEE-GRADE ECG ARRHYTHMIA CLASSIFICATION — MIT-BIH DATASET")
    print("  Random Forest with Patient-Wise Split & Feature Engineering")
    print("="*70)

    # -----------------------------------------------------------------------
    # STEP 1: Record discovery and patient-wise split
    # -----------------------------------------------------------------------
    record_ids = get_record_ids(DATA_PATH, EXCLUDED_RECORDS)
    train_ids, val_ids, test_ids = patient_wise_split(
        record_ids, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
    )

    # -----------------------------------------------------------------------
    # STEP 2: Build datasets (load → extract beats → extract features)
    # -----------------------------------------------------------------------
    X_train, y_train, feat_names = build_dataset(
        train_ids, DATA_PATH, NORMAL_SYMBOLS,
        WINDOW_BEFORE, WINDOW_AFTER, FS, 'Training'
    )
    X_val, y_val, _ = build_dataset(
        val_ids, DATA_PATH, NORMAL_SYMBOLS,
        WINDOW_BEFORE, WINDOW_AFTER, FS, 'Validation'
    )
    X_test, y_test, _ = build_dataset(
        test_ids, DATA_PATH, NORMAL_SYMBOLS,
        WINDOW_BEFORE, WINDOW_AFTER, FS, 'Test'
    )

    # -----------------------------------------------------------------------
    # STEP 3: Dataset statistics
    # -----------------------------------------------------------------------
    print_dataset_statistics(y_train, y_val, y_test)

    # -----------------------------------------------------------------------
    # STEP 4: Hyperparameter tuning on validation set
    # -----------------------------------------------------------------------
    best_model, best_params, results_df = tune_and_train(
        X_train, y_train, X_val, y_val, feat_names
    )

    # -----------------------------------------------------------------------
    # STEP 5: Evaluation on all three splits
    # -----------------------------------------------------------------------
    train_res, y_train_prob = evaluate_model(best_model, X_train, y_train, 'Training')
    val_res,   y_val_prob   = evaluate_model(best_model, X_val,   y_val,   'Validation')
    test_res,  y_test_prob  = evaluate_model(best_model, X_test,  y_test,  'Test')

    # -----------------------------------------------------------------------
    # STEP 6: Visualizations
    # -----------------------------------------------------------------------
    print("\n[VIZ] Generating plots...")

    plot_confusion_matrix(test_res['cm'], 'Test',
                          save_path='/content/confusion_matrix_test.png')

    plot_roc_curve(y_test, y_test_prob, 'Test', test_res['auc'],
                   save_path='/content/roc_curve_test.png')

    plot_feature_importance(best_model, feat_names,
                            save_path='/content/feature_importance.png')

    plot_hyperparameter_analysis(results_df,
                                 save_path='/content/hyperparameter_analysis.png')

    plot_metrics_summary(train_res, val_res, test_res,
                         save_path='/content/metrics_summary.png')

    # -----------------------------------------------------------------------
    # STEP 7: Final summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("  EXPERIMENT SUMMARY")
    print("="*70)
    print(f"  Best Parameters       : {best_params}")
    print(f"  Total Features        : {len(feat_names)}")
    print(f"  Training Accuracy     : {train_res['accuracy']:.4f}")
    print(f"  Validation Accuracy   : {val_res['accuracy']:.4f}")
    print(f"  Test Accuracy         : {test_res['accuracy']:.4f}")
    print(f"  Test F1-Score         : {test_res['f1']:.4f}")
    print(f"  Test ROC-AUC          : {test_res['auc']:.4f}")

    print("\n  Top 5 Features by Importance:")
    importances = best_model.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    for rank, idx in enumerate(top5):
        print(f"    {rank+1}. {feat_names[idx]:<35s} : {importances[idx]:.4f}")

    return best_model, feat_names, results_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Mount Google Drive first in Colab:
    #   from google.colab import drive
    #   drive.mount('/content/drive')
    best_model, feat_names, results_df = main(
    )