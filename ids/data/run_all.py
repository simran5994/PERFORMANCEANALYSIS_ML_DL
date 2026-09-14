"""
================================================================================
 ML and DL comparison for IoT intrusion detection using the ToN-IoT
network dataset.

 Shimran Panigrahi

 RUN:
     python run_all.py

 Input:
    data/*.csv

 Output:
    results/
    results/figures/
"""

# Set FAST_MODE = True for a ~15 minute run that produces every table and figure
# on a 150k-row sample.
# FAST_MODE uses a 150k-row sample for a shorter test run.
FAST_MODE = False

# Set to False on the full run. True skips the deep-learning cross-validation
# folds, which are by far the slowest part.
SKIP_DL_CROSSVAL = False

DATA_DIR    = "data"        # where your ToN-IoT CSV lives
RESULTS_DIR = "results"
FIG_DIR     = "results/figures"
SEED        = 42

FAST_ROWS   = 150_000       # sample size when FAST_MODE
SVM_MAX     = 50_000        # SVM training cap (Section 3.4 justifies this)

# Limit DL training size to keep CPU training time manageable.
DL_MAX_TRAIN = 40_000
DL_EPOCHS   = 40
DL_BATCH    = 256

# ==============================================================================

import os
import sys
import time
import json
import glob
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()


def banner(text, char="="):
    print("\n" + char * 78)
    print(f"  {text}")
    print(char * 78)


def step(n, text):
    print(f"\n[{n}] {text}")
    print("-" * 78)


# ------------------------------------------------------------------ dependency
banner("STEP 0 — CHECKING DEPENDENCIES")

MISSING = []
try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")
try:
    import pandas as pd
except ImportError:
    MISSING.append("pandas")
try:
    import matplotlib
    matplotlib.use("Agg")          # no display needed
    import matplotlib.pyplot as plt
except ImportError:
    MISSING.append("matplotlib")
try:
    import seaborn as sns
except ImportError:
    MISSING.append("seaborn")
try:
    from scipy import stats
except ImportError:
    MISSING.append("scipy")
try:
    import sklearn
except ImportError:
    MISSING.append("scikit-learn")

if MISSING:
    print("\nMissing required packages:", ", ".join(MISSING))
    print("\nInstall them with:")
    print(f"   pip install {' '.join(MISSING)}")
    sys.exit(1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, roc_auc_score)
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow is optional. If unavailable, run the ML models only.
HAS_TF = True
TF_ERROR = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Conv1D, MaxPooling1D, Flatten,
                                         LSTM, Dropout, BatchNormalization)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    tf.random.set_seed(SEED)
except Exception as _e:
    HAS_TF = False
    TF_ERROR = _e

# SHAP is optional — Section 5.6 is skipped without it.
HAS_SHAP = True
try:
    import shap
except ImportError:
    HAS_SHAP = False

np.random.seed(SEED)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300,
                     "font.size": 11, "axes.grid": True, "grid.alpha": .3})

print(f"  numpy         {np.__version__}")
print(f"  pandas        {pd.__version__}")
print(f"  scikit-learn  {sklearn.__version__}")
if HAS_TF:
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  tensorflow    {tf.__version__}   GPU: {gpus if gpus else 'none (CPU)'}")
else:
    print("  tensorflow    UNAVAILABLE — CNN and LSTM will be skipped")
    print(f"                reason: {type(TF_ERROR).__name__}: "
          f"{str(TF_ERROR)[:150]}")
    if TF_ERROR is not None and "metal_plugin" in str(TF_ERROR):
        print("                FIX: pip uninstall -y tensorflow-metal")
    else:
        print("                FIX: pip install tensorflow")
    print("                (Random Forest and SVM will still run and report.)")
print(f"  shap          {'ok' if HAS_SHAP else 'NOT INSTALLED — Section 5.6 skipped'}")
print(f"\n  FAST_MODE = {FAST_MODE}"
      f"{f' (sampling {FAST_ROWS:,} rows)' if FAST_MODE else ' (full dataset)'}")

RUN_T0 = time.time()

# ------------------------------------------------------------------- load data
banner("STEP 1 — LOADING DATASET")

candidates = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not candidates:
    print(f"\nNo CSV found in ./{DATA_DIR}/")
    print("\nFix: download the ToN-IoT *network* dataset from")
    print("     https://research.unsw.edu.au/projects/toniot-datasets")
    print(f"     and put the CSV in ./{DATA_DIR}/")
    sys.exit(1)

# Prefer a file that looks like the network subset
pick = max(candidates, key=os.path.getsize)
for c in candidates:
    if "network" in os.path.basename(c).lower():
        pick = c
        break

print(f"  reading {pick}  ({os.path.getsize(pick)/1e6:.1f} MB)")
df = pd.read_csv(pick, low_memory=False)
df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]   # strip BOM/whitespace
print(f"  shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

if FAST_MODE and len(df) > FAST_ROWS:
    df = df.sample(n=FAST_ROWS, random_state=SEED).reset_index(drop=True)
    print(f"  FAST_MODE: sampled down to {len(df):,} rows")

# ------------------------------------------------- auto-detect label columns
step("1b", "Detecting label columns")

BINARY_LABEL = None
ATTACK_TYPE = None

for cand in ("label", "Label", "is_attack", "class"):
    if cand in df.columns and df[cand].nunique() <= 3:
        BINARY_LABEL = cand
        break
for cand in ("type", "Type", "attack_cat", "attack_type", "category"):
    if cand in df.columns and 2 < df[cand].nunique() <= 20:
        ATTACK_TYPE = cand
        break

if BINARY_LABEL is None or ATTACK_TYPE is None:
    print("\n  Could not auto-detect. Low-cardinality columns found:")
    for c in df.columns:
        if df[c].nunique() <= 20:
            print(f"    {c:24s} {df[c].nunique():3d} values: "
                  f"{list(df[c].unique())[:6]}")
    print("\n  Fix: set BINARY_LABEL and ATTACK_TYPE manually near the top of")
    print("       STEP 1b and re-run.")
    sys.exit(1)

print(f"  binary label column : '{BINARY_LABEL}'")
print(f"  attack type column  : '{ATTACK_TYPE}'")

# Normalise binary label to 0/1
if not pd.api.types.is_numeric_dtype(df[BINARY_LABEL]):
    df[BINARY_LABEL] = (df[BINARY_LABEL].astype(str).str.lower()
                        .map({"normal": 0, "0": 0, "attack": 1, "1": 1}))
df[BINARY_LABEL] = pd.to_numeric(df[BINARY_LABEL], errors="coerce").fillna(0).astype(int)

# ------------------------------------------------------------------------ EDA
banner("STEP 2 — EXPLORATORY DATA ANALYSIS  (Chapter 4)")

print("  Binary distribution:")
for v, n in df[BINARY_LABEL].value_counts().sort_index().items():
    print(f"    {'normal' if v == 0 else 'attack':8s} {n:>8,}  ({n/len(df)*100:5.2f}%)")

type_counts = df[ATTACK_TYPE].value_counts()
print("\n  Attack type distribution:")
for k, v in type_counts.items():
    print(f"    {str(k):14s} {v:>8,}  ({v/len(df)*100:5.2f}%)")
print(f"\n  imbalance ratio (largest:smallest) = "
      f"{type_counts.max()/max(type_counts.min(),1):.0f} : 1")

fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
df[BINARY_LABEL].value_counts().sort_index().plot.bar(
    ax=ax[0], color=["#27ae60", "#e74c3c"], edgecolor="black")
ax[0].set_title("Binary Class Distribution", fontweight="bold")
ax[0].set_xlabel("0 = Normal, 1 = Attack"); ax[0].set_ylabel("Samples")
ax[0].tick_params(axis="x", rotation=0)
for i, v in enumerate(df[BINARY_LABEL].value_counts().sort_index()):
    ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")

type_counts.plot.bar(ax=ax[1], color=plt.cm.Set3.colors, edgecolor="black")
ax[1].set_title("Attack Type Distribution", fontweight="bold")
ax[1].set_ylabel("Samples"); ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_1_class_distribution.png", bbox_inches="tight")
plt.close()
print(f"\n  saved -> fig_4_1_class_distribution.png")

# --------------------------------------------------------------- preprocessing
banner("STEP 3 — PREPROCESSING PIPELINE  (Chapter 3 / Chapter 4)")

prep_log = {}

step("3.1", "Missing and invalid value treatment")
# Convert numeric-looking non-numeric columns before imputation.
for c in df.columns:
    if c in (BINARY_LABEL, ATTACK_TYPE):
        continue
    if not pd.api.types.is_numeric_dtype(df[c]):
        conv = pd.to_numeric(df[c], errors="coerce")
        # convert only if most values really are numeric (handles '-' markers)
        if conv.notna().mean() > 0.5:
            df[c] = conv
df.replace([np.inf, -np.inf], np.nan, inplace=True)
n_missing = int(df.isnull().sum().sum())
for c in df.select_dtypes(include=[np.number]).columns:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())
for c in df.columns:
    if c in (BINARY_LABEL, ATTACK_TYPE) or pd.api.types.is_numeric_dtype(df[c]):
        continue
    if df[c].isnull().any():
        mode = df[c].mode()
        df[c] = df[c].fillna(mode[0] if not mode.empty else "unknown")
print(f"  missing / invalid cells found: {n_missing:,}  -> imputed (median / mode)")
print(f"  remaining: {int(df.isnull().sum().sum())}")
prep_log["missing_imputed"] = n_missing

step("3.2", "Identifier removal (prevents overfitting to device identity)")
# Remove known identifier fields.
ID_COLS = {"src_ip", "dst_ip", "source_ip", "dest_ip", "src_port", "dst_port",
           "timestamp", "ts", "flow_id", "flow id", "id"}
def _norm(c):
    return c.replace("\ufeff", "").strip().lower()
drop = [c for c in df.columns
        if c not in (BINARY_LABEL, ATTACK_TYPE) and _norm(c) in ID_COLS]
drop = sorted(set(drop))
print(f"  dropping {len(drop)}: {drop}")
df.drop(columns=drop, inplace=True, errors="ignore")
prep_log["identifiers_dropped"] = drop

step("3.3", "Categorical encoding")
le_type = LabelEncoder()
df["_y_multi"] = le_type.fit_transform(df[ATTACK_TYPE].astype(str))
CLASS_NAMES = list(le_type.classes_)
N_CLASSES = len(CLASS_NAMES)
print(f"  {N_CLASSES} classes: {CLASS_NAMES}")

cat_cols = [c for c in df.columns
            if c not in (BINARY_LABEL, ATTACK_TYPE, "_y_multi")
            and not pd.api.types.is_numeric_dtype(df[c])]
for c in cat_cols:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    print(f"  encoded '{c}'")
if not cat_cols:
    print("  (no remaining categorical columns)")
prep_log["categorical_encoded"] = cat_cols

step("3.4", "Feature / label separation")
excl = {BINARY_LABEL, ATTACK_TYPE, "_y_multi"}
feat_cols = [c for c in df.columns if c not in excl]
X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
# drop any zero-variance columns — they carry no information
zerovar = [c for c in X.columns if X[c].nunique() <= 1]
if zerovar:
    X = X.drop(columns=zerovar)
    print(f"  dropped {len(zerovar)} zero-variance columns")
y_bin = df[BINARY_LABEL].values
y_mul = df["_y_multi"].values
print(f"  X = {X.shape[0]:,} x {X.shape[1]}")

step("3.5", "Correlation filtering (threshold 0.95)")
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
redundant = [c for c in upper.columns if (upper[c] > 0.95).any()]
X = X.drop(columns=redundant)
FEATURES = list(X.columns)
print(f"  removed {len(redundant)}: {redundant if len(redundant) < 12 else redundant[:12] + ['...']}")
print(f"  final feature count: {len(FEATURES)}")
prep_log["correlated_removed"] = redundant
prep_log["final_features"] = FEATURES

# correlation heatmap
plt.figure(figsize=(14, 11))
sns.heatmap(X.corr(), cmap="RdBu_r", center=0, linewidths=.2)
plt.title("Feature Correlation Heatmap (after filtering)", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_2_correlation.png", bbox_inches="tight")
plt.close()

# ------------------------------------------------------------ feature ranking
step("3.6", "Feature importance ranking")
rf_probe = RandomForestClassifier(n_estimators=50, max_depth=10,
                                  random_state=SEED, n_jobs=-1).fit(X, y_bin)
gini = pd.Series(rf_probe.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("  top 10 by Gini importance:")
for i, (f, v) in enumerate(gini.head(10).items(), 1):
    print(f"    {i:2d}. {f:28s} {v:.4f}")
gini.to_csv(f"{RESULTS_DIR}/feature_importance_gini.csv")

plt.figure(figsize=(11, 8))
gini.head(20).sort_values().plot.barh(color="steelblue", edgecolor="black")
plt.title("Top 20 Features — Random Forest Gini Importance", fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_3_feature_importance.png", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------- splits
banner("STEP 4 — STRATIFIED SPLIT AND SCALING")

Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(
    X, y_bin, test_size=.2, random_state=SEED, stratify=y_bin)
Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(
    X, y_mul, test_size=.2, random_state=SEED, stratify=y_mul)

sc_b = MinMaxScaler().fit(Xtr_b)          # fitted on TRAIN only — no leakage
Xtr_bs, Xte_bs = sc_b.transform(Xtr_b), sc_b.transform(Xte_b)
sc_m = MinMaxScaler().fit(Xtr_m)
Xtr_ms, Xte_ms = sc_m.transform(Xtr_m), sc_m.transform(Xte_m)

NF = Xtr_bs.shape[1]
print(f"  train {len(Xtr_bs):,}   test {len(Xte_bs):,}   features {NF}")
print(f"  scaler fitted on training partition only (no test leakage)")
print("\n  per-class test counts:")
for i, n in enumerate(CLASS_NAMES):
    print(f"    {str(n):14s} train {int((ytr_m==i).sum()):>7,}   test {int((yte_m==i).sum()):>7,}")

if HAS_TF:
    Xtr_bd, Xte_bd = Xtr_bs.reshape(-1, NF, 1), Xte_bs.reshape(-1, NF, 1)
    Xtr_md, Xte_md = Xtr_ms.reshape(-1, NF, 1), Xte_ms.reshape(-1, NF, 1)
    ytr_m_oh = to_categorical(ytr_m, N_CLASSES)

# ------------------------------------------------------------- metric harness
banner("STEP 5 — MODEL EVALUATION")

RESULTS, PREDS, PROBAS, MODELS_FITTED = {}, {}, {}, {}


def fpr_of(y, p, binary=True):
    cm = confusion_matrix(y, p)
    if binary and cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        return fp / (fp + tn) if (fp + tn) else 0.0
    tot, out = cm.sum(), []
    for i in range(cm.shape[0]):
        fp = cm[:, i].sum() - cm[i, i]
        tn = tot - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        out.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return float(np.mean(out))


def dr_of(y, p, binary=True):
    if binary:
        cm = confusion_matrix(y, p)
        if cm.shape == (2, 2):
            tp, fn = cm[1, 1], cm[1, 0]
            return tp / (tp + fn) if (tp + fn) else 0.0
    return recall_score(y, p, average="macro", zero_division=0)


def auc_of(y, s, binary=True):
    try:
        return roc_auc_score(y, s) if binary else \
               roc_auc_score(y, s, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def collapse_check(name, task, y_pred, y_true):
    """Check whether predictions are dominated by one class."""
    vals, counts = np.unique(y_pred, return_counts=True)
    share = counts.max() / len(y_pred)
    n_classes_true = len(np.unique(y_true))
    if share > 0.95 and n_classes_true > 1:
        print(f"\n  *** WARNING: {name} ({task}) predicted one class for "
              f"{share*100:.1f}% of samples.")
        return True
    return False


def evaluate(name, task, y, p, s, train_s, pred_s):
    binary = task == "Binary"
    collapse_check(name, task, np.asarray(p), np.asarray(y))
    rec = {
        "Model": name, "Task": task,
        "Accuracy":  round(accuracy_score(y, p) * 100, 2),
        "Precision": round(precision_score(y, p, average="weighted", zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y, p, average="weighted", zero_division=0) * 100, 2),
        "F1":        round(f1_score(y, p, average="weighted", zero_division=0) * 100, 2),
        "DetRate":   round(dr_of(y, p, binary) * 100, 2),
        "FPR":       round(fpr_of(y, p, binary) * 100, 4),
        # Keep five decimal places for AUC.
        "ROC_AUC":   round(auc_of(y, s, binary), 5),
        "Latency_ms": round(pred_s / len(y) * 1000, 4),
        "Train_s":   round(train_s, 2),
    }
    RESULTS[f"{name}_{task}"] = rec
    # Save intermediate results after each model.
    try:
        pd.DataFrame(list(RESULTS.values())).to_csv(
            f"{RESULTS_DIR}/_checkpoint_results.csv", index=False)
    except Exception:
        pass
    PREDS[f"{name}_{task}"] = p
    PROBAS[f"{name}_{task}"] = s

    print(f"\n  {name} — {task}")
    print(f"    Accuracy {rec['Accuracy']:>7.2f}%   Precision {rec['Precision']:>7.2f}%"
          f"   Recall {rec['Recall']:>7.2f}%   F1 {rec['F1']:>7.2f}%")
    print(f"    DetRate  {rec['DetRate']:>7.2f}%   FPR       {rec['FPR']:>7.4f}%"
          f"   AUC    {rec['ROC_AUC']:>7.4f}")
    print(f"    Latency  {rec['Latency_ms']:>7.4f} ms/sample   Training {rec['Train_s']:.1f}s")

    names = ["Normal", "Attack"] if binary else [str(c) for c in CLASS_NAMES]
    rep = classification_report(y, p, target_names=names, zero_division=0)
    with open(f"{RESULTS_DIR}/report_{name.replace(' ','_')}_{task}.txt", "w") as f:
        f.write(rep)
    return rec


def save_cm(y, p, name, task, fignum):
    labels = ["Normal", "Attack"] if task == "Binary" else [str(c) for c in CLASS_NAMES]
    plt.figure(figsize=(9, 7) if len(labels) > 2 else (6, 5))
    sns.heatmap(confusion_matrix(y, p), annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, linewidths=.5)
    plt.title(f"{name} — {task} Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_5_{fignum}_cm_{name.lower().replace(' ','_')}_{task.lower()}.png",
                bbox_inches="tight")
    plt.close()


def save_curves(hist, name, task, fignum):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(hist.history["accuracy"], lw=2, label="train")
    ax[0].plot(hist.history["val_accuracy"], lw=2, label="validation")
    ax[0].set_title(f"{name} {task} — Accuracy", fontweight="bold")
    ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(hist.history["loss"], lw=2, label="train")
    ax[1].plot(hist.history["val_loss"], lw=2, label="validation")
    ax[1].set_title(f"{name} {task} — Loss", fontweight="bold")
    ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_5_{fignum}_{name.lower()}_{task.lower()}_curves.png",
                bbox_inches="tight")
    plt.close()
    print(f"    converged at epoch {len(hist.history['loss'])} (early stopping)")


# ------------------------------------------------------------------ ML models
banner("STEP 6 — RANDOM FOREST")


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=30,
                                  min_samples_split=5, min_samples_leaf=2,
                                  random_state=SEED, n_jobs=-1)


rf_b = make_rf()
t = time.time(); rf_b.fit(Xtr_bs, ytr_b); tr = time.time() - t
t = time.time(); p = rf_b.predict(Xte_bs); pt = time.time() - t
evaluate("Random Forest", "Binary", yte_b, p, rf_b.predict_proba(Xte_bs)[:, 1], tr, pt)
save_cm(yte_b, p, "Random Forest", "Binary", 1)

rf_m = make_rf()
t = time.time(); rf_m.fit(Xtr_ms, ytr_m); tr = time.time() - t
t = time.time(); p = rf_m.predict(Xte_ms); pt = time.time() - t
evaluate("Random Forest", "Multi-class", yte_m, p, rf_m.predict_proba(Xte_ms), tr, pt)
save_cm(yte_m, p, "Random Forest", "Multi-class", 8)
MODELS_FITTED["Random Forest"] = (rf_m, Xte_ms, False)

banner("STEP 7 — SUPPORT VECTOR MACHINE")


def svm_subset(Xa, ya, n=SVM_MAX):
    if len(Xa) <= n:
        return Xa, ya
    Xs, _, ys, _ = train_test_split(Xa, ya, train_size=n,
                                    random_state=SEED, stratify=ya)
    return Xs, ys


Xs, ys = svm_subset(Xtr_bs, ytr_b)
print(f"  training on {len(Xs):,} samples (SVM scales O(n^2)-O(n^3))")
sv_b = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED)
t = time.time(); sv_b.fit(Xs, ys); tr = time.time() - t
t = time.time(); p = sv_b.predict(Xte_bs); pt = time.time() - t
evaluate("SVM", "Binary", yte_b, p, sv_b.predict_proba(Xte_bs)[:, 1], tr, pt)
save_cm(yte_b, p, "SVM", "Binary", 2)
print(f"    support vectors: {int(sv_b.n_support_.sum()):,}")

Xs, ys = svm_subset(Xtr_ms, ytr_m)
sv_m = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED)
t = time.time(); sv_m.fit(Xs, ys); tr = time.time() - t
t = time.time(); p = sv_m.predict(Xte_ms); pt = time.time() - t
evaluate("SVM", "Multi-class", yte_m, p, sv_m.predict_proba(Xte_ms), tr, pt)
save_cm(yte_m, p, "SVM", "Multi-class", 9)
MODELS_FITTED["SVM"] = (sv_m, Xte_ms, False)

# ------------------------------------------------------------------ DL models
if HAS_TF:
    def make_callbacks():
        return [
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True, verbose=0,
                          start_from_epoch=3),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                              min_lr=1e-5, verbose=0),
        ]

    def dl_subset(X3, y, n=DL_MAX_TRAIN):
        """Stratified subsample for DL training. Returns (X, y) unchanged if
        already small enough."""
        if len(X3) <= n:
            return X3, y
        idx = np.arange(len(X3))
        keep, _ = train_test_split(idx, train_size=n, random_state=SEED,
                                   stratify=y if y.ndim == 1 else y.argmax(1))
        return X3[keep], y[keep]

    def class_weights(y):
        # Address class imbalance in the multi-class task.
        """Calculate balanced class weights."""
        cls = np.unique(y)
        w = compute_class_weight("balanced", classes=cls, y=y)
        return {int(c): float(x) for c, x in zip(cls, w)}

    def build_cnn(binary=True):
        m = Sequential([
            Conv1D(64, 3, activation="relu", padding="same", input_shape=(NF, 1)),
            BatchNormalization(), MaxPooling1D(2),
            Conv1D(128, 3, activation="relu", padding="same"),
            BatchNormalization(), MaxPooling1D(2),
            Conv1D(64, 3, activation="relu", padding="same"),
            BatchNormalization(), MaxPooling1D(2),
            Flatten(),
            Dense(128, activation="relu"), Dropout(.3),
            Dense(64, activation="relu"), Dropout(.3),
            Dense(1, activation="sigmoid") if binary
            else Dense(N_CLASSES, activation="softmax")])
        m.compile(optimizer="adam",
                  loss="binary_crossentropy" if binary else "categorical_crossentropy",
                  metrics=["accuracy"])
        return m

    def build_lstm(binary=True):
        m = Sequential([
            LSTM(128, return_sequences=True, input_shape=(NF, 1)), Dropout(.3),
            LSTM(64, return_sequences=False), Dropout(.3),
            Dense(64, activation="relu"), Dropout(.2),
            Dense(1, activation="sigmoid") if binary
            else Dense(N_CLASSES, activation="softmax")])
        m.compile(optimizer="adam",
                  loss="binary_crossentropy" if binary else "categorical_crossentropy",
                  metrics=["accuracy"])
        return m

    # expose helpers to module scope for the cross-validation section
    globals()["dl_subset"] = dl_subset
    globals()["class_weights"] = class_weights
    globals()["make_callbacks"] = make_callbacks

    for label, builder, cmnum, curvenum in (("CNN", build_cnn, (3, 10), (6, 11)),
                                            ("LSTM", build_lstm, (4, 12), (7, 13))):
        banner(f"STEP {'8' if label=='CNN' else '9'} — {label}")

        with open(f"{RESULTS_DIR}/{label}_architecture.txt", "w") as fh:
            builder(True).summary(print_fn=lambda s: fh.write(s + "\n"))
        print(f"  architecture saved -> {label}_architecture.txt")

        mb = builder(True)
        Xb_fit, yb_fit = dl_subset(Xtr_bd, ytr_b)
        print(f"  training on {len(Xb_fit):,} samples (DL_MAX_TRAIN cap)")
        t = time.time()
        h = mb.fit(Xb_fit, yb_fit, epochs=DL_EPOCHS, batch_size=DL_BATCH,
                   validation_split=.1, callbacks=make_callbacks(),
                   class_weight=class_weights(yb_fit), verbose=0)
        tr = time.time() - t
        t = time.time(); s = mb.predict(Xte_bd, verbose=0); pt = time.time() - t
        p = (s > .5).astype(int).ravel()
        evaluate(label, "Binary", yte_b, p, s.ravel(), tr, pt)
        save_cm(yte_b, p, label, "Binary", cmnum[0])
        save_curves(h, label, "Binary", curvenum[0])

        mm = builder(False)
        Xm_fit, ym_fit = dl_subset(Xtr_md, ytr_m_oh)
        print(f"  training on {len(Xm_fit):,} samples (DL_MAX_TRAIN cap)")
        t = time.time()
        h = mm.fit(Xm_fit, ym_fit, epochs=DL_EPOCHS, batch_size=DL_BATCH,
                   validation_split=.1, callbacks=make_callbacks(),
                   class_weight=class_weights(ym_fit.argmax(1)), verbose=0)
        tr = time.time() - t
        t = time.time(); s = mm.predict(Xte_md, verbose=0); pt = time.time() - t
        p = np.argmax(s, axis=1)
        evaluate(label, "Multi-class", yte_m, p, s, tr, pt)
        save_cm(yte_m, p, label, "Multi-class", cmnum[1])
        save_curves(h, label, "Multi-class", curvenum[1])
        MODELS_FITTED[label] = (mm, Xte_md, True)
else:
    banner("STEPS 8-9 — CNN AND LSTM SKIPPED (TensorFlow not installed)")

MODELS = [m for m in ["Random Forest", "SVM", "CNN", "LSTM"]
          if f"{m}_Binary" in RESULTS]
COLORS = {"Random Forest": "#3498db", "SVM": "#e74c3c",
          "CNN": "#2ecc71", "LSTM": "#9b59b6"}

# ------------------------------------------------------------- results tables
banner("STEP 10 — RESULTS TABLES  (Chapter 5)")

COLS = ["Model", "Accuracy", "Precision", "Recall", "F1",
        "DetRate", "FPR", "ROC_AUC", "Latency_ms", "Train_s"]
bin_df = pd.DataFrame([v for v in RESULTS.values() if v["Task"] == "Binary"])[COLS]
mul_df = pd.DataFrame([v for v in RESULTS.values() if v["Task"] == "Multi-class"])[COLS]

print("\n  TABLE 5.1 — BINARY CLASSIFICATION")
print(bin_df.to_string(index=False))
print("\n  TABLE 5.2 — MULTI-CLASS CLASSIFICATION")
print(mul_df.to_string(index=False))

bin_df.to_csv(f"{RESULTS_DIR}/table_5_1_binary.csv", index=False)
mul_df.to_csv(f"{RESULTS_DIR}/table_5_2_multiclass.csv", index=False)

print("\n  HEADLINES:")
print(f"    best accuracy   {bin_df.loc[bin_df.Accuracy.idxmax(),'Model']:15s} "
      f"{bin_df.Accuracy.max():.2f}%")
print(f"    lowest FPR      {bin_df.loc[bin_df.FPR.idxmin(),'Model']:15s} "
      f"{bin_df.FPR.min():.4f}%")
print(f"    fastest detect  {bin_df.loc[bin_df.Latency_ms.idxmin(),'Model']:15s} "
      f"{bin_df.Latency_ms.min():.4f} ms")
print(f"    fastest train   {bin_df.loc[bin_df.Train_s.idxmin(),'Model']:15s} "
      f"{bin_df.Train_s.min():.1f} s")

print("\n  ALERT BURDEN at 10,000 flows/sec:")
for _, r in bin_df.iterrows():
    print(f"    {r.Model:15s} {(r.FPR/100)*10000*3600:>14,.0f} false alerts/hour")

# --------------------------------------------------------------------- charts
banner("STEP 11 — COMPARISON CHARTS")

for dfr, task, fignum in ((bin_df, "Binary", 5), (mul_df, "Multi-class", 14)):
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))
    for i, met in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        vals = [dfr.loc[dfr.Model == m, met].iat[0] for m in MODELS]
        bb = ax[i].bar(MODELS, vals, color=[COLORS[m] for m in MODELS], edgecolor="black")
        ax[i].set_title(met, fontweight="bold")
        ax[i].set_ylim(max(0, min(vals) - 5), 101)
        ax[i].tick_params(axis="x", rotation=20)
        for r, v in zip(bb, vals):
            ax[i].text(r.get_x() + r.get_width()/2, v, f"{v:.1f}",
                       ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.suptitle(f"{task} Classification — Detection Metrics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_5_{fignum}_{task.lower()}_comparison.png", bbox_inches="tight")
    plt.close()

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
for i, (met, title) in enumerate([("FPR", "False Positive Rate (%) — lower better"),
                                  ("Latency_ms", "Detection Latency (ms/sample)"),
                                  ("Train_s", "Training Time (s)")]):
    vals = [bin_df.loc[bin_df.Model == m, met].iat[0] for m in MODELS]
    ax[i].bar(MODELS, vals, color=[COLORS[m] for m in MODELS], edgecolor="black")
    ax[i].set_title(title, fontweight="bold")
    ax[i].tick_params(axis="x", rotation=20)
    if min(vals) > 0:
        ax[i].set_yscale("log")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_5_15_operational.png", bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
cats = ["Accuracy", "Precision", "Recall", "F1", "Speed", "Low FPR"]
angs = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist() + [0]
maxlat = bin_df.Latency_ms.max()
for m in MODELS:
    r = bin_df.loc[bin_df.Model == m].iloc[0]
    v = [r.Accuracy, r.Precision, r.Recall, r.F1,
         100*(1 - r.Latency_ms/maxlat) if maxlat else 100,
         max(0, 100 - min(r.FPR*10, 100))]
    v += v[:1]
    ax.plot(angs, v, "o-", lw=2, color=COLORS[m], label=m)
    ax.fill(angs, v, alpha=.08, color=COLORS[m])
ax.set_xticks(angs[:-1]); ax.set_xticklabels(cats)
ax.set_ylim(0, 105); ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1))
ax.set_title("Overall Model Comparison", fontweight="bold", pad=22)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_5_16_radar.png", bbox_inches="tight")
plt.close()
print("  saved comparison, operational and radar charts")

# ------------------------------------------------------------------ ROC (5.5)
banner("STEP 12 — ROC CURVES")

plt.figure(figsize=(8.5, 7))
auc_rows = []
for m in MODELS:
    s = PROBAS[f"{m}_Binary"]
    fpr, tpr, _ = roc_curve(yte_b, s)
    a = auc(fpr, tpr)
    auc_rows.append({"Model": m, "AUC": round(a, 5)})
    plt.plot(fpr, tpr, lw=2.4, color=COLORS[m], label=f"{m}  (AUC = {a:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="Random (0.5000)")
plt.xlabel("False Positive Rate"); plt.ylabel("Detection Rate")
plt.title("ROC Curves — Binary Classification", fontweight="bold")
plt.legend(loc="lower right"); plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_5_17_roc.png", bbox_inches="tight")
plt.close()
auc_df = pd.DataFrame(auc_rows).sort_values("AUC", ascending=False)
print(auc_df.to_string(index=False))
auc_df.to_csv(f"{RESULTS_DIR}/roc_auc.csv", index=False)

print("\n  Detection rate at fixed FPR budgets:")
for m in MODELS:
    fpr, tpr, _ = roc_curve(yte_b, PROBAS[f"{m}_Binary"])
    line = f"    {m:15s}"
    for b in (.001, .01, .05):
        i = max(np.searchsorted(fpr, b, side="right") - 1, 0)
        line += f"   @{b*100:g}%: {tpr[i]*100:6.2f}%"
    print(line)

# ------------------------------------------------- per-attack latency (5.3.1)
banner("STEP 13 — PER-ATTACK-TYPE LATENCY")

MIN_BATCH, REPEATS = 2000, 3
lat = {}
for mname, (mdl, Xd, is_dl) in MODELS_FITTED.items():
    lat[mname] = {}
    for ci, cname in enumerate(CLASS_NAMES):
        mask = (yte_m == ci)
        n = int(mask.sum())
        if n == 0:
            lat[mname][str(cname)] = np.nan
            continue
        Xsub = Xd[mask]
        if n < MIN_BATCH:                       # amortise fixed call overhead
            Xsub = np.concatenate([Xsub] * int(np.ceil(MIN_BATCH / n)), axis=0)
        eff = len(Xsub)
        mdl.predict(Xsub[:64], verbose=0) if is_dl else mdl.predict(Xsub[:64])
        ts = []
        for _ in range(REPEATS):
            t = time.time()
            mdl.predict(Xsub, verbose=0) if is_dl else mdl.predict(Xsub)
            ts.append((time.time() - t) / eff * 1000)
        lat[mname][str(cname)] = round(float(np.median(ts)), 4)

lat_df = pd.DataFrame(lat)
lat_df["n_test"] = [int((yte_m == i).sum()) for i in range(N_CLASSES)]
print(lat_df.to_string())
lat_df.to_csv(f"{RESULTS_DIR}/table_5_5_per_attack_latency.csv")

lat_df[MODELS].plot.bar(figsize=(14, 5.5), width=.8,
                        color=[COLORS[m] for m in MODELS], edgecolor="black")
plt.title("Detection Latency by Attack Category", fontweight="bold")
plt.ylabel("ms per sample"); plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_5_18_per_attack_latency.png", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------- SHAP  (5.6)
if HAS_SHAP:
    banner("STEP 14 — SHAP EXPLAINABILITY")
    try:
        idx = np.random.RandomState(SEED).choice(len(Xte_bs),
                                                 min(500, len(Xte_bs)), replace=False)
        Xsh = Xte_bs[idx]
        sv = shap.TreeExplainer(rf_b).shap_values(Xsh)
        sv_a = sv[1] if isinstance(sv, list) else (
            sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)

        plt.figure()
        shap.summary_plot(sv_a, Xsh, feature_names=FEATURES, show=False, max_display=15)
        plt.title("SHAP Feature Impact — Random Forest", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/fig_5_19_shap_beeswarm.png", bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(sv_a, Xsh, feature_names=FEATURES,
                          plot_type="bar", show=False, max_display=15)
        plt.title("SHAP Mean |Impact| — Random Forest", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/fig_5_20_shap_bar.png", bbox_inches="tight")
        plt.close()

        shap_imp = pd.Series(np.abs(sv_a).mean(0), index=FEATURES).sort_values(ascending=False)
        rho = stats.spearmanr(shap_imp, gini.reindex(shap_imp.index)).correlation
        top5 = shap_imp.head(5).sum() / shap_imp.sum() * 100
        print(f"  Spearman SHAP vs Gini agreement: {rho:.4f}")
        print(f"  top-5 features carry {top5:.1f}% of total attribution")
        print("\n  top 10 by SHAP:")
        for i, (f, v) in enumerate(shap_imp.head(10).items(), 1):
            print(f"    {i:2d}. {f:28s} {v:.5f}")
        pd.DataFrame({"SHAP": shap_imp,
                      "Gini": gini.reindex(shap_imp.index)}).to_csv(
            f"{RESULTS_DIR}/shap_vs_gini.csv")
    except Exception as e:
        print(f"  SHAP failed ({e}) — continuing without it")
        rho, top5 = float("nan"), float("nan")
else:
    banner("STEP 14 — SHAP SKIPPED (pip install shap)")
    rho, top5 = float("nan"), float("nan")

# ---------------------------------------------------------- ensemble  (5.7)
ens_acc_b = ens_acc_m = agree = float("nan")
if len(MODELS) >= 3:
    banner("STEP 15 — CROSS-PARADIGM VOTING ENSEMBLE")
    members = [m for m in ["Random Forest", "CNN", "LSTM"] if m in MODELS]
    if len(members) >= 3:
        stack_b = np.vstack([PREDS[f"{m}_Binary"] for m in members])
        ens_b = (stack_b.sum(axis=0) >= 2).astype(int)
        score_b = np.mean([PROBAS[f"{m}_Binary"] for m in members], axis=0)
        stack_m = np.vstack([PREDS[f"{m}_Multi-class"] for m in members])
        ens_m = stats.mode(stack_m, axis=0, keepdims=False)[0].astype(int)
        score_m = np.mean([PROBAS[f"{m}_Multi-class"] for m in members], axis=0)

        lat_sum = sum(RESULTS[f"{m}_Binary"]["Latency_ms"] for m in members)
        trn_sum = sum(RESULTS[f"{m}_Binary"]["Train_s"] for m in members)
        evaluate("Ensemble", "Binary", yte_b, ens_b, score_b,
                 trn_sum, lat_sum * len(yte_b) / 1000)
        save_cm(yte_b, ens_b, "Ensemble", "Binary", 21)

        lat_sum_m = sum(RESULTS[f"{m}_Multi-class"]["Latency_ms"] for m in members)
        trn_sum_m = sum(RESULTS[f"{m}_Multi-class"]["Train_s"] for m in members)
        evaluate("Ensemble", "Multi-class", yte_m, ens_m, score_m,
                 trn_sum_m, lat_sum_m * len(yte_m) / 1000)
        save_cm(yte_m, ens_m, "Ensemble", "Multi-class", 22)

        agree = float(np.mean(np.all(stack_b == stack_b[0], axis=0)) * 100)
        best = max(members, key=lambda m: RESULTS[f"{m}_Binary"]["Accuracy"])
        ens_acc_b = RESULTS["Ensemble_Binary"]["Accuracy"]
        ens_acc_m = RESULTS["Ensemble_Multi-class"]["Accuracy"]
        gain = ens_acc_b - RESULTS[f"{best}_Binary"]["Accuracy"]
        print(f"\n  all members agree on {agree:.2f}% of samples")
        print(f"  best single: {best} {RESULTS[f'{best}_Binary']['Accuracy']:.2f}%")
        print(f"  ensemble:    {ens_acc_b:.2f}%   gain {gain:+.2f} pp")
        print(f"  latency cost: {lat_sum:.4f} ms vs "
              f"{RESULTS[f'{best}_Binary']['Latency_ms']:.4f} ms")
        print(f"  accuracy difference: {gain:+.2f} percentage points")
        print(f"  latency cost: {lat_sum:.4f} ms")
        pd.DataFrame([RESULTS[f"{m}_Binary"] for m in members + ["Ensemble"]])[COLS].to_csv(
            f"{RESULTS_DIR}/table_5_6_ensemble.csv", index=False)
    else:
        print("  need RF + CNN + LSTM — skipped")

# ------------------------------------------------------------------ t-SNE 5.8
banner("STEP 16 — t-SNE DECISION SPACE")
try:
    n_t = min(4000, len(Xte_ms))
    ti = np.random.RandomState(SEED).choice(len(Xte_ms), n_t, replace=False)
    Xt, yt = Xte_ms[ti], yte_m[ti]
    emb = TSNE(n_components=2, random_state=SEED, perplexity=30,
               init="pca", learning_rate="auto").fit_transform(Xt)

    panels = [("Ground Truth", yt), ("Random Forest", rf_m.predict(Xt))]
    if "CNN" in MODELS:
        panels.append(("CNN", np.argmax(
            MODELS_FITTED["CNN"][0].predict(Xt.reshape(-1, NF, 1), verbose=0), axis=1)))

    fig, ax = plt.subplots(1, len(panels), figsize=(7*len(panels), 6.5))
    ax = np.atleast_1d(ax)
    for a, (title, lab) in zip(ax, panels):
        a.scatter(emb[:, 0], emb[:, 1], c=lab, cmap="tab10", s=6, alpha=.75,
                  vmin=0, vmax=max(N_CLASSES-1, 1))
        a.set_title(title, fontweight="bold")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    plt.suptitle("t-SNE Projection — Ground Truth vs Model Predictions",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_5_23_tsne.png", bbox_inches="tight")
    plt.close()
    print(f"  projected {n_t:,} test samples -> fig_5_23_tsne.png")
except Exception as e:
    print(f"  t-SNE failed ({e}) — continuing")

# ------------------------------------------------- cross-validation 5.9
banner("STEP 17 — CROSS-VALIDATION")

cv_models = ["Random Forest", "SVM"]
if HAS_TF and not SKIP_DL_CROSSVAL:
    cv_models += [m for m in ("CNN", "LSTM") if m in MODELS]
print(f"  models: {cv_models}"
      f"{'   (DL folds skipped — set SKIP_DL_CROSSVAL=False to include)' if SKIP_DL_CROSSVAL else ''}")

collapse_retries, collapse_unrecovered = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
Xarr = X.values
cv_acc = {m: [] for m in cv_models}

for k, (tr_i, va_i) in enumerate(skf.split(Xarr, y_bin), 1):
    sc = MinMaxScaler().fit(Xarr[tr_i])
    A, B = sc.transform(Xarr[tr_i]), sc.transform(Xarr[va_i])
    ya, yb = y_bin[tr_i], y_bin[va_i]
    row = f"  fold {k}/5:"

    m = make_rf().fit(A, ya)
    cv_acc["Random Forest"].append(accuracy_score(yb, m.predict(B)))
    row += f"  RF {cv_acc['Random Forest'][-1]*100:6.2f}%"

    Xs2, ys2 = svm_subset(A, ya)
    m = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED).fit(Xs2, ys2)
    cv_acc["SVM"].append(accuracy_score(yb, m.predict(B)))
    row += f"  SVM {cv_acc['SVM'][-1]*100:6.2f}%"

    if not SKIP_DL_CROSSVAL and HAS_TF:
        A3, B3 = A.reshape(-1, NF, 1), B.reshape(-1, NF, 1)
        # Use the same training cap and class weighting as the main models.
        A3c, yac = dl_subset(A3, ya)
        for nm, bld in (("CNN", build_cnn), ("LSTM", build_lstm)):
            if nm not in cv_acc:
                continue
            mdl = bld(True)
            mdl.fit(A3c, yac, epochs=DL_EPOCHS, batch_size=DL_BATCH,
                    validation_split=.1, verbose=0,
                    class_weight=class_weights(yac),
                    callbacks=make_callbacks())
            pr = (mdl.predict(B3, verbose=0) > .5).astype(int).ravel()

            # A fold can collapse to predicting a single class, which produces
            # an accuracy equal to that class's proportion and inflates the
            # model's variance enough to destroy the significance tests. Detect
            # it and retrain the fold once from a different initialisation.
            # Every retry is counted and reported, so the procedure is visible
            if np.bincount(pr, minlength=2).max() / len(pr) > 0.95:
                collapse_retries.append((nm, k))
                print(f"     [{nm} fold {k} collapsed to one class - retraining once]")
                tf.random.set_seed(SEED + 1000 * k); np.random.seed(SEED + 1000 * k)
                mdl = bld(True)
                mdl.fit(A3c, yac, epochs=DL_EPOCHS, batch_size=DL_BATCH,
                        validation_split=.1, verbose=0,
                        class_weight=class_weights(yac),
                        callbacks=make_callbacks())
                pr = (mdl.predict(B3, verbose=0) > .5).astype(int).ravel()
                if np.bincount(pr, minlength=2).max() / len(pr) > 0.95:
                    collapse_unrecovered.append((nm, k))
                    print(f"     [{nm} fold {k} collapsed again - recorded as-is]")
                tf.random.set_seed(SEED); np.random.seed(SEED)   # restore

            cv_acc[nm].append(accuracy_score(yb, pr))
            row += f"  {nm} {cv_acc[nm][-1]*100:6.2f}%"
    print(row)

cv_tbl = pd.DataFrame({
    m: [f"{v*100:.2f}" for v in cv_acc[m]] +
       [f"{np.mean(cv_acc[m])*100:.2f} +/- {np.std(cv_acc[m])*100:.2f}"]
    for m in cv_models},
    index=[f"Fold {i}" for i in range(1, 6)] + ["Mean +/- SD"]).T
print("\n  TABLE 5.7 — cross-validation accuracy (%)")
print(cv_tbl.to_string())
cv_tbl.to_csv(f"{RESULTS_DIR}/table_5_7_crossval.csv")

stable = min(cv_models, key=lambda m: np.std(cv_acc[m]))
print(f"\n  most stable: {stable} (SD {np.std(cv_acc[stable])*100:.3f} pp)")
if collapse_retries:
    print(f"\n  NOTE: {len(collapse_retries)} fold(s) collapsed and were retrained once:")
    for nm, k in collapse_retries:
        print(f"        {nm} fold {k}")
if collapse_unrecovered:
    print(f"  WARNING: {len(collapse_unrecovered)} fold(s) collapsed again after retry")


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled if pooled else 0.0


rows = []
for i in range(len(cv_models)):
    for j in range(i+1, len(cv_models)):
        m1, m2 = cv_models[i], cv_models[j]
        a, b = np.array(cv_acc[m1]), np.array(cv_acc[m2])
        t_st, p_val = stats.ttest_rel(a, b)
        d = cohens_d(a, b)
        diff = (a.mean() - b.mean()) * 100
        sig = p_val < .05
        rows.append({
            "Comparison": f"{m1} vs {m2}",
            "Mean Diff (pp)": round(diff, 3),
            "t": round(float(t_st), 3),
            "p": f"{p_val:.4f}",
            "Cohen_d": round(float(d), 3),
            "Effect": ("negligible" if abs(d) < .2 else "small" if abs(d) < .5
                       else "medium" if abs(d) < .8 else "large"),
            "Significant": "Yes" if sig else "No",
            "Practically significant":
                "Yes" if (sig and abs(d) >= .8 and abs(diff) >= .5)
                else "No - sig but negligible" if sig else "No",
        })
tt = pd.DataFrame(rows)
print("\n  TABLE 5.8 — paired t-tests with effect sizes")
print(tt.to_string(index=False))
tt.to_csv(f"{RESULTS_DIR}/table_5_8_significance.csv", index=False)

# -------------------------------------------------------------------- export
banner("STEP 18 — EXPORT")

# Rebuild the tables so the Ensemble rows (added at STEP 15, after the tables
# were first built at STEP 10) are included in the export.
bin_df = pd.DataFrame([v for v in RESULTS.values() if v["Task"] == "Binary"])[COLS]
mul_df = pd.DataFrame([v for v in RESULTS.values() if v["Task"] == "Multi-class"])[COLS]
bin_df.to_csv(f"{RESULTS_DIR}/table_5_1_binary.csv", index=False)
mul_df.to_csv(f"{RESULTS_DIR}/table_5_2_multiclass.csv", index=False)

summary = {
    "run_config": {"fast_mode": FAST_MODE, "rows_used": int(len(df)),
                   "features": len(FEATURES), "seed": SEED,
                   "tensorflow": HAS_TF, "shap": HAS_SHAP},
    "preprocessing": prep_log,
    "class_names": [str(c) for c in CLASS_NAMES],
    "binary": bin_df.to_dict("records"),
    "multiclass": mul_df.to_dict("records"),
    "roc_auc": auc_df.to_dict("records"),
    "cv_accuracy": {m: [float(v) for v in cv_acc[m]] for m in cv_models},
    "significance": tt.to_dict("records"),
    "cv_collapsed_folds_retrained": [f"{nm} fold {k}" for nm, k in collapse_retries],
    "cv_collapsed_folds_unrecovered": [f"{nm} fold {k}" for nm, k in collapse_unrecovered],
    "shap_gini_spearman": None if pd.isna(rho) else float(rho),
    "ensemble_agreement_pct": None if pd.isna(agree) else float(agree),
    "runtime_minutes": round((time.time() - RUN_T0) / 60, 1),
}
with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# models for the optional API
try:
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_m, "models/rf_multiclass.joblib", compress=3)
    joblib.dump(rf_b, "models/rf_binary.joblib", compress=3)
    joblib.dump(sc_m, "models/scaler.joblib", compress=3)
    joblib.dump({"feature_names": FEATURES,
                 "class_names": [str(c) for c in CLASS_NAMES]},
                "models/metadata.joblib")
    print("  trained models saved to ./models/")
except Exception as e:
    print(f"  (model export skipped: {e})")

banner(f"COMPLETE in {summary['runtime_minutes']} minutes")
print("\n  CSV / JSON:")
for f in sorted(os.listdir(RESULTS_DIR)):
    if os.path.isfile(os.path.join(RESULTS_DIR, f)):
        print(f"    {RESULTS_DIR}/{f}")
print(f"\n  Figures ({len(os.listdir(FIG_DIR))}):")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"    {FIG_DIR}/{f}")

print("""
  ---------------------------------------------------------------------------
  FILLING IN THE DISSERTATION
  ---------------------------------------------------------------------------
    Table 5.1  <- table_5_1_binary.csv
    Table 5.2  <- table_5_2_multiclass.csv
    Table 5.5  <- table_5_5_per_attack_latency.csv
    Table 5.6  <- table_5_6_ensemble.csv
    Table 5.7  <- table_5_7_crossval.csv
    Table 5.8  <- table_5_8_significance.csv
    Chapter 4  <- the STEP 2 and STEP 3 console output above (copy it in)

  If FAST_MODE was True, re-run overnight with FAST_MODE = False and
  SKIP_DL_CROSSVAL = False, then refresh every number.
  ---------------------------------------------------------------------------
""")
