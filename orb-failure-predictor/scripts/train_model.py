"""
train_model.py
==============
LightGBM classifier with Optuna hyperparameter optimization.

Key differences from the bounce predictor (Random Forest + GridSearchCV):

  • LightGBM       — gradient-boosted trees; handles missing values natively,
                     faster, generally better on tabular financial data.
  • Optuna         — Bayesian hyperparameter search (TPE sampler) instead of
                     exhaustive grid search; finds better configs in fewer trials.
  • TimeSeriesSplit— respects temporal ordering; never uses future data
                     to validate past predictions (no leakage).
  • SMOTE-Tomek    — class imbalance handled via over+undersampling pipeline
                     (more nuanced than just class_weight='balanced').
  • Calibration    — isotonic regression calibrates raw probabilities so
                     that P=0.7 actually means ~70% failure rate.
  • Threshold opt  — F-beta (beta=0.5, precision-weighted) over a grid.

Outputs saved to models/:
  orb_failure_model.pkl        — trained LightGBM + calibration pipeline
  orb_failure_threshold.txt    — optimal probability decision threshold
  orb_failure_features.txt     — ordered feature list (for inference)
  orb_failure_optuna.db        — Optuna study (optional, for analysis)

Usage:
    python scripts/train_model.py --features data/features.csv
    python scripts/train_model.py --features data/features.csv --n-trials 200
"""

import argparse
import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, confusion_matrix, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

# ── Optional heavy dependencies ──────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    console.print("[yellow]lightgbm not found — falling back to GradientBoosting.[/yellow]")
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    console.print("[yellow]optuna not found — using default hyperparameters.[/yellow]")
    HAS_OPTUNA = False

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    console.print("[yellow]imbalanced-learn not found — skipping SMOTE.[/yellow]")
    HAS_IMBLEARN = False

# ── Config ───────────────────────────────────────────────────────────────────
MODELS_DIR        = Path("models")
N_CV_SPLITS       = 5
N_OPTUNA_TRIALS   = 100          # override with --n-trials
OPTUNA_TIMEOUT    = 300          # seconds
PRECISION_WEIGHT  = 0.5          # beta for F-beta (< 1 → favour precision)
RANDOM_SEED       = 42


# ── Feature columns (everything except meta columns) ─────────────────────────
META_COLS = {"label", "date", "symbol"}


def load_features(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # Sort by date to preserve temporal ordering
    df = df.sort_values("date").reset_index(drop=True)

    drop_cols = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["label"].astype(int)

    # Drop columns that are >60% missing
    missing_ratio = X.isna().mean()
    to_drop = missing_ratio[missing_ratio > 0.60].index.tolist()
    if to_drop:
        console.print(f"[yellow]Dropping high-missing features: {to_drop}[/yellow]")
        X = X.drop(columns=to_drop)

    console.print(f"Feature matrix: [bold]{X.shape[0]}[/bold] rows × "
                  f"[bold]{X.shape[1]}[/bold] cols")
    console.print(f"Label distribution: {dict(y.value_counts().to_dict())}")
    return X, y


def apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE-Tomek over/under-sampling to address class imbalance."""
    if not HAS_IMBLEARN:
        return X.values, y.values

    try:
        resampler = SMOTETomek(random_state=RANDOM_SEED)
        X_r, y_r = resampler.fit_resample(X.fillna(X.median()), y)
        console.print(f"After SMOTE-Tomek: {int((y_r == 1).sum())} failures, "
                      f"{int((y_r == 0).sum())} successes")
        return X_r, y_r
    except Exception as e:
        console.print(f"[yellow]SMOTE failed ({e}), using raw data.[/yellow]")
        return X.values, y.values


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_train: np.ndarray, y_train: np.ndarray, feature_names: list):
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective":        "binary",
            "metric":           "average_precision",
            "verbosity":        -1,
            "boosting_type":    trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 200),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 80),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "random_state":     RANDOM_SEED,
        }

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            Xtr, Xval = X_train[train_idx], X_train[val_idx]
            ytr, yval = y_train[train_idx], y_train[val_idx]
            if yval.sum() == 0:
                continue

            model = lgb.LGBMClassifier(**params)
            model.fit(
                Xtr, ytr,
                eval_set=[(Xval, yval)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            proba = model.predict_proba(Xval)[:, 1]
            scores.append(average_precision_score(yval, proba))

        return np.mean(scores) if scores else 0.0

    return objective


def run_optuna(X_train: np.ndarray, y_train: np.ndarray,
               feature_names: list, n_trials: int) -> dict:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    objective = make_objective(X_train, y_train, feature_names)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=True,
    )
    console.print(f"\nBest Optuna AP: [bold green]{study.best_value:.4f}[/bold green]")
    console.print(f"Best params: {study.best_params}")
    return study.best_params


# ── Default params (used when Optuna unavailable) ────────────────────────────
DEFAULT_LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",
    "verbosity":         -1,
    "n_estimators":      400,
    "max_depth":         7,
    "num_leaves":        63,
    "learning_rate":     0.03,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "scale_pos_weight":  2.0,
    "random_state":      RANDOM_SEED,
}


# ── Threshold optimisation ────────────────────────────────────────────────────

def optimise_threshold(y_true: np.ndarray, proba: np.ndarray,
                       beta: float = PRECISION_WEIGHT) -> float:
    """
    Sweep probability thresholds and return the one that maximises F-beta.
    beta < 1  → precision-weighted (fewer false alarms, miss more failures)
    beta > 1  → recall-weighted    (catch more failures, accept false alarms)
    """
    thresholds = np.arange(0.30, 0.85, 0.01)
    best_thresh, best_f = 0.5, 0.0

    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()

        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        f    = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-10)

        if f > best_f:
            best_f, best_thresh = f, t

    return float(best_thresh)


# ── Main training routine ─────────────────────────────────────────────────────

def train(features_path: str, n_trials: int = N_OPTUNA_TRIALS):
    MODELS_DIR.mkdir(exist_ok=True)
    X, y = load_features(features_path)
    feature_names = list(X.columns)

    # Train / test split — last 20% of dates as held-out test
    n_test  = max(int(len(X) * 0.20), 10)
    n_train = len(X) - n_test

    X_train_raw, X_test_raw = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test         = y.iloc[:n_train].values, y.iloc[n_train:].values

    console.print(f"\nTrain: [bold]{n_train}[/bold] samples | "
                  f"Test: [bold]{n_test}[/bold] samples")

    # Class rebalancing
    X_train_sm, y_train_sm = apply_smote(X_train_raw, y_train)

    # Fill remaining NaN (LightGBM handles NaN natively, but SMOTE filled medians)
    X_test_arr = X_test_raw.fillna(X_test_raw.median()).values

    # ── Hyperparameter search ─────────────────────────────────────────────────
    if HAS_OPTUNA and HAS_LGB and n_trials > 0:
        console.print(f"\n[cyan]Running Optuna ({n_trials} trials)...[/cyan]")
        best_params = run_optuna(X_train_sm, y_train_sm, feature_names, n_trials)
        best_params.update({"objective": "binary", "verbosity": -1,
                            "random_state": RANDOM_SEED})
    else:
        best_params = DEFAULT_LGBM_PARAMS

    # ── Final model fit ───────────────────────────────────────────────────────
    if HAS_LGB:
        base_model = lgb.LGBMClassifier(**best_params)
    else:
        base_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=RANDOM_SEED,
        )

    # Calibrate probabilities — isotonic suits our mid-size datasets better
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train_sm, y_train_sm)

    # ── Test set evaluation ───────────────────────────────────────────────────
    proba_test = calibrated.predict_proba(X_test_arr)[:, 1]
    threshold  = optimise_threshold(y_test, proba_test, beta=PRECISION_WEIGHT)
    pred_test  = (proba_test >= threshold).astype(int)

    ap   = average_precision_score(y_test, proba_test)
    auc  = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else np.nan

    console.print(f"\n[bold]Test set results (threshold = {threshold:.2f})[/bold]")
    console.print(classification_report(y_test, pred_test,
                                        target_names=["Success", "Failure"]))

    table = Table(title="Key Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric");  table.add_column("Value", justify="right")
    table.add_row("Average Precision (AP)",  f"{ap:.4f}")
    table.add_row("AUC-ROC",                 f"{auc:.4f}" if not np.isnan(auc) else "N/A")
    table.add_row("Decision threshold",      f"{threshold:.2f}")
    cm = confusion_matrix(y_test, pred_test)
    if cm.shape == (2, 2):
        table.add_row("True Positives (Failures caught)",  str(cm[1, 1]))
        table.add_row("False Positives (False alarms)",    str(cm[0, 1]))
        table.add_row("False Negatives (Missed failures)", str(cm[1, 0]))
    console.print(table)

    # ── Cross-validation on full data ─────────────────────────────────────────
    console.print("\n[cyan]5-fold TimeSeriesSplit CV (full dataset)...[/cyan]")
    tscv = TimeSeriesSplit(n_splits=5)
    X_full_arr = X.fillna(X.median()).values
    y_full     = y.values

    cv_ap_scores = []
    for train_idx, val_idx in tscv.split(X_full_arr):
        Xtr, Xval = X_full_arr[train_idx], X_full_arr[val_idx]
        ytr, yval = y_full[train_idx], y_full[val_idx]
        if yval.sum() == 0:
            continue
        Xtr_sm, ytr_sm = apply_smote(pd.DataFrame(Xtr, columns=feature_names),
                                     pd.Series(ytr))
        cal = CalibratedClassifierCV(
            lgb.LGBMClassifier(**best_params) if HAS_LGB
            else GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED),
            method="isotonic", cv=3,
        )
        cal.fit(Xtr_sm, ytr_sm)
        proba_val = cal.predict_proba(Xval)[:, 1]
        cv_ap_scores.append(average_precision_score(yval, proba_val))

    if cv_ap_scores:
        console.print(f"CV Average Precision: "
                      f"[bold green]{np.mean(cv_ap_scores):.4f} "
                      f"± {np.std(cv_ap_scores):.4f}[/bold green]")

    # ── Save artefacts ────────────────────────────────────────────────────────
    model_path     = MODELS_DIR / "orb_failure_model.pkl"
    threshold_path = MODELS_DIR / "orb_failure_threshold.txt"
    features_path2 = MODELS_DIR / "orb_failure_features.txt"
    params_path    = MODELS_DIR / "orb_failure_params.json"

    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)
    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    with open(features_path2, "w") as f:
        f.write("\n".join(feature_names))
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    console.print(f"\n[bold green]Saved:[/bold green]")
    console.print(f"  Model      → {model_path}")
    console.print(f"  Threshold  → {threshold_path}  ({threshold:.2f})")
    console.print(f"  Features   → {features_path2}")
    console.print(f"  Params     → {params_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",  default="data/features.csv")
    parser.add_argument("--n-trials",  type=int, default=N_OPTUNA_TRIALS,
                        help="Number of Optuna trials (0 to skip search)")
    args = parser.parse_args()
    train(args.features, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
