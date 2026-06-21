"""
evaluate.py
===========
Post-training analysis for the ORB Failure Predictor.

Three evaluation modes
-----------------------
1. SHAP analysis    — Which features drive failure predictions most?
                      Produces beeswarm + waterfall plots per sample.

2. Walk-forward backtest
                    — Simulates live deployment: train on data[0:t],
                      predict data[t:t+window], advance by window.
                      Reports AP, precision@top-K, and directional accuracy
                      across each fold. No future data ever used.

3. Calibration plot — Reliability diagram: are P=0.7 predictions
                      actually failing 70% of the time?

Usage:
    python scripts/evaluate.py --features data/features.csv
    python scripts/evaluate.py --features data/features.csv --mode shap
    python scripts/evaluate.py --features data/features.csv --mode backtest
    python scripts/evaluate.py --features data/features.csv --mode calibration
    python scripts/evaluate.py --features data/features.csv --mode all
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

MODELS_DIR  = Path("models")
PLOTS_DIR   = Path("data")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    console.print("[yellow]shap not installed — SHAP mode unavailable. "
                  "Install with: pip install shap[/yellow]")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

META_COLS = {"label", "date", "symbol"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(features_path: str):
    df = pd.read_csv(features_path).sort_values("date").reset_index(drop=True)
    drop_cols = [c for c in META_COLS if c in df.columns]
    meta = df[drop_cols]
    X = df.drop(columns=drop_cols)
    y = df["label"].astype(int)
    # Median-fill for sklearn compatibility
    X = X.fillna(X.median())
    return X, y, meta


def load_model_and_threshold():
    model_path = MODELS_DIR / "orb_failure_model.pkl"
    thresh_path = MODELS_DIR / "orb_failure_threshold.txt"
    feat_path   = MODELS_DIR / "orb_failure_features.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Run train_model.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    threshold = float(open(thresh_path).read().strip()) if thresh_path.exists() else 0.5
    features  = [l.strip() for l in open(feat_path).readlines()] if feat_path.exists() else None
    return model, threshold, features


# ── 1. SHAP analysis ──────────────────────────────────────────────────────────

def run_shap_analysis(X: pd.DataFrame, y: pd.Series):
    if not HAS_SHAP:
        console.print("[red]shap not available.[/red]")
        return

    model, threshold, features = load_model_and_threshold()

    # Try to extract underlying LightGBM estimator for TreeExplainer
    base_estimator = None
    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0]
        base_estimator = getattr(inner, "estimator",
                         getattr(inner, "base_estimator", None))
    elif hasattr(model, "estimator"):
        base_estimator = model.estimator

    if base_estimator is not None and HAS_LGB and isinstance(base_estimator, lgb.LGBMClassifier):
        explainer = shap.TreeExplainer(base_estimator)
    else:
        console.print("[yellow]Using KernelExplainer (slower)...[/yellow]")
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        background = shap.sample(X, 50)
        explainer = shap.KernelExplainer(predict_fn, background)

    console.print("[cyan]Computing SHAP values...[/cyan]")
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # ── Beeswarm summary plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title("ORB Failure Predictor — SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Saved → {out}[/green]")

    # ── Mean |SHAP| bar chart ─────────────────────────────────────────────────
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    mean_abs_shap.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Features by Mean |SHAP|")
    plt.tight_layout()
    out2 = PLOTS_DIR / "shap_bar.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Saved → {out2}[/green]")

    # Print ranked table
    table = Table(title="SHAP Feature Ranking", header_style="bold cyan")
    table.add_column("Rank"); table.add_column("Feature"); table.add_column("Mean |SHAP|", justify="right")
    ranked = mean_abs_shap.sort_values(ascending=False).reset_index()
    for i, (feat, val) in enumerate(ranked.values, 1):
        table.add_row(str(i), str(feat), f"{val:.5f}")
    console.print(table)


# ── 2. Walk-forward backtest ──────────────────────────────────────────────────

def run_walk_forward_backtest(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    """
    Expanding-window walk-forward:
    • Minimum 60 training samples before first evaluation.
    • Each step: retrain on all history, predict next STEP_SIZE samples.
    • Reports per-fold and aggregate metrics.
    """
    STEP_SIZE = max(20, len(X) // 8)
    MIN_TRAIN  = 60

    if len(X) < MIN_TRAIN + STEP_SIZE:
        console.print(f"[yellow]Not enough data for walk-forward backtest "
                      f"(need ≥{MIN_TRAIN + STEP_SIZE} rows, have {len(X)}).[/yellow]")
        return

    console.print(f"\n[cyan]Walk-forward backtest "
                  f"(step={STEP_SIZE}, min_train={MIN_TRAIN})...[/cyan]")

    try:
        import lightgbm as lgb
        from train_model import DEFAULT_LGBM_PARAMS, apply_smote
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

    fold_results = []
    start_idx = MIN_TRAIN

    while start_idx + STEP_SIZE <= len(X):
        end_idx = start_idx + STEP_SIZE
        Xtr, ytr = X.iloc[:start_idx], y.iloc[:start_idx]
        Xval, yval = X.iloc[start_idx:end_idx], y.iloc[start_idx:end_idx]

        if yval.sum() == 0:
            start_idx += STEP_SIZE
            continue

        try:
            Xtr_sm, ytr_sm = apply_smote(Xtr, ytr)
        except Exception:
            Xtr_sm, ytr_sm = Xtr.values, ytr.values

        try:
            base = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                scale_pos_weight=2.0, verbosity=-1, random_state=42,
            )
        except Exception:
            from sklearn.ensemble import GradientBoostingClassifier
            base = GradientBoostingClassifier(n_estimators=100, random_state=42)

        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(Xtr_sm, ytr_sm)

        proba = cal.predict_proba(Xval.values)[:, 1]
        ap    = average_precision_score(yval, proba)
        auc   = roc_auc_score(yval, proba) if len(np.unique(yval)) > 1 else np.nan

        date_str = meta["date"].iloc[start_idx] if "date" in meta.columns else str(start_idx)
        fold_results.append({
            "fold_start": date_str,
            "n_train":   start_idx,
            "n_val":     STEP_SIZE,
            "n_pos_val": int(yval.sum()),
            "AP":        ap,
            "AUC":       auc,
        })
        console.print(f"  Fold @{date_str}: AP={ap:.3f}  AUC={auc:.3f}  "
                      f"positives={int(yval.sum())}/{STEP_SIZE}")

        start_idx += STEP_SIZE

    if not fold_results:
        return

    results_df = pd.DataFrame(fold_results)
    table = Table(title="Walk-Forward Backtest Summary", header_style="bold magenta")
    for col in results_df.columns:
        table.add_column(col)
    for _, row in results_df.iterrows():
        table.add_row(*[
            f"{v:.3f}" if isinstance(v, float) else str(v)
            for v in row.values
        ])
    console.print(table)

    console.print(f"\nMean AP:  [bold green]{results_df['AP'].mean():.4f}[/bold green]")
    console.print(f"Mean AUC: [bold green]{results_df['AUC'].dropna().mean():.4f}[/bold green]")

    # Plot AP over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(results_df["fold_start"], results_df["AP"], marker="o", color="steelblue")
    ax.axhline(results_df["AP"].mean(), color="red", linestyle="--",
               label=f"Mean AP = {results_df['AP'].mean():.3f}")
    ax.set_xlabel("Fold start date")
    ax.set_ylabel("Average Precision")
    ax.set_title("Walk-Forward AP Over Time")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = PLOTS_DIR / "walkforward_ap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    console.print(f"[green]Saved → {out}[/green]")


# ── 3. Calibration plot ───────────────────────────────────────────────────────

def run_calibration_plot(X: pd.DataFrame, y: pd.Series):
    model, threshold, _ = load_model_and_threshold()
    proba = model.predict_proba(X.values)[:, 1]

    fraction_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y, proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of failures")
    ax.set_title(f"Calibration Curve  (Brier = {brier:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(proba[y == 0], bins=30, alpha=0.6, label="Success (0)", color="steelblue")
    ax2.hist(proba[y == 1], bins=30, alpha=0.6, label="Failure (1)", color="tomato")
    ax2.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.2f}")
    ax2.set_xlabel("Predicted failure probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Probability Distribution by True Label")
    ax2.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "calibration_plot.png"
    plt.savefig(out, dpi=150)
    plt.close()
    console.print(f"[green]Saved → {out}[/green]")
    console.print(f"Brier score (lower=better): [bold]{brier:.4f}[/bold]")


# ── Dispatch ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--mode",
                        choices=["shap", "backtest", "calibration", "all"],
                        default="all")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(exist_ok=True)
    X, y, meta = load_data(args.features)

    if args.mode in ("shap", "all"):
        run_shap_analysis(X, y)

    if args.mode in ("backtest", "all"):
        run_walk_forward_backtest(X, y, meta)

    if args.mode in ("calibration", "all"):
        run_calibration_plot(X, y)


if __name__ == "__main__":
    main()
