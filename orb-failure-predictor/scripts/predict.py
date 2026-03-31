"""
predict.py
==========
Real-time ORB Failure inference.

Given today's minute bars (fed in live or loaded from a CSV), this script:

  1. Watches for the Opening Range to form (first 30 min).
  2. Monitors for the first clean breakout above/below the OR.
  3. At breakout detection, extracts the same 20-feature vector used in training.
  4. Loads the saved model and outputs a calibrated failure probability.
  5. Issues a directional recommendation.

Two modes
---------
  live   — poll a CSV that your data feed appends to (e.g. Schwab streaming)
  batch  — score all breakout events in a completed minute-data file

Usage:
    # Live mode (polls every 60 seconds)
    python scripts/predict.py live --symbol NVDA \
        --data-file data/NVDA_minute_data.csv

    # Batch mode (score a historical file, print all events)
    python scripts/predict.py batch --symbol NVDA \
        --data-file data/NVDA_minute_data.csv

    # Single date in batch mode
    python scripts/predict.py batch --symbol NVDA \
        --data-file data/NVDA_minute_data.csv \
        --date 2025-10-30
"""

import argparse
import pickle
import time
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

MODELS_DIR = Path("models")
OR_END     = "09:59"
MARKET_OPEN = "09:30"
MIN_BREAKOUT_PCT = 0.15   # must match orb_detection.py
MIN_OR_WIDTH_PCT = 0.30


# ── Model loading ─────────────────────────────────────────────────────────────

def load_artefacts():
    model_path = MODELS_DIR / "orb_failure_model.pkl"
    thresh_path = MODELS_DIR / "orb_failure_threshold.txt"
    feat_path   = MODELS_DIR / "orb_failure_features.txt"

    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run train_model.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    threshold = (
        float(open(thresh_path).read().strip())
        if thresh_path.exists() else 0.5
    )
    features = (
        [l.strip() for l in open(feat_path).readlines()]
        if feat_path.exists() else None
    )
    return model, threshold, features


# ── Shared helpers (mirror feature_engineering.py) ───────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]  = df["typical"] * df["volume"]
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"]    = df.groupby("date")["volume"].cumsum()
    return df["cum_tp_vol"] / df["cum_vol"]


def compute_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def compute_atr(high, low, close, n=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def load_minute_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    df = df[(df["time"] >= MARKET_OPEN) & (df["time"] < "16:00")].copy()
    return df.reset_index(drop=True)


# ── Single-event feature extraction ──────────────────────────────────────────

def build_live_feature_vector(
    day_df: pd.DataFrame,
    prev_day_df,
    full_history: pd.DataFrame,
    or_high: float,
    or_low: float,
    or_open: float,
    direction: int,
    breakout_time: str,
    penetration_pct: float,
) -> dict:
    """
    Identical feature logic to feature_engineering.py — kept self-contained
    so predict.py has zero imports from other project scripts.
    """
    or_bars = day_df[day_df["time"] <= OR_END].copy()
    hist    = day_df[day_df["time"] <= breakout_time].copy()

    feat = {}
    feat["or_width_pct"]   = (or_high - or_low) / or_open * 100
    feat["penetration_pct"] = penetration_pct
    feat["breakout_direction"] = direction

    h, m = map(int, breakout_time.split(":"))
    feat["breakout_time_min"] = (h - 9) * 60 + m - 30

    if prev_day_df is not None and len(prev_day_df) > 0:
        prev_close = prev_day_df["close"].iloc[-1]
        feat["gap_pct"]               = (or_open - prev_close) / prev_close * 100
        feat["or_high_vs_prev_close"] = (or_high  - prev_close) / prev_close * 100
        feat["or_low_vs_prev_close"]  = (or_low   - prev_close) / prev_close * 100
        feat["prev_day_return"]       = (
            prev_day_df["close"].iloc[-1] - prev_day_df["open"].iloc[0]
        ) / prev_day_df["open"].iloc[0] * 100
        feat["prev_day_range_pct"] = (
            (prev_day_df["high"].max() - prev_day_df["low"].min()) / prev_close * 100
        )
    else:
        for k in ["gap_pct", "or_high_vs_prev_close", "or_low_vs_prev_close",
                  "prev_day_return", "prev_day_range_pct"]:
            feat[k] = np.nan

    feat["or_mid_vs_open_pct"] = ((or_high + or_low) / 2 - or_open) / or_open * 100

    bt_bar = day_df[day_df["time"] == breakout_time]
    if len(bt_bar) > 0 and or_bars["volume"].mean() > 0:
        feat["breakout_volume_ratio"] = (
            bt_bar["volume"].iloc[0] / (or_bars["volume"].mean() + 1e-6)
        )
        feat["breakout_bar_range_pct"] = (
            (bt_bar["high"].iloc[0] - bt_bar["low"].iloc[0]) / or_open * 100
        )
    else:
        feat["breakout_volume_ratio"]  = np.nan
        feat["breakout_bar_range_pct"] = np.nan

    if direction == 1:
        touches = (or_bars["high"] >= or_high * 0.999).sum()
    else:
        touches = (or_bars["low"] <= or_low * 1.001).sum()
    feat["or_boundary_touches"] = int(touches)

    vwap  = compute_vwap(hist).iloc[-1]
    bt_cl = hist["close"].iloc[-1]
    feat["vwap_distance_pct"] = (bt_cl - vwap) / vwap * 100
    feat["price_above_vwap"]  = int(bt_cl > vwap)

    rsi = compute_rsi(hist["close"], 14)
    feat["rsi_at_breakout"] = float(rsi.iloc[-1]) if not rsi.isna().all() else np.nan

    atr = compute_atr(hist["high"], hist["low"], hist["close"], 14)
    feat["atr_at_breakout"] = float(atr.iloc[-1]) if not atr.isna().all() else np.nan

    for k in ["macd_hist_at_breakout", "stoch_k_at_breakout", "adx_at_breakout"]:
        feat[k] = np.nan   # omitted in live mode; model handles NaN natively

    all_atr = compute_atr(
        full_history["high"], full_history["low"], full_history["close"], 20
    )
    hist_mask = full_history["datetime"] <= hist["datetime"].iloc[-1]
    hist_atr  = all_atr[hist_mask].dropna()
    if len(hist_atr) >= 20 and feat.get("atr_at_breakout") is not None:
        feat["atr_ratio"] = feat["atr_at_breakout"] / (hist_atr.iloc[-20:].mean() + 1e-6)
    else:
        feat["atr_ratio"] = np.nan

    ev_date = hist["date"].iloc[0]
    prev_full = full_history[full_history["date"] < ev_date]
    daily_closes = prev_full.groupby("date")["close"].last()
    if len(daily_closes) >= 5:
        feat["realized_vol_5d"] = float(
            daily_closes.pct_change().dropna().iloc[-5:].std() * 100
        )
    else:
        feat["realized_vol_5d"] = np.nan

    if not np.isnan(feat.get("gap_pct", np.nan)):
        gap_dir = 1 if feat["gap_pct"] > 0 else (-1 if feat["gap_pct"] < 0 else 0)
        feat["gap_direction_alignment"] = int(gap_dir == direction)
    else:
        feat["gap_direction_alignment"] = np.nan

    avg_or_vol  = or_bars["volume"].mean()
    prev_vol_df = full_history[full_history["date"] < ev_date]
    daily_vol   = prev_vol_df.groupby("date")["volume"].sum()
    if len(daily_vol) >= 5:
        feat["or_formation_volume_ratio"] = (
            avg_or_vol * 30 / (daily_vol.iloc[-5:].mean() / 390 * 30 + 1e-6)
        )
    else:
        feat["or_formation_volume_ratio"] = np.nan

    feat["avg_bar_range_during_or"] = (
        (or_bars["high"] - or_bars["low"]).mean() / or_open * 100
        if len(or_bars) > 0 else np.nan
    )

    total_range = or_high - or_low
    if total_range > 0:
        or_last_close = or_bars["close"].iloc[-1]
        feat["or_close_position"] = (or_last_close - or_low) / total_range
    else:
        feat["or_close_position"] = 0.5

    return feat


# ── Prediction output ─────────────────────────────────────────────────────────

def print_prediction(
    symbol: str,
    event_date: str,
    breakout_time: str,
    direction: int,
    breakout_price: float,
    or_high: float,
    or_low: float,
    prob_failure: float,
    threshold: float,
    feat: dict,
):
    signal    = prob_failure >= threshold
    dir_label = "UP" if direction == 1 else "DOWN"
    action    = "FADE (expect failure)" if signal else "FOLLOW (expect continuation)"
    color     = "bold red" if signal else "bold green"

    panel_text = (
        f"Symbol: [bold]{symbol}[/bold]  |  Date: {event_date}\n"
        f"Breakout: [bold]{dir_label}[/bold] @ {breakout_time}  |  "
        f"Price: {breakout_price:.2f}\n"
        f"OR Range: {or_low:.2f} – {or_high:.2f}\n\n"
        f"Failure probability: [{color}]{prob_failure:.1%}[/{color}]\n"
        f"Threshold:           {threshold:.2f}\n\n"
        f"→ [{color}]{action}[/{color}]"
    )
    console.print(Panel(panel_text, title="ORB Failure Predictor", border_style="cyan"))

    table = Table(title="Key Features at Breakout", show_header=True)
    table.add_column("Feature"); table.add_column("Value", justify="right")
    show = [
        "rsi_at_breakout", "vwap_distance_pct", "breakout_volume_ratio",
        "or_width_pct", "penetration_pct", "breakout_time_min",
        "gap_pct", "atr_ratio", "realized_vol_5d",
    ]
    for k in show:
        v = feat.get(k, np.nan)
        table.add_row(k, f"{v:.3f}" if not np.isnan(float(v)) else "N/A")
    console.print(table)


# ── Score a single day ────────────────────────────────────────────────────────

def score_day(
    day_df: pd.DataFrame,
    prev_day_df,
    full_history: pd.DataFrame,
    symbol: str,
    model,
    threshold: float,
    feature_names: list,
) -> bool:
    """Detect OR + breakout for one day and score it. Returns True if fired."""
    or_bars = day_df[day_df["time"] <= OR_END]
    post_or = day_df[day_df["time"] > OR_END]

    if len(or_bars) < 20 or len(post_or) < 5:
        return False

    or_high  = or_bars["high"].max()
    or_low   = or_bars["low"].min()
    or_open  = or_bars.iloc[0]["open"]
    or_width = (or_high - or_low) / or_open * 100

    if or_width < MIN_OR_WIDTH_PCT:
        return False

    # Scan for first breakout bar
    for _, row in post_or.iterrows():
        if row["close"] > or_high * (1 + MIN_BREAKOUT_PCT / 100):
            direction, bt_time, bt_price = 1, row["time"], row["close"]
            pen = (bt_price - or_high) / or_high * 100
            break
        if row["close"] < or_low * (1 - MIN_BREAKOUT_PCT / 100):
            direction, bt_time, bt_price = -1, row["time"], row["close"]
            pen = (or_low - bt_price) / or_low * 100
            break
    else:
        return False   # no breakout today

    feat = build_live_feature_vector(
        day_df, prev_day_df, full_history,
        or_high, or_low, or_open,
        direction, bt_time, pen,
    )

    # Align to model's expected feature order
    if feature_names:
        row_vals = [feat.get(f, np.nan) for f in feature_names]
        X = np.array(row_vals, dtype=float).reshape(1, -1)
        X = np.where(np.isnan(X), 0.0, X)   # LightGBM ok with NaN, but be safe
    else:
        X = np.array(list(feat.values()), dtype=float).reshape(1, -1)

    prob_failure = model.predict_proba(X)[0, 1]
    ev_date = str(day_df["date"].iloc[0])

    print_prediction(
        symbol, ev_date, bt_time,
        direction, bt_price, or_high, or_low,
        prob_failure, threshold, feat,
    )
    return True


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(symbol: str, data_file: str, filter_date: str = None):
    model, threshold, feature_names = load_artefacts()
    full_df = load_minute_data(data_file)
    dates   = sorted(full_df["date"].unique())
    fired   = 0

    for i, d in enumerate(dates):
        if filter_date and str(d) != filter_date:
            continue
        day_df     = full_df[full_df["date"] == d].copy()
        prev_day_df = (
            full_df[full_df["date"] == dates[i - 1]].copy() if i > 0 else None
        )
        if score_day(day_df, prev_day_df, full_df, symbol,
                     model, threshold, feature_names):
            fired += 1

    console.print(f"\n[bold]Scored {fired} ORB events across {len(dates)} trading days.[/bold]")


# ── Live mode ─────────────────────────────────────────────────────────────────

def run_live(symbol: str, data_file: str, poll_interval: int = 60):
    """
    Polls the data file every `poll_interval` seconds.
    Fires once per day when the OR completes and a breakout is detected.
    """
    model, threshold, feature_names = load_artefacts()
    console.print(f"[cyan]Live mode — watching {data_file} for {symbol} "
                  f"(polling every {poll_interval}s)...[/cyan]")

    scored_dates = set()

    while True:
        try:
            full_df = load_minute_data(data_file)
            today   = date.today()
            now_str = datetime.now().strftime("%H:%M")

            # Only attempt scoring after OR window closes
            if now_str < "10:00":
                console.print(f"[dim]Waiting for OR to complete ({now_str})...[/dim]")
                time.sleep(poll_interval)
                continue

            if today in scored_dates:
                time.sleep(poll_interval)
                continue

            today_df = full_df[full_df["date"] == today]
            if len(today_df) < 30:
                time.sleep(poll_interval)
                continue

            dates_so_far = sorted(full_df["date"].unique())
            idx = dates_so_far.index(today) if today in dates_so_far else -1
            prev_day_df = (
                full_df[full_df["date"] == dates_so_far[idx - 1]]
                if idx > 0 else None
            )

            fired = score_day(
                today_df, prev_day_df, full_df, symbol,
                model, threshold, feature_names,
            )
            if fired:
                scored_dates.add(today)

        except KeyboardInterrupt:
            console.print("\n[yellow]Live mode stopped.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        time.sleep(poll_interval)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    batch_p = sub.add_parser("batch")
    batch_p.add_argument("--symbol",    required=True)
    batch_p.add_argument("--data-file", required=True)
    batch_p.add_argument("--date",      default=None, help="Filter to a single date YYYY-MM-DD")

    live_p = sub.add_parser("live")
    live_p.add_argument("--symbol",    required=True)
    live_p.add_argument("--data-file", required=True)
    live_p.add_argument("--poll",      type=int, default=60, help="Polling interval in seconds")

    args = parser.parse_args()

    if args.mode == "batch":
        run_batch(args.symbol, args.data_file, args.date)
    else:
        run_live(args.symbol, args.data_file, args.poll)


if __name__ == "__main__":
    main()
