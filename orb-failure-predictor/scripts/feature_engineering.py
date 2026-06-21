"""
feature_engineering.py
=======================
Feature matrix construction for the ORB Failure Predictor.

For each detected ORB event, features are computed STRICTLY from data
available at the moment the breakout bar closes — zero lookahead.

Feature groups
--------------
A. Opening Range geometry
   or_width_pct, or_high_vs_prev_close, or_low_vs_prev_close,
   or_range_vs_atr20, or_close_position

B. Breakout characteristics
   breakout_direction, penetration_pct, breakout_time_min,
   breakout_volume_ratio, breakout_bar_range_pct

C. VWAP context
   vwap_distance_pct, price_above_vwap

D. Momentum / oscillators (pandas-ta)
   rsi_at_breakout, macd_hist_at_breakout, stoch_k_at_breakout,
   adx_at_breakout, cci_at_breakout

E. Volatility context
   atr_ratio (current / 20-day), bb_pct_b, realized_vol_5d

F. Gap & prior day context
   gap_pct, gap_direction_alignment, prev_day_return,
   prev_day_range_pct, or_vs_premarket_high (if available)

G. Market microstructure (derived from OHLCV)
   or_formation_volume_ratio, avg_bar_range_during_or,
   high_low_ratio_or

Output: data/features.csv
  Columns: all features above + label, date, symbol (for reference)

Usage:
    python scripts/feature_engineering.py \
        --events data/orb_events.csv \
        --data-dir data/ \
        --output data/features.csv
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("[WARN] pandas-ta not installed — oscillator features will be NaN. "
          "Install with: pip install pandas-ta")

MARKET_OPEN = "09:30"
OR_END      = "09:59"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_minute_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    df = df[(df["time"] >= MARKET_OPEN) & (df["time"] < "16:00")].copy()
    return df.reset_index(drop=True)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday cumulative VWAP reset each day."""
    df = df.copy()
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]  = df["typical"] * df["volume"]
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"]    = df.groupby("date")["volume"].cumsum()
    return df["cum_tp_vol"] / df["cum_vol"]


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def compute_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


# ── Per-event feature extraction ─────────────────────────────────────────────

def extract_features_for_event(
    event: pd.Series,
    day_df: pd.DataFrame,
    prev_day_df: pd.DataFrame | None,
    symbol_df: pd.DataFrame,
) -> dict:
    """
    Returns a flat feature dict for one ORB event.
    All lookbacks use only data up to and including the breakout bar.
    """
    date         = pd.to_datetime(event["date"]).date()
    direction    = int(event["direction"])
    breakout_time = event["breakout_time"]
    or_high      = event["or_high"]
    or_low       = event["or_low"]
    or_open      = event["or_open"]
    or_width_pct = event["or_width_pct"]

    # Subset: data up to (and including) the breakout bar
    hist = day_df[day_df["time"] <= breakout_time].copy()
    or_bars = day_df[day_df["time"] <= OR_END].copy()

    # Need at least 30 bars for meaningful features
    if len(hist) < 15:
        return {}

    # ── A. Opening Range geometry ────────────────────────────────────────────
    feat = {}
    feat["or_width_pct"] = or_width_pct

    if prev_day_df is not None and len(prev_day_df) > 0:
        prev_close = prev_day_df["close"].iloc[-1]
        feat["gap_pct"]               = (or_open - prev_close) / prev_close * 100
        feat["or_high_vs_prev_close"] = (or_high  - prev_close) / prev_close * 100
        feat["or_low_vs_prev_close"]  = (or_low   - prev_close) / prev_close * 100
        feat["prev_day_return"]       = (
            prev_day_df["close"].iloc[-1] - prev_day_df["open"].iloc[0]
        ) / prev_day_df["open"].iloc[0] * 100
        prev_day_range = (prev_day_df["high"].max() - prev_day_df["low"].min())
        feat["prev_day_range_pct"]    = prev_day_range / prev_close * 100
    else:
        feat["gap_pct"]               = np.nan
        feat["or_high_vs_prev_close"] = np.nan
        feat["or_low_vs_prev_close"]  = np.nan
        feat["prev_day_return"]       = np.nan
        feat["prev_day_range_pct"]    = np.nan

    # OR close position: where is OR mid vs full day open
    feat["or_mid_vs_open_pct"] = ((or_high + or_low) / 2 - or_open) / or_open * 100

    # ── B. Breakout characteristics ──────────────────────────────────────────
    feat["breakout_direction"] = direction
    feat["penetration_pct"]    = event["penetration_pct"]

    # Time of breakout in minutes since open (9:30 = 0)
    h, m = map(int, breakout_time.split(":"))
    feat["breakout_time_min"] = (h - 9) * 60 + m - 30   # 0 = 9:30

    # Volume at the breakout candle vs avg OR bar volume
    bt_bar = day_df[day_df["time"] == breakout_time]
    if len(bt_bar) > 0 and len(or_bars) > 0 and or_bars["volume"].mean() > 0:
        feat["breakout_volume_ratio"] = (
            bt_bar["volume"].iloc[0] / (or_bars["volume"].mean() + 1e-6)
        )
    else:
        feat["breakout_volume_ratio"] = np.nan

    # Breakout bar's own intraday range
    if len(bt_bar) > 0 and or_open > 0:
        feat["breakout_bar_range_pct"] = (
            (bt_bar["high"].iloc[0] - bt_bar["low"].iloc[0]) / or_open * 100
        )
    else:
        feat["breakout_bar_range_pct"] = np.nan

    # Number of times price touched OR boundary during OR formation
    if direction == 1:
        touches = (or_bars["high"] >= or_high * 0.999).sum()
    else:
        touches = (or_bars["low"] <= or_low * 1.001).sum()
    feat["or_boundary_touches"] = int(touches)

    # ── C. VWAP context ──────────────────────────────────────────────────────
    vwap_series = compute_vwap(hist)
    vwap_at_bt  = vwap_series.iloc[-1]
    bt_close    = hist["close"].iloc[-1]
    feat["vwap_distance_pct"] = (bt_close - vwap_at_bt) / vwap_at_bt * 100
    feat["price_above_vwap"]  = int(bt_close > vwap_at_bt)

    # ── D. Momentum / oscillators ────────────────────────────────────────────
    # Built manually so the project works without pandas-ta if needed
    rsi = compute_rsi(hist["close"], n=14)
    feat["rsi_at_breakout"] = float(rsi.iloc[-1]) if not rsi.isna().all() else np.nan

    atr = compute_atr(hist["high"], hist["low"], hist["close"], n=14)
    feat["atr_at_breakout"] = float(atr.iloc[-1]) if not atr.isna().all() else np.nan

    if HAS_PANDAS_TA:
        try:
            macd_df = ta.macd(hist["close"], fast=12, slow=26, signal=9)
            if macd_df is not None and len(macd_df.columns) >= 3:
                feat["macd_hist_at_breakout"] = float(macd_df.iloc[-1, 2])
        except Exception:
            feat["macd_hist_at_breakout"] = np.nan

        try:
            stoch_df = ta.stoch(hist["high"], hist["low"], hist["close"])
            if stoch_df is not None:
                feat["stoch_k_at_breakout"] = float(stoch_df.iloc[-1, 0])
        except Exception:
            feat["stoch_k_at_breakout"] = np.nan

        try:
            adx_df = ta.adx(hist["high"], hist["low"], hist["close"])
            if adx_df is not None:
                feat["adx_at_breakout"] = float(adx_df.iloc[-1, 0])
        except Exception:
            feat["adx_at_breakout"] = np.nan
    else:
        feat["macd_hist_at_breakout"] = np.nan
        feat["stoch_k_at_breakout"]   = np.nan
        feat["adx_at_breakout"]       = np.nan

    # ── E. Volatility context ─────────────────────────────────────────────────
    # ATR ratio: today's current ATR vs 20-day historical ATR
    # Use symbol_df (full cross-day history) for rolling 20-day context
    all_atr = compute_atr(symbol_df["high"], symbol_df["low"], symbol_df["close"], n=20)
    hist_mask = symbol_df["datetime"] <= hist["datetime"].iloc[-1]
    hist_atr  = all_atr[hist_mask].dropna()

    if len(hist_atr) >= 20 and feat.get("atr_at_breakout") is not None:
        atr_20d = hist_atr.iloc[-20:].mean()
        feat["atr_ratio"] = feat["atr_at_breakout"] / (atr_20d + 1e-6)
    else:
        feat["atr_ratio"] = np.nan

    # 5-day realized volatility: std of daily returns
    days_in_hist = symbol_df[symbol_df["datetime"] < hist["datetime"].iloc[0]]
    daily_closes = (
        days_in_hist.groupby(days_in_hist["datetime"].dt.date)["close"].last()
    )
    if len(daily_closes) >= 5:
        returns = daily_closes.pct_change().dropna()
        feat["realized_vol_5d"] = float(returns.iloc[-5:].std()) * 100
    else:
        feat["realized_vol_5d"] = np.nan

    # ── F. Gap alignment ──────────────────────────────────────────────────────
    if not np.isnan(feat.get("gap_pct", np.nan)):
        gap_dir = 1 if feat["gap_pct"] > 0 else (-1 if feat["gap_pct"] < 0 else 0)
        feat["gap_direction_alignment"] = int(gap_dir == direction)
    else:
        feat["gap_direction_alignment"] = np.nan

    # ── G. Market microstructure ──────────────────────────────────────────────
    avg_or_vol = or_bars["volume"].mean()
    all_prev = symbol_df[symbol_df["datetime"].dt.date < date]
    prev_days_avg_vol = (
        all_prev.groupby(all_prev["datetime"].dt.date)["volume"].sum()
    )
    if len(prev_days_avg_vol) >= 5:
        feat["or_formation_volume_ratio"] = (
            avg_or_vol * 30 / (prev_days_avg_vol.iloc[-5:].mean() / 390 * 30 + 1e-6)
        )
    else:
        feat["or_formation_volume_ratio"] = np.nan

    feat["avg_bar_range_during_or"] = (
        (or_bars["high"] - or_bars["low"]).mean() / or_open * 100
        if len(or_bars) > 0 else np.nan
    )

    # High-low ratio: how much of the OR was up-vs-down
    total_range = or_high - or_low
    if total_range > 0:
        or_first_close = or_bars["close"].iloc[-1]
        feat["or_close_position"] = (or_first_close - or_low) / total_range
    else:
        feat["or_close_position"] = 0.5

    return feat


# ── Main ──────────────────────────────────────────────────────────────────────

def build_features(events_path: str, data_dir: str, output_path: str) -> pd.DataFrame:
    events = pd.read_csv(events_path, parse_dates=["date"])
    events = events[events["label"] != -1].reset_index(drop=True)

    data_dir = Path(data_dir)
    rows = []
    skipped = 0

    for symbol, sym_events in events.groupby("symbol"):
        csv_path = data_dir / f"{symbol}_minute_data.csv"
        if not csv_path.exists():
            print(f"[SKIP] {symbol}: no minute data file found")
            skipped += len(sym_events)
            continue

        symbol_df = load_minute_data(str(csv_path))
        symbol_df["vwap"] = compute_vwap(symbol_df)

        dates = sorted(symbol_df["date"].unique())

        for _, event in sym_events.iterrows():
            ev_date = event["date"].date()
            if ev_date not in dates:
                skipped += 1
                continue

            day_df = symbol_df[symbol_df["date"] == ev_date].copy()

            prev_dates = [d for d in dates if d < ev_date]
            prev_day_df = (
                symbol_df[symbol_df["date"] == prev_dates[-1]].copy()
                if prev_dates else None
            )

            feats = extract_features_for_event(event, day_df, prev_day_df, symbol_df)
            if not feats:
                skipped += 1
                continue

            feats["label"]  = int(event["label"])
            feats["date"]   = str(ev_date)
            feats["symbol"] = symbol
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nFeature matrix: {len(df)} rows × {len(df.columns)} cols")
    print(f"Skipped: {skipped} events")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"Saved → {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events",    default="data/orb_events.csv")
    parser.add_argument("--data-dir",  default="data/")
    parser.add_argument("--output",    default="data/features.csv")
    args = parser.parse_args()
    build_features(args.events, args.data_dir, args.output)


if __name__ == "__main__":
    main()
