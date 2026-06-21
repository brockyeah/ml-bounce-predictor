"""
orb_detection.py
================
Opening Range Breakout (ORB) Detection

The niche pattern this model hunts:
  1. The first 30 minutes (9:30–9:59 ET) form the "Opening Range" (OR):
       OR_high = highest high of bars 1–30
       OR_low  = lowest low  of bars 1–30
  2. After 10:00 ET, the first bar whose CLOSE breaches the OR boundary by
     >= MIN_BREAKOUT_PCT triggers an "ORB event".
  3. This event is labeled:
       1 (FAILURE)  — price re-enters the OR within FAILURE_WINDOW_MIN minutes
       0 (SUCCESS)  — price never re-enters the OR within the window
      -1 (UNCLEAR)  — everything else (excluded from training)

Output: data/orb_events.csv
  Columns: date, symbol, direction, or_high, or_low, or_width_pct,
            breakout_time, breakout_price, breakout_bar_idx, label

Usage:
    python scripts/orb_detection.py --data-dir data/ --symbol NVDA
    python scripts/orb_detection.py --data-dir data/ --all-symbols
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

# ── Tunable thresholds ──────────────────────────────────────────────────────
OR_MINUTES        = 30          # how many minutes define the Opening Range
MIN_BREAKOUT_PCT  = 0.15        # minimum % through the OR boundary to count
FAILURE_WINDOW_MIN = 60         # minutes to observe after breakout
MIN_OR_WIDTH_PCT  = 0.30        # discard tiny-range days (noise)
MARKET_OPEN       = "09:30"
MARKET_CLOSE      = "16:00"
OR_END            = "09:59"     # last minute included in OR window


def load_minute_data(filepath: str) -> pd.DataFrame:
    """Load minute OHLCV CSV (same format as ml-bounce-predictor)."""
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")

    # Filter to regular trading hours only
    df = df[(df["time"] >= MARKET_OPEN) & (df["time"] < MARKET_CLOSE)].copy()
    df = df.reset_index(drop=True)
    return df


def compute_opening_range(day_df: pd.DataFrame) -> dict | None:
    """
    Given a single day's minute data, compute the Opening Range.
    Returns None if the day doesn't meet quality thresholds.
    """
    or_bars = day_df[day_df["time"] <= OR_END]
    post_or = day_df[day_df["time"] > OR_END]

    if len(or_bars) < OR_MINUTES - 5:   # allow a few missing bars
        return None
    if len(post_or) < 30:               # need observation window
        return None

    or_high  = or_bars["high"].max()
    or_low   = or_bars["low"].min()
    or_open  = or_bars.iloc[0]["open"]   # first bar's open = day open

    # Discard featureless days (tiny range)
    or_width_pct = (or_high - or_low) / or_open * 100
    if or_width_pct < MIN_OR_WIDTH_PCT:
        return None

    return {
        "or_high":      or_high,
        "or_low":       or_low,
        "or_open":      or_open,
        "or_width_pct": or_width_pct,
        "or_bars":      or_bars,
        "post_or":      post_or,
    }


def detect_breakout(post_or: pd.DataFrame, or_high: float, or_low: float) -> dict | None:
    """
    Scan post-OR bars for the first close that breaches the OR boundary.
    Returns breakout metadata or None if no clean breakout exists.
    """
    for idx, row in post_or.iterrows():
        # Upside breakout: close above OR high
        if row["close"] > or_high * (1 + MIN_BREAKOUT_PCT / 100):
            return {
                "direction":       1,        # 1 = up, -1 = down
                "breakout_price":  row["close"],
                "breakout_time":   row["time"],
                "breakout_dt":     row["datetime"],
                "breakout_bar_idx": idx,
                "or_boundary":     or_high,
                "penetration_pct": (row["close"] - or_high) / or_high * 100,
            }
        # Downside breakout: close below OR low
        if row["close"] < or_low * (1 - MIN_BREAKOUT_PCT / 100):
            return {
                "direction":       -1,
                "breakout_price":  row["close"],
                "breakout_time":   row["time"],
                "breakout_dt":     row["datetime"],
                "breakout_bar_idx": idx,
                "or_boundary":     or_low,
                "penetration_pct": (or_low - row["close"]) / or_low * 100,
            }
    return None


def label_breakout(
    post_or: pd.DataFrame,
    breakout: dict,
    or_high: float,
    or_low: float,
) -> int:
    """
    Label the breakout event.
      1  = FAILURE  (price re-enters OR within FAILURE_WINDOW_MIN)
      0  = SUCCESS  (price stays outside OR for the full window)
     -1  = UNCLEAR  (window extends past market close — not used)
    """
    bt_dt    = breakout["breakout_dt"]
    direction = breakout["direction"]

    # Observation window: bars after the breakout bar
    window_end = bt_dt + pd.Timedelta(minutes=FAILURE_WINDOW_MIN)
    obs = post_or[
        (post_or["datetime"] > bt_dt) &
        (post_or["datetime"] <= window_end)
    ]

    if len(obs) < 10:            # window cut short by market close
        return -1

    if direction == 1:           # upside breakout — failure = low dips back below OR high
        re_entry = (obs["low"] < or_high).any()
    else:                        # downside breakout — failure = high pops back above OR low
        re_entry = (obs["high"] > or_low).any()

    return 1 if re_entry else 0


def detect_orb_events(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Main detection loop: one OR event per trading day."""
    records = []
    dates   = df["date"].unique()

    for date in dates:
        day_df = df[df["date"] == date].copy()
        day_df = day_df.reset_index(drop=True)

        or_data = compute_opening_range(day_df)
        if or_data is None:
            continue

        breakout = detect_breakout(
            or_data["post_or"],
            or_data["or_high"],
            or_data["or_low"],
        )
        if breakout is None:
            continue

        label = label_breakout(
            or_data["post_or"],
            breakout,
            or_data["or_high"],
            or_data["or_low"],
        )

        records.append({
            "date":             str(date),
            "symbol":           symbol,
            "direction":        breakout["direction"],
            "or_high":          round(or_data["or_high"], 4),
            "or_low":           round(or_data["or_low"], 4),
            "or_open":          round(or_data["or_open"], 4),
            "or_width_pct":     round(or_data["or_width_pct"], 4),
            "breakout_time":    breakout["breakout_time"],
            "breakout_price":   round(breakout["breakout_price"], 4),
            "penetration_pct":  round(breakout["penetration_pct"], 4),
            "or_boundary":      round(breakout["or_boundary"], 4),
            "label":            label,
        })

    return pd.DataFrame(records)


def print_summary(events: pd.DataFrame) -> None:
    """Rich terminal summary of detected events."""
    total   = len(events)
    labeled = events[events["label"] != -1]
    failures = (labeled["label"] == 1).sum()
    success  = (labeled["label"] == 0).sum()

    table = Table(title="ORB Event Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric",  style="bold")
    table.add_column("Value",   justify="right")

    table.add_row("Total ORB events detected",      str(total))
    table.add_row("Labeled events (excl. unclear)", str(len(labeled)))
    table.add_row("Label 1 — Failures",             str(failures))
    table.add_row("Label 0 — Successes",            str(success))
    if len(labeled) > 0:
        table.add_row("Failure rate",
                      f"{failures/len(labeled)*100:.1f}%")

    up_breaks   = (labeled["direction"] ==  1).sum()
    down_breaks = (labeled["direction"] == -1).sum()
    table.add_row("Upside breakouts",   str(up_breaks))
    table.add_row("Downside breakouts", str(down_breaks))

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Detect ORB events from minute data")
    parser.add_argument("--data-dir",    default="data/",  help="Directory with *_minute_data.csv files")
    parser.add_argument("--output",      default="data/orb_events.csv")
    parser.add_argument("--symbol",      default=None,     help="Single symbol to process")
    parser.add_argument("--all-symbols", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_events = []

    if args.symbol:
        files = list(data_dir.glob(f"{args.symbol}_minute_data.csv"))
    else:
        files = list(data_dir.glob("*_minute_data.csv"))

    if not files:
        console.print("[red]No minute data files found.[/red]")
        return

    for f in files:
        symbol = f.stem.replace("_minute_data", "")
        console.print(f"[cyan]Processing {symbol}...[/cyan]")
        df = load_minute_data(str(f))
        events = detect_orb_events(df, symbol)
        all_events.append(events)
        console.print(f"  [green]{len(events)} events detected[/green]")

    combined = pd.concat(all_events, ignore_index=True)
    combined.to_csv(args.output, index=False)
    console.print(f"\n[bold green]Saved {len(combined)} events → {args.output}[/bold green]\n")
    print_summary(combined)


if __name__ == "__main__":
    main()
