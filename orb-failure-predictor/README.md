# ORB Failure Predictor

**A niche ML model for predicting Opening Range Breakout failures on intraday minute data.**

---

## The Niche Pattern

Most ORB (Opening Range Breakout) traders assume that once a stock breaks cleanly above or below its first-30-minute range, it will continue in that direction. **This model predicts when they're wrong.**

### Setup definition

| Step | What happens |
|------|-------------|
| 1 | The first 30 minutes (9:30–9:59 ET) form the **Opening Range**: `OR_high` and `OR_low` |
| 2 | After 10:00, the first bar whose **close** exceeds the OR boundary by ≥ 0.15% triggers an **ORB event** |
| 3 | The model predicts: will the breakout **fail** (price re-enters the OR within 60 minutes)? |

### Labels

| Label | Meaning | Training target |
|-------|---------|----------------|
| `1` | **Failure** — price re-enters OR within 60 min | Positive class |
| `0` | **Success** — price never re-enters OR within 60 min | Negative class |
| `-1` | **Unclear** — observation window cut short by market close | Excluded |

### Why this is hard and interesting

- Many ORBs fail — estimates suggest 40–60% of breakouts reverse
- But knowing *which specific* breakouts fail, *before* they fail, is the edge
- The failure signal is a function of OR geometry, volume profile, VWAP context,
  momentum, and gap structure — all computable at the exact moment of breakout

---

## Data Format

Same OHLCV format as `ml-bounce-predictor`:

```
datetime,symbol,open,high,low,close,volume
2025-10-30 09:30:00,NVDA,135.51,135.98,134.88,135.43,267285
2025-10-30 09:31:00,NVDA,135.43,136.10,135.20,135.87,198342
...
```

Place your files in `data/` as `{SYMBOL}_minute_data.csv`.

---

## Pipeline

```
data/{SYMBOL}_minute_data.csv
          │
          ▼
  1. orb_detection.py        ─── Detect ORB events → data/orb_events.csv
          │
          ▼
  2. feature_engineering.py  ─── Build 20-feature matrix → data/features.csv
          │
          ▼
  3. train_model.py           ─── LightGBM + Optuna → models/
          │
          ▼
  4. evaluate.py              ─── SHAP + walk-forward backtest + calibration
          │
          ▼
  5. predict.py               ─── Live or batch inference
```

---

## Usage

### Step 1 — Detect ORB events

```bash
# All symbols in data/
python scripts/orb_detection.py --data-dir data/ --all-symbols

# Single symbol
python scripts/orb_detection.py --data-dir data/ --symbol NVDA
```

Output: `data/orb_events.csv`

### Step 2 — Build features

```bash
python scripts/feature_engineering.py \
    --events data/orb_events.csv \
    --data-dir data/ \
    --output data/features.csv
```

### Step 3 — Train

```bash
# Full Optuna search (100 trials, recommended)
python scripts/train_model.py --features data/features.csv

# Quick run (no Optuna)
python scripts/train_model.py --features data/features.csv --n-trials 0
```

### Step 4 — Evaluate

```bash
python scripts/evaluate.py --features data/features.csv --mode all
# modes: shap | backtest | calibration | all
```

### Step 5 — Predict

```bash
# Batch: score all days in a file
python scripts/predict.py batch --symbol NVDA --data-file data/NVDA_minute_data.csv

# Single date
python scripts/predict.py batch --symbol NVDA \
    --data-file data/NVDA_minute_data.csv --date 2025-10-30

# Live: polls every 60 seconds while market is open
python scripts/predict.py live --symbol NVDA \
    --data-file data/NVDA_minute_data.csv --poll 60
```

---

## Features (20 total)

### A. Opening Range geometry
| Feature | Description |
|---------|-------------|
| `or_width_pct` | OR range as % of day open — narrow ORs break more often but fail faster |
| `or_mid_vs_open_pct` | Whether OR is skewed above/below the open |
| `or_close_position` | Where OR's closing bar sits within the range (0=bottom, 1=top) |

### B. Breakout characteristics
| Feature | Description |
|---------|-------------|
| `breakout_direction` | +1 = up, −1 = down |
| `penetration_pct` | How far price has moved through the OR boundary |
| `breakout_time_min` | Minutes since open (0=9:30). Early breakouts fail more |
| `breakout_volume_ratio` | Breakout bar volume / avg OR bar volume |
| `breakout_bar_range_pct` | Breakout bar's H−L range as % of open |
| `or_boundary_touches` | Times price touched OR boundary before breaking (more = weaker breakout) |

### C. VWAP context
| Feature | Description |
|---------|-------------|
| `vwap_distance_pct` | % distance between price and VWAP at breakout |
| `price_above_vwap` | Binary: 1 = price above VWAP |

### D. Momentum
| Feature | Description |
|---------|-------------|
| `rsi_at_breakout` | RSI(14) at the breakout bar |
| `macd_hist_at_breakout` | MACD histogram — expanding or contracting? |
| `stoch_k_at_breakout` | Stochastic %K — overbought/oversold at breakout |
| `adx_at_breakout` | ADX — trend strength; weak ADX = more likely to fail |

### E. Volatility context
| Feature | Description |
|---------|-------------|
| `atr_at_breakout` | ATR(14) on the current day |
| `atr_ratio` | Current ATR / 20-day mean ATR — is today unusually volatile? |
| `realized_vol_5d` | 5-day realized volatility (std of daily returns) |

### F. Gap & prior day
| Feature | Description |
|---------|-------------|
| `gap_pct` | Gap from previous close as % |
| `gap_direction_alignment` | 1 if gap direction matches breakout direction |
| `prev_day_return` | Prior day's close-to-close return |
| `prev_day_range_pct` | Prior day's H−L range |
| `or_high_vs_prev_close` / `or_low_vs_prev_close` | OR boundary position vs prior close |

### G. Microstructure
| Feature | Description |
|---------|-------------|
| `or_formation_volume_ratio` | OR-period total volume vs 5-day avg |
| `avg_bar_range_during_or` | Mean bar range during OR — rangy OR = fragile breakout |

---

## ML Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Model | **LightGBM** | Handles NaN natively, faster/better than RF on tabular data |
| Hyperparameter search | **Optuna (TPE sampler)** | Bayesian search finds better configs than GridSearchCV in fewer trials |
| Cross-validation | **TimeSeriesSplit** | Temporal ordering respected — no future leakage |
| Class imbalance | **SMOTE-Tomek** | Combined over+under-sampling; more nuanced than `class_weight='balanced'` |
| Probability calibration | **Isotonic regression** | P=0.7 actually means ~70% failure rate |
| Threshold optimisation | **F-beta (β=0.5)** | Precision-weighted; reduces false alarms in a trading context |
| Explainability | **SHAP TreeExplainer** | Per-prediction feature attribution |

---

## Key Differences vs `ml-bounce-predictor`

| | Bounce Predictor | ORB Failure Predictor |
|--|--|--|
| Setup trigger | ≥2% drop + ≥0.5% bounce from rolling low | ORB detection after 30-min range |
| Prediction target | Will price recover to open by EOD? | Will the breakout fail within 60 min? |
| Time horizon | Full day | 60-minute observation window |
| Model | Random Forest | LightGBM |
| Hyperparameter search | GridSearchCV (108 combos) | Optuna TPE (Bayesian, 100 trials) |
| Cross-validation | StratifiedKFold | **TimeSeriesSplit** (no leakage) |
| Class imbalance | `class_weight='balanced'` | SMOTE-Tomek |
| Calibration | None | Isotonic regression |
| Explainability | Feature importance (Gini) | SHAP values |

---

## Directory Structure

```
orb-failure-predictor/
├── data/
│   ├── {SYMBOL}_minute_data.csv   ← your data goes here
│   ├── orb_events.csv             ← generated by orb_detection.py
│   ├── features.csv               ← generated by feature_engineering.py
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── walkforward_ap.png
│   └── calibration_plot.png
├── models/
│   ├── orb_failure_model.pkl
│   ├── orb_failure_threshold.txt
│   ├── orb_failure_features.txt
│   └── orb_failure_params.json
├── scripts/
│   ├── orb_detection.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
└── requirements.txt
```
