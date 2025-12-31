import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# plot configuration
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 7)

# data pull
DATA_DIR = "./data"

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index
    
    RSI ranges from 0-100:
    - Below 30: Oversold (potential bounce)
    - Above 70: Overbought
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Returns the MACD histogram (MACD line - signal line)
    Positive = bullish momentum, Negative = bearish momentum
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_histogram


def load_stock_data(symbol):
    """
    Load minute-level data for a given stock
    
    Args:
        symbol: stock ticker (e.g., 'NVDA')
    
    Returns:
        DataFrame with minute bars in EST timezone
    """
    filepath = os.path.join(DATA_DIR, f'{symbol}_minute_data.csv')
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    
    # convert UTC to Eastern Time
    # df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df
    

def plot_stock_intraday(df, symbol):
    """Plot minute-level price chart for a stock"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    
    # price chart
    ax1.plot(df['datetime'], df['close'], linewidth=0.8, color='#1f77b4')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} - Minute-Level Price Action')
    ax1.grid(True, alpha=0.3)
    
    # volume chart
    ax2.bar(df['datetime'], df['volume'], width=0.0005, color='gray', alpha=0.6)
    ax2.set_xlabel('Date/Time')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(DATA_DIR, f'{symbol}_intraday_plot.png')
    fig.savefig(save_path)
    print(f"Intraday plot saved to {save_path}")
    

def plot_single_day(df, symbol, date):
    """Plot intraday movements for a specific date"""
    # filter to just that day
    day_data = df[df['datetime'].dt.date == pd.to_datetime(date).date()]
    
    if len(day_data) == 0:
        print(f"No data for {date}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    
    # price chart
    ax1.plot(day_data['datetime'], day_data['close'], linewidth=1.2, color='#1f77b4')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} - {date} Intraday')
    ax1.grid(True, alpha=0.3)
    
    # volume
    ax2.bar(day_data['datetime'], day_data['volume'], width=0.0005, color='gray', alpha=0.6)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(DATA_DIR, f'{symbol}_{date}_intraday_plot.png')
    fig.savefig(save_path)
    print(f"Single day plot saved to {save_path}")
    
    
def calculate_daily_stats(df):
    """
    Calculate key intraday metrics for each trading day
    Returns DataFrame with one row per day
    """
    df['date'] = df['datetime'].dt.date
    
    daily_stats = df.groupby('date').agg({
        'open': 'first',      # opening price
        'high': 'max',        # intraday high
        'low': 'min',         # intraday low
        'close': 'last',      # closing price
        'volume': 'sum'       # total volume
    }).reset_index()
    
    # calculate % changes
    daily_stats['day_change_pct'] = ((daily_stats['close'] - daily_stats['open']) / daily_stats['open']) * 100
    daily_stats['intraday_range_pct'] = ((daily_stats['high'] - daily_stats['low']) / daily_stats['open']) * 100
    daily_stats['max_drawdown_pct'] = ((daily_stats['low'] - daily_stats['open']) / daily_stats['open']) * 100
    
    return daily_stats.round(2)


# def detect_bounces(df, min_drop_pct, min_bounce_pct):
#     """
#     Detect potential bounce moments in minute data
    
#     A bounce is when:
#     1. Stock has dropped min_drop_pct% from day's open
#     2. Price suddenly ticks up min_bounce_pct% from recent low
    
#     Returns list of bounce moments with metadata
#     """
#     df = df.copy()
#     df['date'] = df['datetime'].dt.date
    
#     bounces = []
    
#     # process each day separately
#     for date in df['date'].unique():
#         day_df = df[df['date'] == date].copy()
#         day_open = day_df.iloc[0]['open']
        
#         # track rolling low
#         day_df['rolling_low'] = day_df['low'].cummin()
        
#         for i in range(1, len(day_df)):
#             current_price = day_df.iloc[i]['close']
#             prev_low = day_df.iloc[i-1]['rolling_low']
            
#             # how far down from open?
#             drop_from_open = ((current_price - day_open) / day_open) * 100
            
#             # did we bounce from the low?
#             bounce_from_low = ((current_price - prev_low) / prev_low) * 100
            
#             # check if this qualifies as a bounce moment
#             if drop_from_open <= -min_drop_pct and bounce_from_low >= min_bounce_pct:
#                 bounces.append({
#                     'datetime': day_df.iloc[i]['datetime'],
#                     'symbol': day_df.iloc[i]['symbol'],
#                     'price': current_price,
#                     'drop_from_open_pct': round(drop_from_open, 2),
#                     'bounce_from_low_pct': round(bounce_from_low, 2),
#                     'volume': day_df.iloc[i]['volume']
#                 })
    
#     return pd.DataFrame(bounces)
def detect_bounces(df, min_drop_pct=2.0, min_bounce_pct=0.5):
    """
    Vectorized bounce detection - 10-20x faster
    """
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    
    all_bounces = []
    
    # process each day
    for date in df['date'].unique():
        day_df = df[df['date'] == date].copy().reset_index(drop=True)
        
        if len(day_df) < 2:
            continue
            
        day_open = day_df.iloc[0]['open']
        
        # vectorized calculations
        day_df['rolling_low'] = day_df['low'].cummin()
        day_df['drop_from_open_pct'] = ((day_df['close'] - day_open) / day_open) * 100
        day_df['prev_rolling_low'] = day_df['rolling_low'].shift(1)
        day_df['bounce_from_low_pct'] = ((day_df['close'] - day_df['prev_rolling_low']) / day_df['prev_rolling_low']) * 100
        
        # find bounces using boolean indexing
        bounce_mask = (
            (day_df['drop_from_open_pct'] <= -min_drop_pct) &
            (day_df['bounce_from_low_pct'] >= min_bounce_pct)
        )
        
        # extract bounces
        bounces = day_df[bounce_mask][['datetime', 'symbol', 'close', 'drop_from_open_pct', 
                                        'bounce_from_low_pct', 'volume']].copy()
        bounces.columns = ['datetime', 'symbol', 'price', 'drop_from_open_pct', 
                          'bounce_from_low_pct', 'volume']
        
        if len(bounces) > 0:
            all_bounces.append(bounces)
    
    if len(all_bounces) == 0:
        return pd.DataFrame()
    
    return pd.concat(all_bounces, ignore_index=True)


# def label_bounces(df, bounces_df):
#     """
#     Label each bounce as real recovery (1) or fake-out (0)
#     Stops labeling a day once first Label 1 is found
#     """
#     df = df.copy()
#     df['date'] = df['datetime'].dt.date
#     bounces_df = bounces_df.sort_values('datetime')  # process chronologically
    
#     labeled_bounces = []
#     days_with_recovery = set()  # track which days already found recovery
    
#     for _, bounce in bounces_df.iterrows():
#         bounce_time = bounce['datetime']
#         bounce_date = bounce_time.date()
        
#         # skip if we already found the recovery bounce for this day
#         if bounce_date in days_with_recovery:
#             continue
        
#         bounce_price = bounce['price']
        
#         # get that day's data
#         day_df = df[df['date'] == bounce_date].copy()
#         day_open = day_df.iloc[0]['open']
#         day_close = day_df.iloc[-1]['close']
        
#         # calculate recovery percentage
#         drop_from_open = day_open - bounce_price
#         actual_recovery = day_close - bounce_price
#         recovery_pct = (actual_recovery / drop_from_open) * 100 if drop_from_open != 0 else 0
        
#         # check if it dropped back within 60 mins
#         future_data = day_df[day_df['datetime'] > bounce_time].head(60)
#         dropped_back = (future_data['low'].min() < bounce_price) if len(future_data) > 0 else False
        
#         # assign label
#         if recovery_pct >= 35:
#             label = 1
#             days_with_recovery.add(bounce_date)  # mark this day as done
#         elif dropped_back:
#             label = 0
#         else:
#             label = -1
        
#         labeled_bounces.append({
#             'datetime': bounce_time,
#             'symbol': bounce['symbol'],
#             'price': bounce_price,
#             'drop_from_open_pct': bounce['drop_from_open_pct'],
#             'bounce_from_low_pct': bounce['bounce_from_low_pct'],
#             'volume': bounce['volume'],
#             'recovery_pct': round(recovery_pct, 2),
#             'dropped_back': dropped_back,
#             'label': label
#         })
    
#     return pd.DataFrame(labeled_bounces)
def label_bounces(df, bounces_df):
    """
    Vectorized bounce labeling - 100x faster
    """
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    bounces_df = bounces_df.copy()
    bounces_df['date'] = bounces_df['datetime'].dt.date
    bounces_df = bounces_df.sort_values('datetime').reset_index(drop=True)
    
    print(f"  Labeling {len(bounces_df)} bounces...")
    
    # Pre-compute day open/close for all dates
    day_stats = df.groupby('date').agg({
        'open': 'first',
        'close': 'last',
        'datetime': ['min', 'max']
    }).reset_index()
    day_stats.columns = ['date', 'day_open', 'day_close', 'day_start', 'day_end']
    
    # Merge day stats into bounces
    bounces_df = bounces_df.merge(day_stats, on='date', how='left')
    
    # Calculate recovery percentage (vectorized)
    drop_from_open = bounces_df['day_open'] - bounces_df['price']
    actual_recovery = bounces_df['day_close'] - bounces_df['price']
    bounces_df['recovery_pct'] = (actual_recovery / drop_from_open * 100).fillna(0)
    
    # Check if dropped back within 60 mins (still needs some iteration, but optimized)
    print("  Checking drop-backs...")
    dropped_back_list = []
    
    # Group by date for efficiency
    for date in bounces_df['date'].unique():
        day_df = df[df['date'] == date].sort_values('datetime')
        day_bounces = bounces_df[bounces_df['date'] == date].copy()
        
        for idx, bounce in day_bounces.iterrows():
            bounce_time = bounce['datetime']
            bounce_price = bounce['price']
            
            # Get next 60 bars
            future_mask = day_df['datetime'] > bounce_time
            future_data = day_df[future_mask].head(60)
            
            if len(future_data) > 0:
                dropped = future_data['low'].min() < bounce_price
            else:
                dropped = False
            
            dropped_back_list.append(dropped)
    
    bounces_df['dropped_back'] = dropped_back_list
    
    # Assign labels (vectorized)
    print("  Assigning labels...")
    bounces_df['label'] = -1  # default
    bounces_df.loc[bounces_df['recovery_pct'] >= 35, 'label'] = 1
    bounces_df.loc[(bounces_df['recovery_pct'] < 35) & bounces_df['dropped_back'], 'label'] = 0
    
    # Handle "only one Label 1 per day" rule
    print("  Applying one-recovery-per-day rule...")
    recovery_bounces = bounces_df[bounces_df['label'] == 1].copy()
    
    # For each day with Label 1, keep only the first one
    first_recoveries = recovery_bounces.groupby('date').first().reset_index()
    
    # Remove ALL bounces after first Label 1 (not just relabel)
    rows_to_keep = []

    for date in bounces_df['date'].unique():
        day_bounces = bounces_df[bounces_df['date'] == date].copy()
        
        label_1_bounces = day_bounces[day_bounces['label'] == 1]
        
        if len(label_1_bounces) > 0:
            # Keep only bounces UP TO first Label 1 (inclusive)
            first_recovery_time = label_1_bounces.iloc[0]['datetime']
            day_bounces_to_keep = day_bounces[day_bounces['datetime'] <= first_recovery_time]
            rows_to_keep.append(day_bounces_to_keep)
        else:
            # No Label 1 - keep all bounces
            rows_to_keep.append(day_bounces)

    bounces_df = pd.concat(rows_to_keep, ignore_index=True)  # ← Rebuilds dataset!
    
    # Select final columns
    result = bounces_df[[
        'datetime', 'symbol', 'price', 'drop_from_open_pct',
        'bounce_from_low_pct', 'volume', 'recovery_pct', 'dropped_back', 'label'
    ]].copy()
    
    result['recovery_pct'] = result['recovery_pct'].round(2)
    
    return result


def process_all_stocks(symbols, min_drop_pct, min_bounce_pct):
    """
    Process multiple stocks and combine labeled bounces
    """
    all_labeled = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        # load data
        df = load_stock_data(symbol)
        
        # detect bounces
        bounces = detect_bounces(df, min_drop_pct, min_bounce_pct)
        print(f"  Found {len(bounces)} bounces")
        
        # label them
        if len(bounces) > 0:
            labeled = label_bounces(df, bounces)
            all_labeled.append(labeled)
            
            # show distribution
            print(f"  Label distribution:")
            print(labeled['label'].value_counts().to_dict())
    
    # combine all stocks
    combined = pd.concat(all_labeled, ignore_index=True)
    return combined


# def engineer_features(df, bounces_df, spy_df):
#     """
#     Create features for each bounce moment to help predict Label 1 vs 0
#     Now includes: volume ratio, prior momentum, number of prior bounces, SPY trend
#     """
#     df = df.copy()
#     bounces_df = bounces_df.copy()
#     spy_df = spy_df.copy()
    
#     features = []
    
#     for _, bounce in bounces_df.iterrows():
#         bounce_time = bounce['datetime']
#         bounce_date = bounce_time.date()
        
#         # get that day's data
#         day_df = df[(df['datetime'].dt.date == bounce_date) & 
#                     (df['symbol'] == bounce['symbol'])].copy()
        
#         # data up to bounce moment
#         prior_df = day_df[day_df['datetime'] <= bounce_time]
        
#         if len(prior_df) < 10:  # need some history
#             continue
            
#         # basic features from bounce data
#         feat = {
#             'datetime': bounce_time,
#             'symbol': bounce['symbol'],
#             'drop_from_open_pct': bounce['drop_from_open_pct'],
#             'bounce_size_pct': bounce['bounce_from_low_pct'],
#             'label': bounce['label']
#         }
        
#         # time-based features
#         # feat['hour'] = bounce_time.hour
#         # feat['minute'] = bounce_time.minute
        
#         feat['minutes_since_open'] = (bounce_time.hour - 9) * 60 + (bounce_time.minute - 30)
        
#         # volume features
#         recent_vol = prior_df.tail(5)['volume'].mean()
#         day_avg_vol = prior_df['volume'].mean()
#         feat['volume_ratio'] = recent_vol / day_avg_vol if day_avg_vol > 0 else 1
#         feat['bounce_volume'] = bounce['volume']
        
#         # price momentum features
#         feat['prior_5min_change'] = ((prior_df.iloc[-1]['close'] - prior_df.iloc[-6]['close']) / 
#                                      prior_df.iloc[-6]['close'] * 100) if len(prior_df) >= 6 else 0
        
#         # how many bounces already today?
#         feat['num_prior_bounces'] = len(bounces_df[(bounces_df['datetime'] < bounce_time) & 
#                                                     (bounces_df['datetime'].dt.date == bounce_date) &
#                                                     (bounces_df['symbol'] == bounce['symbol'])])
        
#                 # NEW: Market context features from SPY
#         spy_day = spy_df[spy_df['datetime'].dt.date == bounce_date].copy()
        
#         if len(spy_day) > 0:
#             spy_open = spy_day.iloc[0]['open']
#             spy_at_bounce = spy_day[spy_day['datetime'] <= bounce_time].iloc[-1]['close'] if len(spy_day[spy_day['datetime'] <= bounce_time]) > 0 else spy_open
            
#             # How much is SPY down from its open?
#             feat['spy_drop_from_open_pct'] = ((spy_at_bounce - spy_open) / spy_open) * 100
            
#             # Is stock dropping more or less than market?
#             feat['drop_vs_spy'] = feat['drop_from_open_pct'] - feat['spy_drop_from_open_pct']
#         else:
#             feat['spy_drop_from_open_pct'] = 0
#             feat['drop_vs_spy'] = feat['drop_from_open_pct']
        
#         features.append(feat)
    
#     return pd.DataFrame(features)
# def engineer_features(df, bounces_df, spy_df):
#     """
#     Vectorized feature engineering - much faster
#     """
#     if len(bounces_df) == 0:
#         return pd.DataFrame()
    
#     df = df.copy()
#     bounces_df = bounces_df.copy()
#     spy_df = spy_df.copy()
    
#     # add date column to all dataframes
#     df['date'] = df['datetime'].dt.date
#     bounces_df['date'] = bounces_df['datetime'].dt.date
#     spy_df['date'] = spy_df['datetime'].dt.date
    
#     # pre-calculate day-level features on df
#     df['day_open'] = df.groupby('date')['open'].transform('first')
    
#     # calculate volume features on df
#     df['rolling_volume_5'] = df.groupby(['date', 'symbol'])['volume'].transform(
#         lambda x: x.rolling(5, min_periods=1).mean()
#     )
#     df['day_avg_volume'] = df.groupby(['date', 'symbol'])['volume'].transform('mean')
    
#     # calculate momentum features on df
#     df['close_shift6'] = df.groupby(['date', 'symbol'])['close'].shift(6)
    
#     # merge all df features into bounces at once
#     df_features = df[['datetime', 'symbol', 'close', 'day_open', 
#                       'rolling_volume_5', 'day_avg_volume', 'close_shift6']].copy()
    
#     bounces_enriched = bounces_df.merge(
#         df_features,
#         on=['datetime', 'symbol'],
#         how='left'
#     )
    
#         # Calculate RSI on the full day's data
#     df['rsi'] = df.groupby(['date', 'symbol'])['close'].transform(
#         lambda x: calculate_rsi(x, period=14)
#     )

#     # Calculate MACD
#     df['macd'], df['macd_signal'] = calculate_macd(df.groupby(['date', 'symbol'])['close'])
#     df['macd_diff'] = df['macd'] - df['macd_signal']

#     # Merge into bounces
#     df_indicators = df[['datetime', 'symbol', 'rsi', 'macd_diff']].copy()
#     bounces_enriched = bounces_enriched.merge(
#         df_indicators,
#         on=['datetime', 'symbol'],
#         how='left'
#     )
    
#     # time features
#     bounces_enriched['minutes_since_open'] = (
#         (bounces_enriched['datetime'].dt.hour - 9) * 60 + 
#         (bounces_enriched['datetime'].dt.minute - 30)
#     )
    
#     # volume ratio
#     bounces_enriched['volume_ratio'] = (
#         bounces_enriched['rolling_volume_5'] / bounces_enriched['day_avg_volume']
#     ).fillna(1)
    
#     # price momentum (use 'close' from the merge, not 'price' from bounces)
#     bounces_enriched['prior_5min_change'] = (
#         (bounces_enriched['close'] - bounces_enriched['close_shift6']) / 
#         bounces_enriched['close_shift6'] * 100
#     ).fillna(0)
    
#     # count prior bounces
#     bounces_enriched = bounces_enriched.sort_values(['symbol', 'date', 'datetime'])
#     bounces_enriched['num_prior_bounces'] = bounces_enriched.groupby(['symbol', 'date']).cumcount()
    
#     # SPY features
#     spy_df['spy_open'] = spy_df.groupby('date')['open'].transform('first')
#     spy_df['spy_drop_from_open_pct'] = ((spy_df['close'] - spy_df['spy_open']) / spy_df['spy_open']) * 100
    
#     # merge SPY data
#     spy_subset = spy_df[['datetime', 'spy_drop_from_open_pct']].sort_values('datetime')
#     bounces_enriched = bounces_enriched.sort_values('datetime')
    
#     bounces_enriched = pd.merge_asof(
#         bounces_enriched,
#         spy_subset,
#         on='datetime',
#         direction='backward'
#     )
    
#     bounces_enriched['spy_drop_from_open_pct'] = bounces_enriched['spy_drop_from_open_pct'].fillna(0)
#     bounces_enriched['drop_vs_spy'] = (
#         bounces_enriched['drop_from_open_pct'] - bounces_enriched['spy_drop_from_open_pct']
#     )
    
#     # select final columns
#     features = bounces_enriched[[
#         'datetime', 'symbol', 'drop_from_open_pct', 'bounce_from_low_pct',
#         'minutes_since_open', 'volume_ratio', 'volume',
#         'prior_5min_change', 'num_prior_bounces',
#         'spy_drop_from_open_pct', 'drop_vs_spy',
#         'rsi', 'macd_diff',
#     ]].copy()
    
#     features.columns = [
#         'datetime', 'symbol', 'drop_from_open_pct', 'bounce_size_pct',
#         'minutes_since_open', 'volume_ratio', 'bounce_volume',
#         'prior_5min_change', 'num_prior_bounces',
#         'spy_drop_from_open_pct', 'drop_vs_spy'
#     ]
    
#     return features
def engineer_features(df, bounces_df, spy_df):
    """
    Vectorized feature engineering with technical indicators
    """
    if len(bounces_df) == 0:
        return pd.DataFrame()
    
    df = df.copy()
    bounces_df = bounces_df.copy()
    spy_df = spy_df.copy()
    
    # add date column
    df['date'] = df['datetime'].dt.date
    bounces_df['date'] = bounces_df['datetime'].dt.date
    spy_df['date'] = spy_df['datetime'].dt.date
    
    # pre-calculate day-level features
    df['day_open'] = df.groupby('date')['open'].transform('first')
    
    # calculate volume features
    df['rolling_volume_5'] = df.groupby(['date', 'symbol'])['volume'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['day_avg_volume'] = df.groupby(['date', 'symbol'])['volume'].transform('mean')
    
    # calculate momentum features
    df['close_shift6'] = df.groupby(['date', 'symbol'])['close'].shift(6)
    
    # NEW: Calculate technical indicators
    # print("  Calculating RSI...")
    df['rsi'] = df.groupby(['date', 'symbol'])['close'].transform(
        lambda x: calculate_rsi(x, period=14)
    )
    
    # print("  Calculating MACD...")
    df['macd_histogram'] = df.groupby(['date', 'symbol'])['close'].transform(
        lambda x: calculate_macd(x, fast=12, slow=26, signal=9)
    )
    
    # merge all df features into bounces
    df_features = df[['datetime', 'symbol', 'close', 'day_open', 
                      'rolling_volume_5', 'day_avg_volume', 'close_shift6',
                      'rsi', 'macd_histogram']].copy()  # Added RSI and MACD
    
    bounces_enriched = bounces_df.merge(
        df_features,
        on=['datetime', 'symbol'],
        how='left'
    )
    
    # ============================================================
    # Calculate market open datetime (9:30 AM) for each date
    bounces_enriched['market_open'] = pd.to_datetime(
        bounces_enriched['date'].astype(str) + ' 09:30:00'
    )

    # Calculate minutes since 9:30 AM market open
    bounces_enriched['minutes_since_open'] = (
        (bounces_enriched['datetime'] - bounces_enriched['market_open'])
        .dt.total_seconds() / 60
    ).astype(int)
    # ============================================================
        
    # volume ratio
    bounces_enriched['volume_ratio'] = (
        bounces_enriched['rolling_volume_5'] / bounces_enriched['day_avg_volume']
    ).fillna(1)
    
    # price momentum
    bounces_enriched['prior_5min_change'] = (
        (bounces_enriched['close'] - bounces_enriched['close_shift6']) / 
        bounces_enriched['close_shift6'] * 100
    ).fillna(0)
    
    # count prior bounces
    bounces_enriched = bounces_enriched.sort_values(['symbol', 'date', 'datetime'])
    bounces_enriched['num_prior_bounces'] = bounces_enriched.groupby(['symbol', 'date']).cumcount()
    
    # SPY features
    spy_df['spy_open'] = spy_df.groupby('date')['open'].transform('first')
    spy_df['spy_drop_from_open_pct'] = ((spy_df['close'] - spy_df['spy_open']) / spy_df['spy_open']) * 100
    
    spy_subset = spy_df[['datetime', 'spy_drop_from_open_pct']].sort_values('datetime')
    bounces_enriched = bounces_enriched.sort_values('datetime')
    
    bounces_enriched = pd.merge_asof(
        bounces_enriched,
        spy_subset,
        on='datetime',
        direction='backward'
    )
    
    bounces_enriched['spy_drop_from_open_pct'] = bounces_enriched['spy_drop_from_open_pct'].fillna(0)
    bounces_enriched['drop_vs_spy'] = (
        bounces_enriched['drop_from_open_pct'] - bounces_enriched['spy_drop_from_open_pct']
    )
    
    # select final columns (including new technical indicators)
    # features = bounces_enriched[[
    #     'datetime', 'symbol', 'drop_from_open_pct', 'bounce_from_low_pct',
    #     'minutes_since_open', 'volume_ratio', 'volume',
    #     'prior_5min_change', 'num_prior_bounces',
    #     'spy_drop_from_open_pct', 'drop_vs_spy',
    #     'rsi', 'macd_histogram'  # NEW
    # ]].copy()
    
    # features.columns = [
    #     'datetime', 'symbol', 'drop_from_open_pct', 'bounce_size_pct',
    #     'minutes_since_open', 'volume_ratio', 'bounce_volume',
    #     'prior_5min_change', 'num_prior_bounces',
    #     'spy_drop_from_open_pct', 'drop_vs_spy',
    #     'rsi', 'macd_histogram'  # NEW
    # ]
    
    features = bounces_enriched[[
        'datetime', 'symbol', 
        'drop_from_open_pct', 'bounce_from_low_pct',
        'minutes_since_open', 'volume_ratio',
        'num_prior_bounces', 'rsi'
    ]].copy()

    features.columns = [
        'datetime', 'symbol', 
        'drop_from_open_pct', 'bounce_size_pct',
        'minutes_since_open', 'volume_ratio',
        'num_prior_bounces', 'rsi'
    ]
    
    return features


# def analyze_features(features_df):
#     """
#     Compare feature distributions between Label 1 and Label 0
#     """
#     # drop unclear labels
#     df = features_df[features_df['label'].isin([0, 1])].copy()
    
#     label_1 = df[df['label'] == 1]
#     label_0 = df[df['label'] == 0]
    
#     print("\n" + "="*60)
#     print("FEATURE ANALYSIS: Label 1 vs Label 0")
#     print("="*60)
    
#     print(f"\nSample sizes:")
#     print(f"  Label 1 (real recovery): {len(label_1)} examples")
#     print(f"  Label 0 (fake-out): {len(label_0)} examples")
    
#     # ALL numeric features including new SPY ones
#     numeric_features = ['drop_from_open_pct', 'bounce_size_pct', 'minutes_since_open',
#                        'volume_ratio', 'prior_5min_change', 'num_prior_bounces',
#                        'spy_drop_from_open_pct', 'drop_vs_spy']  # NEW
    
#     print("\n" + "-"*60)
#     print("AVERAGE VALUES BY LABEL:")
#     print("-"*60)
#     print(f"{'Feature':<25} {'Label 1 (recovery)':<20} {'Label 0 (fake-out)':<20}")
#     print("-"*60)
    
#     for feat in numeric_features:
#         if feat in df.columns:  # check if feature exists
#             avg_1 = label_1[feat].mean()
#             avg_0 = label_0[feat].mean()
#             print(f"{feat:<25} {avg_1:<20.2f} {avg_0:<20.2f}")
    
#     print("\n" + "-"*60)
#     print("KEY INSIGHTS:")
#     print("-"*60)
    
#     # existing insights
#     if label_1['volume_ratio'].mean() > label_0['volume_ratio'].mean():
#         print("✓ Real recoveries have HIGHER volume ratios (more buying pressure)")
    
#     if label_1['num_prior_bounces'].mean() < label_0['num_prior_bounces'].mean():
#         print("✓ Real recoveries tend to happen EARLIER (fewer prior fake-outs)")
    
#     # NEW: Market context insights
#     if 'spy_drop_from_open_pct' in df.columns:
#         spy_1 = label_1['spy_drop_from_open_pct'].mean()
#         spy_0 = label_0['spy_drop_from_open_pct'].mean()
        
#         if spy_1 < spy_0:  # SPY down more during recoveries
#             print(f"✓ Real recoveries happen when SPY is also down ({spy_1:.2f}% vs {spy_0:.2f}%)")
#             print("  → Market-wide selloffs create better recovery opportunities")
#         else:
#             print(f"✗ SPY drop doesn't clearly distinguish recoveries")
    
#     if 'drop_vs_spy' in df.columns:
#         drop_vs_1 = label_1['drop_vs_spy'].mean()
#         drop_vs_0 = label_0['drop_vs_spy'].mean()
        
#         if abs(drop_vs_1) < abs(drop_vs_0):
#             print(f"✓ Real recoveries: stock drops WITH market (relative drop: {drop_vs_1:.2f}%)")
#             print(f"  Fake-outs: stock drops MORE than market (relative drop: {drop_vs_0:.2f}%)")
#             print("  → Stock-specific weakness is harder to recover from")
    
#     # New: Technical indicator insights
#     if 'rsi' in df.columns:
#         rsi_1 = label_1['rsi'].mean()
#         rsi_0 = label_0['rsi'].mean()
        
#         if rsi_1 < rsi_0:
#             print(f"✓ Real recoveries have LOWER RSI (more oversold: {rsi_1:.2f} vs {rsi_0:.2f})")
#             print("  → Oversold conditions favor stronger bounces")
    
#     if 'macd_diff' in df.columns: # macd_diff feature
#         macd_1 = label_1['macd_diff'].mean()
#         macd_0 = label_0['macd_diff'].mean()
        
#         if macd_1 > macd_0:
#             print(f"✓ Real recoveries have HIGHER MACD difference ({macd_1:.4f} vs {macd_0:.4f})")
#             print("  → Positive momentum supports successful bounces")
def analyze_features(features_df):
    """
    Compare feature distributions between Label 1 and Label 0
    """
    # drop unclear labels
    df = features_df[features_df['label'].isin([0, 1])].copy()
    
    label_1 = df[df['label'] == 1]
    label_0 = df[df['label'] == 0]
    
    print("\n" + "="*60)
    print("FEATURE ANALYSIS: Label 1 vs Label 0")
    print("="*60)
    
    print(f"\nSample sizes:")
    print(f"  Label 1 (real recovery): {len(label_1)} examples")
    print(f"  Label 0 (fake-out): {len(label_0)} examples")
    
    # ALL numeric features including new ones
    numeric_features = ['drop_from_open_pct', 'bounce_size_pct', 'minutes_since_open',
                       'volume_ratio', 'prior_5min_change', 'num_prior_bounces',
                       'spy_drop_from_open_pct', 'drop_vs_spy',
                       'rsi', 'macd_histogram']  # Added RSI and MACD
    
    print("\n" + "-"*60)
    print("AVERAGE VALUES BY LABEL:")
    print("-"*60)
    print(f"{'Feature':<25} {'Label 1 (recovery)':<20} {'Label 0 (fake-out)':<20}")
    print("-"*60)
    
    for feat in numeric_features:
        if feat in df.columns:
            avg_1 = label_1[feat].mean()
            avg_0 = label_0[feat].mean()
            print(f"{feat:<25} {avg_1:<20.2f} {avg_0:<20.2f}")
    
    print("\n" + "-"*60)
    print("KEY INSIGHTS:")
    print("-"*60)
    
    # existing insights
    if label_1['volume_ratio'].mean() > label_0['volume_ratio'].mean():
        print("✓ Real recoveries have HIGHER volume ratios (more buying pressure)")
    
    if label_1['num_prior_bounces'].mean() < label_0['num_prior_bounces'].mean():
        print("✓ Real recoveries tend to happen EARLIER (fewer prior fake-outs)")
    
    # SPY insights
    if 'spy_drop_from_open_pct' in df.columns:
        spy_1 = label_1['spy_drop_from_open_pct'].mean()
        spy_0 = label_0['spy_drop_from_open_pct'].mean()
        
        if spy_1 < spy_0:
            print(f"✓ Real recoveries happen when SPY is also down ({spy_1:.2f}% vs {spy_0:.2f}%)")
            print("  → Market-wide selloffs create better recovery opportunities")
    
    if 'drop_vs_spy' in df.columns:
        drop_vs_1 = label_1['drop_vs_spy'].mean()
        drop_vs_0 = label_0['drop_vs_spy'].mean()
        
        if abs(drop_vs_1) < abs(drop_vs_0):
            print(f"✓ Real recoveries: stock drops WITH market (relative drop: {drop_vs_1:.2f}%)")
            print(f"  Fake-outs: stock drops MORE than market (relative drop: {drop_vs_0:.2f}%)")
            print("  → Stock-specific weakness is harder to recover from")
    
    # NEW: Technical indicator insights
    if 'rsi' in df.columns:
        rsi_1 = label_1['rsi'].mean()
        rsi_0 = label_0['rsi'].mean()
        
        if rsi_1 < rsi_0:
            print(f"✓ Real recoveries have LOWER RSI ({rsi_1:.1f} vs {rsi_0:.1f})")
            print("  → More oversold = better recovery potential")
        else:
            print(f"✗ RSI doesn't clearly distinguish ({rsi_1:.1f} vs {rsi_0:.1f})")
    
    if 'macd_histogram' in df.columns:
        macd_1 = label_1['macd_histogram'].mean()
        macd_0 = label_0['macd_histogram'].mean()
        
        if macd_1 > macd_0:
            print(f"✓ Real recoveries have MORE positive momentum (MACD: {macd_1:.3f} vs {macd_0:.3f})")
        else:
            print(f"✗ MACD doesn't clearly distinguish ({macd_1:.3f} vs {macd_0:.3f})")            


if __name__ == "__main__":
    
    """load_stock_data tests"""
    # ---------------------
    # nvda = load_stock_data("NVDA")
    # print(f"Loaded {len(nvda)} rows of NVDA data")
    # print("\nFirst 5 rows:")
    # print(nvda.head())
    # print("\nColumn info:")
    # print(nvda.info())
    
    
    """plot_stock_intraday tests"""
    # ---------------------
    # nvda = load_stock_data("NVDA")
    # print(f"Loaded {len(nvda)} rows of NVDA data\n")
    
    # plot_stock_intraday(nvda, "NVDA")
    
    
    """plot_single_day tests"""
    # ---------------------
    # crwv = load_stock_data("CRWV")
    # plot_single_day(crwv, "CRWV", "2025-12-08")
    
    
    """calculate_daily_stats tests"""
    # ---------------------
    # nvda = load_stock_data("NVDA")
    
    # stats = calculate_daily_stats(nvda)
    # print("\nDaily Statistics:")
    # print(stats)
    
    # # find most volatile days
    # print("\nMost volatile days (by intraday range):")
    # print(stats.nlargest(3, 'intraday_range_pct')[['date', 'day_change_pct', 'intraday_range_pct', 'max_drawdown_pct']])
    
    """detect_bounces tests"""
    # ---------------------
    # nvda = load_stock_data("NVDA")
    
    # bounces = detect_bounces(nvda, min_drop_pct=2.0, min_bounce_pct=0.5)
    # print(f"\nDetected {len(bounces)} potential bounce moments")
    # print("\nSample bounces:")
    # print(bounces.head(10))
    
    """label_bounces tests"""
    # ---------------------
    # nvda = load_stock_data("NVDA")
    # bounces = detect_bounces(nvda, min_drop_pct=2.0, min_bounce_pct=0.5)
    
    # labeled = label_bounces(nvda, bounces)
    
    # print(f"\nLabeled {len(labeled)} bounces:")
    # print(labeled['label'].value_counts())
    # print("\nSample labeled bounces:")
    # print(labeled.head(10))
    
    # # save for later
    # labeled.to_csv(f"{DATA_DIR}/labeled_bounces.csv", index=False)
    # print(f"\nSaved labeled data to {DATA_DIR}/labeled_bounces.csv")
    
    """process_all_stocks tests"""
    # ---------------------
    # SYMBOLS = ["NVDA", "TSLA", "CRWV"]
    
    # all_bounces = process_all_stocks(SYMBOLS, min_drop_pct=2.0, min_bounce_pct=0.5)
    
    # print("\n" + "="*50)
    # print("COMBINED RESULTS (6 weeks):")
    # print("="*50)
    # print(f"\nTotal bounces: {len(all_bounces)}")
    # print(f"\nLabel distribution:")
    # print(all_bounces['label'].value_counts())
    
    # # calculate percentages
    # total = len(all_bounces)
    # for label in [1, 0, -1]:
    #     count = len(all_bounces[all_bounces['label'] == label])
    #     pct = (count/total)*100
    #     print(f"  Label {label}: {count} ({pct:.1f}%)")
    
    # all_bounces.to_csv(f"{DATA_DIR}/all_labeled_bounces.csv", index=False)
    
    """engineer_features tests"""
    # ---------------------
    import time
    
    # SYMBOLS = ["NVDA", "TSLA", "CRWV", "AMD", "COIN", "SHOP", 
    #            "JPM", "GS", "NKE", "DIS", "RIVN", "PLTR", 
    #            "BABA", "SNAP", "ROKU", "ZM", "DKNG", "ABNB",
    #            "AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    
    # SYMBOLS = [
    #     "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX",
    #     "AMD", "AVGO", "QCOM", "MU", "LRCX",
    #     "COIN", "PLTR", "RIVN", "CRWV",
    #     "SHOP", "CRM", "ADBE", "NOW", "SNOW",
    #     "JPM", "GS",
    # ]
    
    # SYMBOLS = [
    #     # ===== TIER 1: Elite Liquid Tech (Core Holdings) =====
    #     "NVDA",    # Volume: 350M, Range: 3-5%, Beta: 1.7
    #             # → King of bounces, extremely technical
        
    #     "TSLA",    # Volume: 130M, Range: 3-6%, Beta: 2.0
    #             # → Most volatile mega-cap, follows patterns
        
    #     "AMD",     # Volume: 90M, Range: 2-4%, Beta: 1.8
    #             # → Semiconductor, similar to NVDA
        
    #     "META",    # Volume: 18M, Range: 2-4%, Beta: 1.3
    #             # → Mega-cap with good volatility
        
    #     "AAPL",    # Volume: 60M, Range: 1-2%, Beta: 1.2
    #             # → Most liquid stock, reliable patterns
        
    #     "MSFT",    # Volume: 25M, Range: 1-2%, Beta: 1.1
    #             # → Second most liquid, stable patterns
        
    #     "GOOGL",   # Volume: 24M, Range: 2-3%, Beta: 1.2
    #             # → Quality mega-cap, technical
        
    #     "AMZN",    # Volume: 45M, Range: 2-3%, Beta: 1.3
    #             # → E-commerce leader, volatile enough
        
    #     # ===== TIER 2: High-Beta Growth (Volatility Kings) =====
    #     "COIN",    # Volume: 20M, Range: 4-8%, Beta: 3.5
    #             # → Crypto proxy, extreme volatility
        
    #     "PLTR",    # Volume: 70M, Range: 3-5%, Beta: 2.2
    #             # → High volume, very bouncy
        
    #     "RIVN",    # Volume: 40M, Range: 4-6%, Beta: 2.5
    #             # → EV sector, high volatility
        
    #     "CRWV",    # Volume: 8M, Range: 5-10%, Beta: 2.8
    #             # → Your requested stock, very volatile
        
    #     # ===== TIER 3: Quality Mid-Caps (Diversity) =====
    #     "SHOP",    # Volume: 12M, Range: 2-4%, Beta: 1.6
    #             # → E-commerce, technical patterns
        
    #     "CRM",     # Volume: 8M, Range: 2-3%, Beta: 1.2
    #             # → Enterprise SaaS, quality
        
    #     "SNOW",    # Volume: 6M, Range: 3-5%, Beta: 1.7
    #             # → Cloud computing, volatile
        
    #     # ===== TIER 4: Finance (Sector Diversity) =====
    #     "JPM",     # Volume: 12M, Range: 1-2%, Beta: 1.1
    #             # → Banking sector, most liquid bank
        
    #     "GS",      # Volume: 3M, Range: 1-3%, Beta: 1.3
    #             # → Investment banking, more volatile
    # ]
    
    # SYMBOLS = [
    #     # === TIER 1: High Contributors (Keep ALL of these) ===
    #     "CRWV",    # 13 Label 1s - 19% of dataset!
    #     "RIVN",    # 9 Label 1s
    #     "COIN",    # 7 Label 1s
    #     "AMD",     # 5 Label 1s
    #     "ROKU",    # 5 Label 1s - ERROR to remove!
    #     "PLTR",    # 5 Label 1s
    #     "SHOP",    # 4 Label 1s
    #     "TSLA",    # 3 Label 1s
    #     "NVDA",    # 3 Label 1s
    #     "GOOGL",   # 3 Label 1s
        
    #     # === TIER 2: Medium Contributors ===
    #     "DKNG",    # 2 Label 1s - needed!
    #     "GS",      # 2 Label 1s
    #     "META",    # 1 Label 1
    # ]

    SYMBOLS = [
        "NVDA", "GOOGL", "AAPL"]

    # load SPY
    spy = load_stock_data("SPY")
    
    # load all stock data
    all_stock_data = {}
    for symbol in SYMBOLS:
        all_stock_data[symbol] = load_stock_data(symbol)
    
    # process each stock with output
    all_labeled_list = []
    
    for symbol in SYMBOLS:
        print(f"\nProcessing {symbol}...")
        
        # detect bounces
        bounces = detect_bounces(all_stock_data[symbol], min_drop_pct=2.0, min_bounce_pct=0.5)
        print(f"  Found {len(bounces)} bounces")
        
        if len(bounces) == 0:
            continue
        
        # label bounces
        labeled = label_bounces(all_stock_data[symbol], bounces)
        
        # show label distribution
        print(f"  Label distribution:")
        label_counts = labeled['label'].value_counts().to_dict()
        print(label_counts)
        
        all_labeled_list.append(labeled)
    
    # combine all labeled bounces
    all_labeled = pd.concat(all_labeled_list, ignore_index=True)
    
    # engineer features (vectorized - all at once for speed)
    print("\nEngineering features...")
    start = time.time()
    feature_dfs = []
    
    for symbol in SYMBOLS:
        symbol_labeled = all_labeled[all_labeled['symbol'] == symbol]
        if len(symbol_labeled) > 0:
            features = engineer_features(all_stock_data[symbol], symbol_labeled, spy)
            feature_dfs.append(features)
    
    all_features = pd.concat(feature_dfs, ignore_index=True)
    
    # merge labels back in
    all_features = all_features.merge(
        all_labeled[['datetime', 'symbol', 'label']],
        on=['datetime', 'symbol'],
        how='left'
    )
    
    print(f"✓ Created {len(all_features)} feature rows ({time.time()-start:.1f}s)\n")
    
    # save
    all_features.to_csv(f"{DATA_DIR}/features.csv", index=False)
    print(f"Saved to {DATA_DIR}/features.csv")
    
    # analyze
    analyze_features(all_features)