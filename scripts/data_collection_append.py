import os
from datetime import datetime, timedelta
import pandas as pd
from rich.console import Console
from schwab import auth
from schwab.client import Client
from dotenv import load_dotenv

load_dotenv()

console = Console()

# Credentials
APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
CALLBACK_URL = os.getenv("CALLBACK_URL")
TOKEN_PATH = os.getenv("TOKEN_PATH", "./token.json")

# SYMBOLS = ["NVDA", "TSLA", "CRWV", "AMD", "COIN", "SHOP", 
#            "JPM", "GS", "NKE", "DIS", "RIVN", "PLTR", 
#            "BABA", "SNAP", "ROKU", "ZM", "DKNG", "ABNB",
#            "AAPL", "MSFT", "AMZN", "GOOGL", "META",
#            "SPY"]

# SYMBOLS = [
#     "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX",
#     "AMD", "AVGO", "QCOM", "MU", "LRCX",
#     "COIN", "PLTR", "RIVN", 
#     "SHOP", "CRM", "ADBE", "NOW", "SNOW",
#     "JPM", "GS",
#     "SPY"
# ]

SYMBOLS = [
    # ===== TIER 1: Elite Liquid Tech (Core Holdings) =====
    "NVDA",    # Volume: 350M, Range: 3-5%, Beta: 1.7
               # → King of bounces, extremely technical
    
    "TSLA",    # Volume: 130M, Range: 3-6%, Beta: 2.0
               # → Most volatile mega-cap, follows patterns
    
    "AMD",     # Volume: 90M, Range: 2-4%, Beta: 1.8
               # → Semiconductor, similar to NVDA
    
    "META",    # Volume: 18M, Range: 2-4%, Beta: 1.3
               # → Mega-cap with good volatility
    
    "AAPL",    # Volume: 60M, Range: 1-2%, Beta: 1.2
               # → Most liquid stock, reliable patterns
    
    "MSFT",    # Volume: 25M, Range: 1-2%, Beta: 1.1
               # → Second most liquid, stable patterns
    
    "GOOGL",   # Volume: 24M, Range: 2-3%, Beta: 1.2
               # → Quality mega-cap, technical
    
    "AMZN",    # Volume: 45M, Range: 2-3%, Beta: 1.3
               # → E-commerce leader, volatile enough
    
    # ===== TIER 2: High-Beta Growth (Volatility Kings) =====
    "COIN",    # Volume: 20M, Range: 4-8%, Beta: 3.5
               # → Crypto proxy, extreme volatility
    
    "PLTR",    # Volume: 70M, Range: 3-5%, Beta: 2.2
               # → High volume, very bouncy
    
    "RIVN",    # Volume: 40M, Range: 4-6%, Beta: 2.5
               # → EV sector, high volatility
    
    "CRWV",    # Volume: 8M, Range: 5-10%, Beta: 2.8
               # → Your requested stock, very volatile
    
    # ===== TIER 3: Quality Mid-Caps (Diversity) =====
    "SHOP",    # Volume: 12M, Range: 2-4%, Beta: 1.6
               # → E-commerce, technical patterns
    
    "CRM",     # Volume: 8M, Range: 2-3%, Beta: 1.2
               # → Enterprise SaaS, quality
    
    "SNOW",    # Volume: 6M, Range: 3-5%, Beta: 1.7
               # → Cloud computing, volatile
    
    # ===== TIER 4: Finance (Sector Diversity) =====
    "JPM",     # Volume: 12M, Range: 1-2%, Beta: 1.1
               # → Banking sector, most liquid bank
    
    "GS",      # Volume: 3M, Range: 1-3%, Beta: 1.3
               # → Investment banking, more volatile
    
    # ===== TIER 5: Market Benchmark =====
    "SPY"      # Volume: 80M, Range: 0.5-1.5%, Beta: 1.0
               # → For drop_vs_spy feature
]

OUTPUT_DIR = "./data"

def get_schwab_client():
    """Get authenticated Schwab client"""
    if os.path.isfile(TOKEN_PATH):
        try:
            client = auth.client_from_token_file(
                token_path=TOKEN_PATH,
                api_key=APP_KEY,
                app_secret=APP_SECRET,
            )
            return client
        except Exception:
            pass
    
    return auth.client_from_manual_flow(
        api_key=APP_KEY,
        app_secret=APP_SECRET,
        callback_url=CALLBACK_URL,
        token_path=TOKEN_PATH,
    )

def append_new_data():
    """Append new minute data to existing CSVs"""
    
    console.print("[cyan]🔐 Authenticating...[/]")
    client = get_schwab_client()
    console.print("[green]✅ Authenticated[/]\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for symbol in SYMBOLS:
        console.print(f"[cyan]📊 Processing {symbol}...[/]")
        
        filepath = f"{OUTPUT_DIR}/{symbol}_minute_data.csv"
        
        try:
            # Check if file exists and get latest date
            existing_df = None
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath, parse_dates=['datetime'])
                latest_date = existing_df['datetime'].max()
                console.print(f"  Existing data until: {latest_date}")
                
                # Pull from latest date + 1 minute to now
                start_date = latest_date + timedelta(minutes=1)
            else:
                console.print(f"  No existing data, pulling 45 days")
                start_date = datetime.now() - timedelta(days=45)
            
            end_date = datetime.now()
            
            # Fetch new data
            response = client.get_price_history(
                symbol=symbol,
                period_type=Client.PriceHistory.PeriodType.DAY,
                frequency_type=Client.PriceHistory.FrequencyType.MINUTE,
                frequency=Client.PriceHistory.Frequency.EVERY_MINUTE,
                start_datetime=start_date,
                end_datetime=end_date,
                need_extended_hours_data=False
            )
            
            response.raise_for_status()
            data = response.json()
            
            if 'candles' in data and data['candles']:
                new_df = pd.DataFrame(data['candles'])
                new_df['datetime'] = pd.to_datetime(new_df['datetime'], unit='ms')
                new_df['symbol'] = symbol
                new_df = new_df[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                
                # Append or create
                if existing_df is not None:
                    # Combine and remove duplicates
                    combined_df = pd.concat([existing_df, new_df])
                    combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                    combined_df = combined_df.sort_values('datetime')
                else:
                    combined_df = new_df
                
                # Save
                combined_df.to_csv(filepath, index=False)
                
                new_bars = len(new_df)
                total_bars = len(combined_df)
                date_range = f"{combined_df['datetime'].min()} to {combined_df['datetime'].max()}"
                
                console.print(f"[green]✅ Added {new_bars} new bars (total: {total_bars})[/]")
                console.print(f"   Full range: {date_range}\n")
            else:
                console.print(f"[yellow]⚠️  No new data available\n")
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}\n")
    
    console.print("[green]✅ Data append complete![/]")

if __name__ == "__main__":
    append_new_data()