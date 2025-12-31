from __future__ import annotations
import json
from datetime import datetime, timedelta
from typing import List
import pandas as pd
from rich.console import Console
from schwab import auth, client as schwab_client
import os
from dotenv import load_dotenv

console = Console()

# -------------------------------------------------------------------------
# Credentials
# -------------------------------------------------------------------------
load_dotenv()

APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
CALLBACK_URL = os.getenv("CALLBACK_URL")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
TOKEN_PATH = os.getenv("TOKEN_PATH")

# -------------------------------------------------------------------------
# Stocks to collect data for
# -------------------------------------------------------------------------
SYMBOLS = ["NVDA", "TSLA", "CRWV", "AMD", "COIN", "SHOP", 
           "JPM", "GS", "NKE", "DIS", "RIVN", "PLTR", 
           "BABA", "SNAP", "ROKU", "ZM", "DKNG", "ABNB",
           "AAPL", "MSFT", "AMZN", "GOOGL", "META",
           "SPY"]

# SYMBOLS = [
#     # Core mega-cap tech (liquid, volatile, follows market)
#     "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX",
    
#     # High-quality semis (similar to NVDA)
#     "AMD", "AVGO", "QCOM", "MU", "LRCX",
    
#     # High-beta growth (volatile, liquid)
#     "COIN", "PLTR", "RIVN", "CRWV",
    
#     # Quality SaaS (technical, liquid)
#     "SHOP", "CRM", "ADBE", "NOW", "SNOW",
    
#     # Finance (different sector, still technical)
#     "JPM", "GS",
    
#     # Market index
#     "SPY"
# ]

OUTPUT_DIR = "./data"

# -------------------------------------------------------------------------
# Data Collection
# -------------------------------------------------------------------------
def get_schwab_client():
    """Get authenticated Schwab client, reusing token if available"""
    import os.path
    
    # Check if token file exists
    if TOKEN_PATH and os.path.isfile(TOKEN_PATH):
        console.print("[cyan]🔐 Using existing token...[/]")
        try:
            # Try to use existing token
            client = auth.client_from_token_file(
                token_path=TOKEN_PATH,
                api_key=APP_KEY,
                app_secret=APP_SECRET,
            )
            console.print("[green]✅ Authenticated with existing token[/]")
            return client
        except Exception as e:
            console.print(f"[yellow]⚠️  Existing token invalid: {e}[/]")
            console.print("[cyan]Creating new token...[/]")
    
    # If no token or token invalid, do manual flow
    client = auth.client_from_manual_flow(
        api_key=APP_KEY,
        app_secret=APP_SECRET,
        callback_url=CALLBACK_URL,
        token_path=TOKEN_PATH,
    )
    console.print("[green]✅ New token created[/]")
    return client


def collect_historical_data():
    """Pull minute-level data for target stocks"""
    
    console.print("[cyan]🔐 Authenticating with Schwab...[/]")
    
    try:
        client = get_schwab_client()
        console.print("[green]✅ Authenticated successfully[/]\n")
        
        
    except Exception as e:
        console.print(f"[red]❌ Authentication failed: {e}[/]")
        return
    
    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect data for each symbol
    for symbol in SYMBOLS:
        console.print(f"[cyan]📊 Fetching data for {symbol}...[/]")
        
        try:
            # Get last 10 trading days of minute-level data
            # Schwab API: period_type='day', period=10 gives ~2 weeks
            response = client.get_price_history_every_minute(
                symbol=symbol,
                start_datetime=datetime.now() - timedelta(days=180),
                end_datetime=datetime.now(),
                need_extended_hours_data=False
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Parse into DataFrame
            if 'candles' in data and data['candles']:
                df = pd.DataFrame(data['candles'])
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Reorder columns
                df = df[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                
                # Save to CSV
                filename = f"{OUTPUT_DIR}/{symbol}_minute_data.csv"
                df.to_csv(filename, index=False)
                
                console.print(f"[green]✅ {symbol}: {len(df)} bars saved to {filename}[/]")
                console.print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                
            else:
                console.print(f"[yellow]⚠️  {symbol}: No data returned[/]")
                
        except Exception as e:
            console.print(f"[red]❌ {symbol}: Error - {e}[/]")
    
    console.print("\n[green]✅ Data collection complete![/]")

if __name__ == "__main__":
    collect_historical_data()