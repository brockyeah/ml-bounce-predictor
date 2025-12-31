# for downloading flat files from Massive S3
import os
import subprocess
from datetime import datetime, timedelta
import pandas as pd
import gzip
import shutil
from rich.console import Console

console = Console()

OUTPUT_DIR = "./data"
TEMP_DIR = "./data/temp"

SYMBOLS = [
    "NVDA", "TSLA", "CRWV", "AMD", "COIN", "SHOP",
    "RIVN", "PLTR", "ROKU", "DKNG", "GS", "GOOGL", "META", "SPY"
]

def download_flat_files(start_date, end_date):
    """
    Download minute aggregate flat files from Massive S3
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    console.print(f"[cyan]Downloading data from {start_date} to {end_date}...[/]\n")
    
    current_date = start_date
    downloaded_files = []
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        year = current_date.strftime("%Y")
        month = current_date.strftime("%m")
        date_str = current_date.strftime("%Y-%m-%d")
        
        # S3 path
        s3_path = f"s3://flatfiles/us_stocks_sip/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz"
        local_path = f"{TEMP_DIR}/{date_str}.csv.gz"
        
        console.print(f"[cyan]Downloading {date_str}...[/]")
        
        try:
            # Download using AWS CLI
            cmd = [
                "aws", "s3", "cp",
                s3_path,
                local_path,
                "--endpoint-url=https://files.massive.com"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                downloaded_files.append(local_path)
                console.print(f"[green]✅ Downloaded {date_str}[/]")
            else:
                console.print(f"[yellow]⚠️  No data for {date_str} (likely holiday/weekend)[/]")
        
        except Exception as e:
            console.print(f"[red]❌ Error downloading {date_str}: {e}[/]")
        
        current_date += timedelta(days=1)
    
    console.print(f"\n[green]Downloaded {len(downloaded_files)} files[/]\n")
    
    # Process and filter for our symbols
    console.print("[cyan]Processing files...[/]")
    process_files(downloaded_files)
    
    # Cleanup
    console.print("[cyan]Cleaning up temp files...[/]")
    shutil.rmtree(TEMP_DIR)
    
    console.print("[green]✅ Complete![/]")

def process_files(files):
    """
    Extract and combine data for our specific symbols
    """
    # Dictionary to store data for each symbol
    symbol_data = {symbol: [] for symbol in SYMBOLS}
    
    for file_path in files:
        console.print(f"  Processing {os.path.basename(file_path)}...")
        
        try:
            # Decompress and read
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f)
            
            # Filter for our symbols
            df = df[df['ticker'].isin(SYMBOLS)]
            
            if len(df) == 0:
                continue
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['window_start'], unit='ns')
            df = df.rename(columns={'ticker': 'symbol', 'window_start': 'timestamp'})
            df = df[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Group by symbol
            for symbol in SYMBOLS:
                symbol_df = df[df['symbol'] == symbol]
                if len(symbol_df) > 0:
                    symbol_data[symbol].append(symbol_df)
        
        except Exception as e:
            console.print(f"[red]  Error processing {file_path}: {e}[/]")
    
    # Combine and save each symbol
    for symbol in SYMBOLS:
        if symbol_data[symbol]:
            combined = pd.concat(symbol_data[symbol], ignore_index=True)
            combined = combined.sort_values('datetime')
            
            # Convert to market hours only (9:30 AM - 4:00 PM EST)
            combined['datetime'] = pd.to_datetime(combined['datetime'])
            combined = combined[
                (combined['datetime'].dt.hour >= 9) & 
                ((combined['datetime'].dt.hour < 16) | 
                 ((combined['datetime'].dt.hour == 9) & (combined['datetime'].dt.minute >= 30)))
            ]
            
            filepath = f"{OUTPUT_DIR}/{symbol}_minute_data.csv"
            combined.to_csv(filepath, index=False)
            
            console.print(f"[green]  ✅ {symbol}: {len(combined):,} bars saved[/]")
            console.print(f"     Range: {combined['datetime'].min()} to {combined['datetime'].max()}")

if __name__ == "__main__":
    # Download 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    download_flat_files(start_date, end_date)