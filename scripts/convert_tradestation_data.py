import pandas as pd
from datetime import datetime
import os
from rich.console import Console

console = Console()

INPUT_DIR = "./tradestation_data"
OUTPUT_DIR = "./data"

def parse_tradestation_csv(filepath, symbol):
    """
    Convert TradeStation format to standard format
    
    Date format: 1YYMMDD (e.g., 1201223 = Dec 23, 2020)
    Time format: HHMM (e.g., 932 = 09:32)
    """
    console.print(f"[cyan]Processing {symbol}...[/]")
    
    records = []
    malformed = 0
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                parts = line.strip().split(',')
                
                if len(parts) < 6:
                    malformed += 1
                    continue
                
                # Parse date: 1YYMMDD
                date_str = parts[0].strip()
                
                if len(date_str) != 7 or date_str[0] != '1':
                    malformed += 1
                    continue
                
                yy = int(date_str[1:3])  # Year: 20, 23, 25, etc.
                month = int(date_str[3:5])
                day = int(date_str[5:7])
                
                # Convert YY to full year
                year = 2000 + yy
                
                # Parse time: HHMM (pad to 4 digits)
                time_str = parts[1].strip().zfill(4)
                hour = int(time_str[0:2])
                minute = int(time_str[2:4])
                
                # Validate
                if not (1 <= month <= 12 and 1 <= day <= 31):
                    malformed += 1
                    continue
                
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    malformed += 1
                    continue
                
                # Filter to market hours during parsing (faster)
                if hour < 9 or hour >= 16:
                    continue
                if hour == 9 and minute < 30:
                    continue
                
                # Create datetime
                dt = datetime(year, month, day, hour, minute)
                
                # Parse OHLC
                open_price = float(parts[2])
                high = float(parts[3])
                low = float(parts[4])
                close = float(parts[5])
                
                # Calculate total volume (columns 6 + 7: up_volume + down_volume)
                if len(parts) >= 8:
                    volume_up = int(parts[6])
                    volume_down = int(parts[7])
                    volume = volume_up + volume_down
                else:
                    volume = 0
                
                records.append({
                    'datetime': dt,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
                
            except Exception as e:
                malformed += 1
                if malformed <= 3:  # Show first few errors for debugging
                    console.print(f"[yellow]  Row {i+1} error: {e}[/]")
                continue
    
    if not records:
        console.print(f"[red]❌ {symbol}: No valid records[/]")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Ensure datetime is proper datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate stats
    total_days = (df['datetime'].max() - df['datetime'].min()).days
    years = total_days / 365.25
    
    console.print(f"[green]✅ {symbol}:[/]")
    console.print(f"   Total bars: {len(df):,}")
    console.print(f"   Malformed rows: {malformed:,}")
    console.print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    console.print(f"   Duration: {years:.1f} years ({total_days} days)")
    console.print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}\n")
    
    return df

def convert_all():
    """
    Convert all TradeStation CSV files
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Map files to symbols
    files = {
        'NVDA_1m.csv': 'NVDA',
        'GOOG_1m.csv': 'GOOGL',
        'AAPL_1m.csv': 'AAPL'
    }
    
    for filename, symbol in files.items():
        filepath = f"{INPUT_DIR}/{filename}"
        
        if not os.path.exists(filepath):
            console.print(f"[yellow]⚠️  {filename} not found, skipping[/]")
            continue
        
        console.print(f"\n[cyan]{'='*60}[/]")
        df = parse_tradestation_csv(filepath, symbol)
        
        if df is not None:
            # Save in our standard format
            output_path = f"{OUTPUT_DIR}/{symbol}_minute_data.csv"
            df.to_csv(output_path, index=False)
            console.print(f"[green]💾 Saved to {output_path}[/]")

if __name__ == "__main__":
    console.print("\n[cyan]🔄 Converting TradeStation Data...[/]")
    console.print("[cyan]Date format: 1YYMMDD (e.g., 1201223 = Dec 23, 2020)[/]")
    console.print("[cyan]Time format: HHMM (e.g., 932 = 09:32 AM)[/]\n")
    
    convert_all()
    
    console.print("\n[green]✅ Conversion complete![/]")
    console.print("\n[cyan]Next steps:[/]")
    console.print("  1. Verify dates look correct in data/*.csv")
    console.print("  2. Run: python scripts/eda_visualization.py")
    console.print("  3. Run: python scripts/train_model.py")