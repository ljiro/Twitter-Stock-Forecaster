import pandas as pd
import os

FILE_PATH = "data/live_predictions.csv"
BACKUP_PATH = "data/live_predictions_backup.csv"

def clean_csv():
    if not os.path.exists(FILE_PATH):
        print("❌ File not found.")
        return

    print("🧹 Cleaning CSV Schema...")
    
    # 1. Read raw lines to handle variable column counts
    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()

    cleaned_rows = []
    
    # Define standard header
    header = ["timestamp", "price", "sentiment", "gb_pred"]
    
    for i, line in enumerate(lines):
        if i == 0: continue # Skip old header
        
        parts = line.strip().split(',')
        
        # Handle mixed formats (assuming comma delimiter)
        # If parts is small (likely 1 value), skip
        if len(parts) < 2: continue
        
        row_dict = {}
        
        # Heuristic to detect row type
        try:
            # Case A: Old Format (5+ cols) -> [Time, Sent, Price, 1d, 3d]
            if len(parts) >= 5:
                row_dict['timestamp'] = parts[0]
                row_dict['sentiment'] = parts[1]
                row_dict['price'] = parts[2]
                row_dict['gb_pred'] = parts[4] # Use 3d pred
            
            # Case B: New Format (3 cols) -> [Time, Price, Pred]
            elif len(parts) == 3:
                row_dict['timestamp'] = parts[0]
                row_dict['price'] = parts[1]
                row_dict['gb_pred'] = parts[2]
                row_dict['sentiment'] = 0.0 # Default missing
                
            # Case C: Already Fixed (4 cols)
            elif len(parts) == 4:
                row_dict['timestamp'] = parts[0]
                row_dict['price'] = parts[1]
                row_dict['sentiment'] = parts[2]
                row_dict['gb_pred'] = parts[3]
                
            cleaned_rows.append(row_dict)
            
        except Exception as e:
            print(f"⚠️ Skipping bad row {i}: {line.strip()} - {e}")

    # 2. Create DataFrame and standardise dates
    if not cleaned_rows:
        print("⚠️ No valid data found. Creating fresh file.")
        df = pd.DataFrame(columns=header)
    else:
        df = pd.DataFrame(cleaned_rows)
        # Fix Date Formats to ISO (YYYY-MM-DD HH:MM:SS)
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True).dt.strftime('%Y-%m-%d %H:%M:%S')
        # Ensure numerics
        cols = ['price', 'sentiment', 'gb_pred']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')

    # 3. Save
    os.rename(FILE_PATH, BACKUP_PATH)
    df[header].to_csv(FILE_PATH, index=False)
    print(f"✅ Fixed! Saved {len(df)} rows to {FILE_PATH}")
    print(f"   (Old file backed up to {BACKUP_PATH})")

if __name__ == "__main__":
    clean_csv()