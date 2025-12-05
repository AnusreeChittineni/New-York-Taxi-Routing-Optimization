"""
Extract a few examples of training and validation data to CSV files for inspection.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataconnection.duckdb_connector import connect_duckdb, load_trip_data

def extract_samples(
    db_path: str = "data/nyc_traffic_2016.duckdb",
    output_dir: str = "data/samples_preview",
):
    """
    Query DuckDB for trip samples and save the first 10 rows to CSV.
    Splits them into 'train' and 'val' examples for inspection.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to DuckDB at {db_path}...")
    try:
        conn = connect_duckdb(db_path)
        
        # Load 10 rows
        print("Querying for 10 sample trips...")
        df = load_trip_data(conn, limit=10)
        
        if df.empty:
            print("No data found in DuckDB table 'taxi_data'.")
            return

        # Split into train/val examples
        train_head = df.iloc[:5]
        val_head = df.iloc[5:10]
        
        output_train = f"{output_dir}/train_examples_head.csv"
        train_head.to_csv(output_train, index=False)
        print(f"Saved 5 training examples to {output_train}")
        print(train_head.to_csv(index=False))
        
        print("-" * 40)
        
        output_val = f"{output_dir}/val_examples_head.csv"
        val_head.to_csv(output_val, index=False)
        print(f"Saved 5 validation examples to {output_val}")
        print(val_head.to_csv(index=False))
        
    except Exception as e:
        print(f"Error querying DuckDB: {e}")
        print("Ensure the database file exists and is accessible.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/nyc_traffic_2016.duckdb")
    parser.add_argument("--output", default="data/samples_preview")
    args = parser.parse_args()
    
    extract_samples(args.db, args.output)
