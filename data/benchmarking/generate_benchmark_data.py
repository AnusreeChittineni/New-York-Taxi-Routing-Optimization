import duckdb
import pandas as pd
from io import StringIO
import requests
import time
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DB_PATH = "nyc_routing_benchmark.duckdb"
OUTPUT_DIR = "bench"
CHUNK_SIZE = 50_000
API_URL = "https://data.cityofnewyork.us/api/views/uacg-pexx/rows.csv?accessType=DOWNLOAD"
APP_TOKEN = "FvrjNyrm0p2bfhsooy9kNZ7ib"

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# CONNECT TO DUCKDB
# ------------------------------------------------------------
con = duckdb.connect(DB_PATH)
print(f"Connected to DuckDB at {DB_PATH}")

# ------------------------------------------------------------
# HELPER FUNCTION TO FLUSH BUFFER
# ------------------------------------------------------------
def flush(buf, header, first):
    csv_data = header + "\n" + "\n".join(buf)
    df = pd.read_csv(StringIO(csv_data))
    if first:
        con.execute("DROP TABLE IF EXISTS taxi_raw;")
        con.register("chunk", df)
        con.execute("CREATE TABLE taxi_raw AS SELECT * FROM chunk;")
        con.unregister("chunk")
        return False
    else:
        con.register("chunk", df)
        con.execute("INSERT INTO taxi_raw SELECT * FROM chunk;")
        con.unregister("chunk")
        return False

# ------------------------------------------------------------
# 1. STREAM AND LOAD TAXI DATA (RESUME-FRIENDLY)
# ------------------------------------------------------------
def load_taxi():
    # Check if table already exists
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "taxi_raw" in tables:
        row_count = con.execute("SELECT COUNT(*) FROM taxi_raw").fetchone()[0]
        print(f"taxi_raw already exists with {row_count:,} rows — skipping full download.")
        return

    print("\n[1/3] Loading Yellow Taxi Trip Data (2016)…")
    start_time = time.time()

    first = True
    buffer = []
    rows = 0
    headers = {"X-App-Token": APP_TOKEN}

    with requests.get(API_URL, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        lines = r.iter_lines(decode_unicode=True)
        header = next(lines)

        for line in lines:
            if not line:
                continue
            buffer.append(line)
            if len(buffer) >= CHUNK_SIZE:
                first = flush(buffer, header, first)
                rows += len(buffer)
                buffer.clear()

        # Flush remaining rows
        if buffer:
            first = flush(buffer, header, first)
            rows += len(buffer)

    elapsed = time.time() - start_time
    print(f"Completed ingestion: {rows:,} rows in {elapsed:.2f} sec")

# ------------------------------------------------------------
# 2. CREATE PARQUET BENCHMARK SUBSETS (ALWAYS OVERWRITE)
# ------------------------------------------------------------
def create_subset(limit, output_path):
    print(f"\nCreating subset of {limit:,} rows → {output_path}")
    start = time.time()

    subset_table = f"trips_{limit}"
    con.execute(f"DROP TABLE IF EXISTS {subset_table};")
    con.execute(f"""
        CREATE TABLE {subset_table} AS
        SELECT *
        FROM taxi_raw
        LIMIT {limit};
    """)
    con.execute(f"""
        COPY {subset_table}
        TO '{output_path}'
        (FORMAT PARQUET);
    """)
    elapsed = time.time() - start
    print(f"Finished {limit:,} rows → {output_path} in {elapsed:.2f} sec")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    load_taxi()
    create_subset(1_000_000, f"{OUTPUT_DIR}/trips_1M.parquet")
    create_subset(5_000_000, f"{OUTPUT_DIR}/trips_5M.parquet")
    print("\nAll benchmark Parquet subsets created successfully!")
