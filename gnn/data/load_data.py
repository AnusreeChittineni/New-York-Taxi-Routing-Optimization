import duckdb
import pandas as pd
from sodapy import Socrata
import requests
import time
from io import StringIO

# Database setup
DB_PATH = "nyc_traffic_2016.duckdb"
conn = duckdb.connect(DB_PATH)
print(f"Connected to {DB_PATH}")

client = Socrata("data.cityofnewyork.us", app_token="FvrjNyrm0p2bfhsooy9kNZ7ib", timeout=60)

# File paths (local)
traffic_csv = "data/Automated_Traffic_Volume_Counts_20251102.csv"
collisions_csv = "data/Motor_Vehicle_Collisions_-_Crashes_20251102.csv"

# ================================================================
# Load taxi data (2016 only) — streamed directly from NYC Open Data
# ================================================================

API_URL = "https://data.cityofnewyork.us/api/views/uacg-pexx/rows.csv?accessType=DOWNLOAD"
APP_TOKEN = "FvrjNyrm0p2bfhsooy9kNZ7ib"

CHUNK_SIZE = 50_000  # rows per insert
MAX_RETRIES = 5

headers = {"X-App-Token": APP_TOKEN}
row_buffer = []
rows_inserted = 0
first_chunk = True
retries = 0

print("\nLoading 2016 Yellow Taxi Trip Data directly from NYC Open Data...")


def flush_buffer(buf, header_line, first_chunk):
    """Append buffered CSV rows to DuckDB."""
    global rows_inserted
    if not buf:
        return first_chunk

    # Combine header + buffered data into a temporary CSV
    csv_data = header_line + "\n" + "\n".join(buf)
    df_chunk = pd.read_csv(StringIO(csv_data))

    if df_chunk.empty:
        print("Empty chunk skipped.")
        buf.clear()
        return first_chunk

    if first_chunk:
        conn.execute("DROP TABLE IF EXISTS taxi_data;")
        conn.register("df_chunk", df_chunk)
        conn.execute("CREATE TABLE taxi_data AS SELECT * FROM df_chunk;")
        conn.unregister("df_chunk")
        print(f"Created taxi_data table with {len(df_chunk.columns)} columns.")
        first_chunk = False
    else:
        conn.register("df_chunk", df_chunk)
        conn.execute("INSERT INTO taxi_data SELECT * FROM df_chunk;")
        conn.unregister("df_chunk")

    rows_inserted += len(df_chunk)
    print(f"Inserted total {rows_inserted:,} rows so far.")
    buf.clear()
    return first_chunk


while True:
    try:
        with requests.get(API_URL, headers=headers, stream=True, timeout=180) as r:
            r.raise_for_status()
            lines = r.iter_lines(decode_unicode=True)

            # Extract header once
            header_line = next(lines)
            print("Header:", header_line[:100], "...")

            for line in lines:
                if not line:
                    continue
                row_buffer.append(line)
                if len(row_buffer) >= CHUNK_SIZE:
                    first_chunk = flush_buffer(row_buffer, header_line, first_chunk)
                    row_buffer = []

            # Flush final partial chunk
            if row_buffer:
                first_chunk = flush_buffer(row_buffer, header_line, first_chunk)

        print("Completed full taxi data stream.")
        break

    except requests.exceptions.ChunkedEncodingError:
        retries += 1
        print(f"\nConnection dropped — retry {retries}/{MAX_RETRIES}")
        if retries >= MAX_RETRIES:
            print("Too many dropouts, aborting stream.")
            break
        print("Sleeping 10 seconds, then resuming...")
        time.sleep(10)
        continue

print(f"\nFinal count: {rows_inserted:,} rows inserted into taxi_data.\n")

# ================================================================
# Load Traffic Volume Counts (filter only 2016)
# ================================================================
print("Loading 2016 Traffic Volume Counts...")

df_sample = pd.read_csv(traffic_csv, nrows=5)
print("Traffic columns:", list(df_sample.columns))

conn.execute(
    f"""
CREATE TABLE IF NOT EXISTS traffic_2016 AS
SELECT *
FROM read_csv_auto('{traffic_csv}')
WHERE "Yr" = 2016;
"""
)

print("traffic_2016 table created successfully.")
print(conn.execute("SELECT COUNT(*) FROM traffic_2016;").fetchone(), "rows loaded.\n")

# ================================================================
# Load Motor Vehicle Collisions (filter only 2016)
# ================================================================
print("Loading 2016 Motor Vehicle Collisions data...")

df_sample = pd.read_csv(collisions_csv, nrows=5)
print("Collision columns:", list(df_sample.columns))

conn.execute(
    f"""
CREATE TABLE IF NOT EXISTS collisions_2016 AS
SELECT *,
       STRPTIME(CAST("CRASH DATE" AS VARCHAR) || ' ' || CAST("CRASH TIME" AS VARCHAR),
                '%Y-%m-%d %H:%M:%S') AS crash_datetime
FROM read_csv_auto(
    '{collisions_csv}',
    types={{'ZIP CODE': 'VARCHAR'}},
    nullstr=''
)
WHERE EXTRACT(year FROM STRPTIME(CAST("CRASH DATE" AS VARCHAR), '%Y-%m-%d')) = 2016;
"""
)

print("collisions_2016 table created successfully.")
print(conn.execute("SELECT COUNT(*) FROM collisions_2016;").fetchone(), "rows loaded.\n")

# ================================================================
# Show summary + preview
# ================================================================
print("\nTables in database:")
print(conn.execute("SHOW TABLES;").fetchdf())

print("\nSample from taxi_data:")
try:
    print(conn.execute("SELECT * FROM taxi_data LIMIT 5;").fetchdf())
except Exception as e:
    print("Could not sample taxi_data yet:", e)
