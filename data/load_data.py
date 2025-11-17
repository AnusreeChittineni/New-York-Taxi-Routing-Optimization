import duckdb
import pandas as pd
from sodapy import Socrata
import requests
import time
from io import StringIO

# ================================================================
# Database setup
# ================================================================
DB_PATH = "nyc_traffic_2016.duckdb"
conn = duckdb.connect(DB_PATH)
print(f"Connected to {DB_PATH}")

client = Socrata("data.cityofnewyork.us",
                 app_token="FvrjNyrm0p2bfhsooy9kNZ7ib",
                 timeout=60)

# Local CSVs
traffic_csv = "data/Automated_Traffic_Volume_Counts_20251102.csv"
collisions_csv = "data/Motor_Vehicle_Collisions_-_Crashes_20251102.csv"

# ================================================================
# NYC Open Data stream (Yellow Taxi Trip Data — 2016)
# ================================================================
API_URL = "https://data.cityofnewyork.us/api/views/uacg-pexx/rows.csv?accessType=DOWNLOAD"
APP_TOKEN = "FvrjNyrm0p2bfhsooy9kNZ7ib"

CHUNK_SIZE = 50_000
MAX_RETRIES = 5
headers = {"X-App-Token": APP_TOKEN}

print("\nPreparing to load taxi data (2016)…")

# ------------------------------------------------
# STEP 1 — Determine if taxi_data already exists
# ------------------------------------------------
try:
    row_count = conn.execute("SELECT COUNT(*) FROM taxi_data;").fetchone()[0]
    print(f"Resuming load — detected {row_count:,} existing rows in taxi_data.")
    table_exists = True
except duckdb.CatalogException:
    print("No existing taxi_data table found — starting fresh.")
    row_count = 0
    table_exists = False


# ================================================================
# Helper: Insert buffered lines into DuckDB WITH notifications
# ================================================================
def flush_buffer(buf, header_line, first_chunk):
    """Append buffered CSV rows to DuckDB, with progress notifications."""
    global row_count

    if not buf:
        print("Buffer empty; nothing to flush.")
        return first_chunk

    print(f"\nFlushing buffer with {len(buf):,} rows…")

    # Build CSV chunk
    csv_data = header_line + "\n" + "\n".join(buf)
    df_chunk = pd.read_csv(StringIO(csv_data))

    if df_chunk.empty:
        print("Chunk empty — skipping.")
        buf.clear()
        print("Buffer cleared.")
        return first_chunk

    # If first chunk:
    if first_chunk:
        if not table_exists:
            print("Creating taxi_data table from first chunk…")
            conn.execute("DROP TABLE IF EXISTS taxi_data;")
            conn.register("df_chunk", df_chunk)
            conn.execute("CREATE TABLE taxi_data AS SELECT * FROM df_chunk;")
            conn.unregister("df_chunk")
        else:
            print("Continuing inserts into existing taxi_data table…")
            conn.register("df_chunk", df_chunk)
            conn.execute("INSERT INTO taxi_data SELECT * FROM df_chunk;")
            conn.unregister("df_chunk")

        first_chunk = False

    else:
        conn.register("df_chunk", df_chunk)
        conn.execute("INSERT INTO taxi_data SELECT * FROM df_chunk;")
        conn.unregister("df_chunk")

    row_count += len(df_chunk)

    print(f"Inserted {len(df_chunk):,} rows this flush.")
    print(f"Total rows inserted so far: {row_count:,}.")
    buf.clear()
    print("Buffer cleared.")

    return first_chunk


# ================================================================
# STEP 2 — Stream CSV + resume logic
# ================================================================
print("\nStarting streamed download from NYC Open Data…")

retries = 0
row_buffer = []
first_chunk = not table_exists  # only create table if new

while True:
    try:
        with requests.get(API_URL, headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            lines = r.iter_lines(decode_unicode=True)

            header_line = next(lines)
            print("Header loaded.")
            print(f"Skipping first {row_count:,} rows already stored…")

            skip_counter = 0

            for line in lines:
                if not line:
                    continue

                # Skip rows already inserted
                if skip_counter < row_count:
                    skip_counter += 1
                    continue

                row_buffer.append(line)

                # Flush every CHUNK_SIZE rows
                if len(row_buffer) >= CHUNK_SIZE:
                    print(f"\n--- Reached {CHUNK_SIZE:,} buffered rows ---")
                    first_chunk = flush_buffer(row_buffer, header_line, first_chunk)
                    row_buffer = []

            # Final flush
            if row_buffer:
                first_chunk = flush_buffer(row_buffer, header_line, first_chunk)

        print("\nCompleted full taxi_data stream.")
        break

    except requests.exceptions.ChunkedEncodingError:
        retries += 1
        print(f"Connection dropped — retry {retries}/{MAX_RETRIES}")
        if retries >= MAX_RETRIES:
            print("Too many failures — aborting.")
            break
        print("Sleeping 10 seconds before retrying…")
        time.sleep(10)
        continue

print(f"\nFinal taxi_data row count: {row_count:,}\n")


# ================================================================
# Load Traffic Volume Counts (2016)
# ================================================================
print("Loading 2016 Traffic Volume Counts…")

df_sample = pd.read_csv(traffic_csv, nrows=5)
print("Traffic columns:", list(df_sample.columns))

conn.execute(f"""
CREATE TABLE IF NOT EXISTS traffic_2016 AS
SELECT *
FROM read_csv_auto('{traffic_csv}')
WHERE "Yr" = 2016;
""")

print(conn.execute("SELECT COUNT(*) FROM traffic_2016;").fetchone(),
      "rows loaded.\n")


# ================================================================
# Load Motor Vehicle Collisions (2016)
# ================================================================
print("Loading 2016 Motor Vehicle Collisions…")

df_sample = pd.read_csv(collisions_csv, nrows=5)
print("Collision columns:", list(df_sample.columns))

conn.execute(f"""
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
""")

print(conn.execute("SELECT COUNT(*) FROM collisions_2016;").fetchone(),
      "rows loaded.\n")


# ================================================================
# Summary + sample
# ================================================================
print("Tables in database:")
print(conn.execute("SHOW TABLES;").fetchdf())

print("\nSample rows from taxi_data:")
try:
    print(conn.execute("SELECT * FROM taxi_data LIMIT 5;").fetchdf())
except Exception as e:
    print("Cannot sample taxi_data yet:", e)

conn.close()
print(f"\nDatabase build complete — saved as {DB_PATH}")
