import duckdb
import pandas as pd
import time
import requests
from io import StringIO

DB_PATH = "nyc_routing.duckdb"
con = duckdb.connect(DB_PATH)
print(f"Connected to {DB_PATH}")

# ------------------------------------------------------------
# 1. LOAD 2016 TAXI TRIPS (streaming from NYC Open Data)
# ------------------------------------------------------------

API_URL = "https://data.cityofnewyork.us/api/views/uacg-pexx/rows.csv?accessType=DOWNLOAD"
APP_TOKEN = "FvrjNyrm0p2bfhsooy9kNZ7ib"
CHUNK_SIZE = 50_000

def flush(buf, header, first, table_cols=None):
    csv_data = header + "\n" + "\n".join(buf)
    df = pd.read_csv(StringIO(csv_data))

    # Align columns to existing table if table exists
    if table_cols is not None:
        for col in table_cols:
            if col not in df.columns:
                df[col] = None
        df = df[table_cols]  # ensure correct order

    con.register("chunk", df)

    if first:
        # First insert: create table
        con.execute("DROP TABLE IF EXISTS taxi_raw;")
        con.execute("CREATE TABLE taxi_raw AS SELECT * FROM chunk;")
        con.unregister("chunk")
        return False, len(df)
    else:
        # Subsequent insert
        inserted = con.execute("INSERT INTO taxi_raw SELECT * FROM chunk;").rowcount
        con.unregister("chunk")
        return False, inserted
    
def load_taxi_2016():
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    table_exists = "taxi_raw" in tables

    if table_exists:
        row_count = con.execute("SELECT COUNT(*) FROM taxi_raw").fetchone()[0]
        print(f"taxi_raw already exists with {row_count:,} rows — resuming download if needed.\n")

    first = not table_exists
    table_cols = None
    if table_exists:
        # Get existing table columns to align future inserts
        table_cols = [c[0] for c in con.execute("PRAGMA table_info('taxi_raw')").fetchall()]

    buffer = []
    total_rows_streamed = 0
    total_rows_inserted = 0

    headers = {"X-App-Token": APP_TOKEN}
    start_time = time.time()

    with requests.get(API_URL, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        lines = r.iter_lines(decode_unicode=True)
        header_line = next(lines)

        for line in lines:
            if not line.strip():
                continue
            buffer.append(line)
            if len(buffer) >= CHUNK_SIZE:
                first, inserted = flush(buffer, header_line, first, table_cols)
                total_rows_streamed += len(buffer)
                total_chunks_inserted += inserted
                buffer.clear()
                print(f"Streamed {total_rows_streamed:,} rows, inserted {total_rows_inserted:,} chunks…")

        # Flush remaining rows
        if buffer:
            first, inserted = flush(buffer, header_line, first, table_cols)
            total_rows_streamed += len(buffer)
            total_rows_inserted += inserted

    elapsed = time.time() - start_time
    print(f"\nTaxi ingestion completed: {total_rows_streamed:,} rows streamed")
    print(f"Total rows inserted into taxi_raw: {total_rows_inserted:,}")
    print(f"Total time: {elapsed:.2f} sec\n")

load_taxi_2016()

# ------------------------------------------------------------
# 2. LOAD TRAFFIC VOLUME COUNTS (2016)
# ------------------------------------------------------------

print("\n[2/3] Loading Traffic Volume Counts…")
traffic_file = "Automated_Traffic_Volume_Counts_20251102.csv"

t0 = time.time()
con.execute(f"""
CREATE TABLE IF NOT EXISTS traffic_2016 AS
SELECT *
FROM read_csv_auto('{traffic_file}')
WHERE "Yr" = 2016;
""")
t1 = time.time()

print(f"Traffic rows loaded: {con.execute('SELECT COUNT(*) FROM traffic_2016').fetchone()[0]}")
print(f"Ingestion time {t1-t0:.2f} sec")

# ------------------------------------------------------------
# 3. LOAD COLLISION DATA (2016)
# ------------------------------------------------------------

print("\n[3/3] Loading Collision Data…")
crash_file = "Motor_Vehicle_Collisions_-_Crashes_20251102.csv"

t0 = time.time()

con.execute(f"""
CREATE OR REPLACE TABLE collisions_2016 AS
SELECT
    *,
    -- Convert crash date
    STRPTIME(CAST("CRASH DATE" AS VARCHAR), '%m/%d/%Y') AS crash_date,
    
    -- Convert crash datetime
    STRPTIME(
        CAST("CRASH DATE" AS VARCHAR) || ' ' || CAST("CRASH TIME" AS VARCHAR),
        '%m/%d/%Y %H:%M'
    ) AS crash_datetime

FROM read_csv_auto(
    '{crash_file}',

    -- Type overrides to avoid auto-detection failures
    types={{
        'ZIP CODE': 'VARCHAR',
        'CRASH DATE': 'VARCHAR',
        'CRASH TIME': 'VARCHAR',
        'BOROUGH': 'VARCHAR',
        'LATITUDE': 'DOUBLE',
        'LONGITUDE': 'DOUBLE',
        'LOCATION': 'VARCHAR',
        'NUMBER OF PERSONS INJURED': 'INTEGER',
        'NUMBER OF PERSONS KILLED': 'INTEGER',
        'NUMBER OF PEDESTRIANS INJURED': 'INTEGER',
        'NUMBER OF PEDESTRIANS KILLED': 'INTEGER',
        'NUMBER OF CYCLIST INJURED': 'INTEGER',
        'NUMBER OF CYCLIST KILLED': 'INTEGER',
        'NUMBER OF MOTORIST INJURED': 'INTEGER',
        'NUMBER OF MOTORIST KILLED': 'INTEGER',
        'COLLISION_ID': 'BIGINT'
    }},

    nullstr=''   -- treat blank ZIP codes etc as NULL
)

-- Filter down to year 2016 after parsing date
WHERE EXTRACT(year FROM STRPTIME(CAST("CRASH DATE" AS VARCHAR), '%m/%d/%Y')) = 2016;
""")

t1 = time.time()

print(f"Collision rows loaded: {con.execute('SELECT COUNT(*) FROM collisions_2016').fetchone()[0]}")
print(f"Ingestion time {t1-t0:.2f} sec")

print("\nWarehouse build complete!")
con.close()
