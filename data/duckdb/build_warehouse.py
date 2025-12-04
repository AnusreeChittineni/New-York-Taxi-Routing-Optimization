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
        con.execute("""
        CREATE TABLE taxi_raw AS
        SELECT
            STRPTIME(tpep_pickup_datetime, '%m/%d/%Y %I:%M:%S %p') AS tpep_pickup_datetime,
            STRPTIME(tpep_dropoff_datetime, '%m/%d/%Y %I:%M:%S %p') AS tpep_dropoff_datetime,
            passenger_count,
            trip_distance,
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            VendorID,
            RatecodeID,
            store_and_fwd_flag,
            payment_type,
            fare_amount,
            extra,
            mta_tax,
            tip_amount,
            tolls_amount,
            improvement_surcharge,
            total_amount,
            PULocationID,
            DOLocationID
        FROM chunk;
        """)

        con.unregister("chunk")
        return False, len(df)
    else:
        # Subsequent insert
        inserted = con.execute("""
        INSERT INTO taxi_raw
        SELECT
            STRPTIME(tpep_pickup_datetime, '%m/%d/%Y %I:%M:%S %p') AS tpep_pickup_datetime,
            STRPTIME(tpep_dropoff_datetime, '%m/%d/%Y %I:%M:%S %p') AS tpep_dropoff_datetime,
            passenger_count,
            trip_distance,
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            VendorID,
            RatecodeID,
            store_and_fwd_flag,
            payment_type,
            fare_amount,
            extra,
            mta_tax,
            tip_amount,
            tolls_amount,
            improvement_surcharge,
            total_amount,
            PULocationID,
            DOLocationID
        FROM chunk;
    """).rowcount

        con.unregister("chunk")
        return False, inserted
    
def load_taxi_2016():
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    table_exists = False

    """
    if table_exists:
        row_count = con.execute("SELECT COUNT(*) FROM taxi_raw").fetchone()[0]
        print(f"taxi_raw already exists with {row_count:,} rows — resuming download if needed.\n")
    """

    first = True
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
                total_rows_inserted += inserted
                buffer.clear()
                print(f"Streamed {total_rows_streamed:,} rows, inserted {total_rows_inserted:,} rows...")

        # Flush remaining rows
        if buffer:
            first, inserted = flush(buffer, header_line, first, table_cols)
            total_rows_streamed += len(buffer)
            total_rows_inserted += inserted

    elapsed = time.time() - start_time
    print(f"\nTaxi ingestion completed: {total_rows_streamed:,} rows streamed")
    print(f"Total rows inserted into taxi_raw: {total_rows_inserted:,}")
    print(f"Total time: {elapsed:.2f} sec\n")

# load_taxi_2016()


PARQUET_URLS = [
    f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2016-{m:02}.parquet"
    for m in range(1, 13)
]

def update_location_ids_from_parquet(con):
    """
    Connects to the DuckDB connection (con) and updates the PULocationID and 
    DOLocationID columns in the 'taxi_raw' table using data from the 2016 
    TLC Yellow Taxi Parquet files.
    """
    
    # 1. Prepare the Parquet URL string for DuckDB
    parquet_list_str = "','".join(PARQUET_URLS)
    
    print("\n--- STAGE: ENRICHING LOCATION IDs ---")
    print("Using 2016 TLC Parquet files to update Location IDs in taxi_raw.")
    start_time = time.time()

    # 2. Execute the Massive Update
    try:
        # Check current status before update (Optional but helpful)
        current_null_count = con.execute("""
            SELECT COUNT(*) FROM taxi_raw 
            WHERE PULocationID IS NULL OR DOLocationID IS NULL
        """).fetchone()[0]
        print(f"Rows missing Location IDs before update: {current_null_count:,}")
        
        # The key UPDATE FROM query: joins taxi_raw (t) with the Parquet data (p)
        update_count = con.execute(f"""
        UPDATE taxi_raw AS t
        SET 
            PULocationID = p.PULocationID,
            DOLocationID = p.DOLocationID
        FROM read_parquet(['{parquet_list_str}']) AS p
        WHERE 
            t.tpep_pickup_datetime = p.tpep_pickup_datetime 
            AND t.tpep_dropoff_datetime = p.tpep_dropoff_datetime;
        """).rowcount

        elapsed_time = time.time() - start_time
        print(f"\nUpdate Complete. Total rows enriched: {update_count:,}")
        print(f"Time taken for update: {elapsed_time:.2f} seconds")

        # 3. Final Verification
        final_null_count = con.execute("""
            SELECT COUNT(*) FROM taxi_raw 
            WHERE PULocationID IS NULL OR DOLocationID IS NULL
        """).fetchone()[0]
        print(f"Rows still missing Location IDs after update: {final_null_count:,}")

    except Exception as e:
        print(f"An error occurred during the Location ID update: {e}")


update_location_ids_from_parquet(con)

# ------------------------------------------------------------
# 2. LOAD TRAFFIC VOLUME COUNTS (2016)
# ------------------------------------------------------------

print("\n[2/3] Loading Traffic Volume Counts…")
traffic_file = "../Automated_Traffic_Volume_Counts_20251203.csv"

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
crash_file = "../Motor_Vehicle_Collisions_-_Crashes_20251203.csv"

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
