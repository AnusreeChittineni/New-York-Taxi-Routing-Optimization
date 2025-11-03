import duckdb
import pandas as pd
from sodapy import Socrata


# Connect to DuckDB (persistent database file)
DB_PATH = "nyc_traffic_2016.duckdb"
conn = duckdb.connect(DB_PATH)
print(f"Connected to {DB_PATH}")


client = Socrata("data.cityofnewyork.us", app_token="FvrjNyrm0p2bfhsooy9kNZ7ib", timeout=60)


# file paths
# taxi_excel = "data\Yellow_Taxi_Trip_Data_Data_Dictionary.xlsx"          # already 2016 only
traffic_csv = "data\Automated_Traffic_Volume_Counts_20251102.csv"   # contains multiple years
collisions_csv = "data\Motor_Vehicle_Collisions_-_Crashes_20251102.csv"    # contains multiple years

# --------------------------
# Load taxi data (already 2016)
# --------------------------

print("\n Loading 2016 Yellow Taxi data from TLC Parquet files...")

# Drop existing table if re-running
conn.execute("DROP TABLE IF EXISTS taxi_data;")

conn.execute("""
CREATE TABLE IF NOT EXISTS taxi_data AS 
SELECT * FROM read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2016-01.parquet')
LIMIT 0;
""")

# 2. Append each month
parquet_urls = [
    f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2016-{m:02}.parquet"
    for m in range(1, 13)
]

for url in parquet_urls:
    print(f"Inserting {url}...", flush=True)
    conn.execute(f"""
        INSERT INTO taxi_data
        SELECT * FROM read_parquet('{url}');
    """)

print("taxi_data table created successfully.")
print(conn.execute("SELECT COUNT(*) FROM taxi_data;").fetchone(), "rows loaded.\n")

# --------------------------
# Load traffic CSV (filter only 2016)
# --------------------------

df_sample = pd.read_csv(traffic_csv, nrows=5)
print(df_sample.columns)

conn.execute(f"""
CREATE TABLE IF NOT EXISTS traffic_2016 AS
SELECT *
FROM read_csv_auto('{traffic_csv}')
WHERE "Yr" = 2016;
""")

print("traffic_2016 table created successfully.")
print(conn.execute("SELECT COUNT(*) FROM traffic_2016;").fetchone(), "rows loaded.\n")

# --------------------------
# Load collisions CSV (filter only 2016)
# --------------------------

df_sample = pd.read_csv(collisions_csv, nrows=5)
print(df_sample)

conn.execute(f"""
CREATE TABLE IF NOT EXISTS collisions_2016 AS
SELECT *,
       STRPTIME(CAST("CRASH DATE" AS VARCHAR) || ' ' || CAST("CRASH TIME" AS VARCHAR), '%Y-%m-%d %H:%M:%S') AS crash_datetime
FROM read_csv_auto(
    '{collisions_csv}',
    types={{'ZIP CODE': 'VARCHAR'}},   -- force ZIP CODE to string
    nullstr=''                         -- treat empty strings as NULL
)
WHERE EXTRACT(year FROM STRPTIME(CAST("CRASH DATE" AS VARCHAR), '%Y-%m-%d')) = 2016;
""")

print("collisions_2016 table created successfully.")
print(conn.execute("SELECT COUNT(*) FROM collisions_2016;").fetchone(), "rows loaded.\n")


print("\n Tables in database:")
print(conn.execute("SHOW TABLES;").fetchdf())

print("\n Sample from taxi_data:")
print(conn.execute("SELECT * FROM taxi_data LIMIT 5;").fetchdf())

conn.close()
print("\n Database build complete — saved as", DB_PATH)

