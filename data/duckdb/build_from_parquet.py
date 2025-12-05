import duckdb
import pandas as pd
from sodapy import Socrata


# Connect to DuckDB (persistent database file)
DB_PATH = "nyc_routing.duckdb"
conn = duckdb.connect(DB_PATH)
print(f"Connected to {DB_PATH}")


client = Socrata("data.cityofnewyork.us", app_token="FvrjNyrm0p2bfhsooy9kNZ7ib", timeout=60)

# --------------------------
# Load taxi data from parquet (already 2016)
# --------------------------

print("\n Loading 2016 Yellow Taxi data from TLC Parquet files...")

# Drop existing table if re-running
conn.execute("DROP TABLE IF EXISTS taxi_raw_parquet;")

conn.execute("""
CREATE TABLE IF NOT EXISTS taxi_raw_parquet AS 
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
        INSERT INTO taxi_raw_parquet
        SELECT * FROM read_parquet('{url}');
    """)

print("taxi_raw_parquet table created successfully.")
print(conn.execute("SELECT COUNT(*) FROM taxi_raw_parquet;").fetchone(), "rows loaded.\n")


print("\n Tables in database:")
print(conn.execute("SHOW TABLES;").fetchdf())

print("\n Sample from taxi_raw_parquet:")
print(conn.execute("SELECT * FROM taxi_raw_parquet LIMIT 5;").fetchdf())

conn.close()
print("\n Database build complete — saved as", DB_PATH)
