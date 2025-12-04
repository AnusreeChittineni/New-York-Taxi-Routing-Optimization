import duckdb
import pandas as pd

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
DB_PATH = "../nyc_routing.duckdb"
CSV_PATH = "segment_to_pulocationid_lookup.csv"
TABLE_NAME = "segment_to_zone_lookup"

# --------------------------------------------------------
# CONNECT TO DUCKDB
# --------------------------------------------------------
con = duckdb.connect(DB_PATH)

# --------------------------------------------------------
# LOAD CSV INTO PANDAS
# --------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded CSV: {len(df)} rows")

# --------------------------------------------------------
# WRITE INTO DUCKDB TABLE
# --------------------------------------------------------
# Replace table if it exists
con.execute(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM df")

print(f"Table '{TABLE_NAME}' created successfully in DuckDB.")

# --------------------------------------------------------
# OPTIONAL: Verify first 5 rows
# --------------------------------------------------------
result = con.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5").fetchdf()
print("Example rows:")
print(result)

# --------------------------------------------------------
# CLOSE CONNECTION
# --------------------------------------------------------
con.close()
