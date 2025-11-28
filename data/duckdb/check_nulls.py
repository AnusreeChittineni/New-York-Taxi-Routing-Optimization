import duckdb

con = duckdb.connect("nyc_routing.duckdb")

# Print one row to inspect the data format
sample_df = con.execute("SELECT * FROM taxi_raw LIMIT 1").fetchdf()
print(sample_df)

print("\n=== Checking nulls and invalid values in taxi_raw ===\n")

checks = {
    "tpep_pickup_datetime IS NULL": """
        SELECT COUNT(*) FROM taxi_raw
        WHERE tpep_pickup_datetime IS NULL
    """,
    "tpep_dropoff_datetime IS NULL": """
        SELECT COUNT(*) FROM taxi_raw
        WHERE tpep_dropoff_datetime IS NULL
    """,
    "trip_distance <= 0 OR trip_distance IS NULL": """
        SELECT COUNT(*) FROM taxi_raw
        WHERE trip_distance IS NULL OR trip_distance <= 0
    """,
    "pickup_longitude IS NULL": """
        SELECT COUNT(*) FROM taxi_raw
        WHERE pickup_longitude IS NULL
    """,
    "pickup_latitude IS NULL": """
        SELECT COUNT(*) FROM taxi_raw
        WHERE pickup_latitude IS NULL
    """,
    "dropoff_longitude IS NULL": """
        SELECT COUNT(*) FROM taxi_raw
        WHERE dropoff_longitude IS NULL
    """,
    "dropoff_latitude IS NULL": """
        SELECT COUNT(*) FROM taxi_raw
        WHERE dropoff_latitude IS NULL
    """
}

# Print counts for each issue
for label, query in checks.items():
    count = con.execute(query).fetchone()[0]
    print(f"{label}: {count:,}")

# Compute rows that satisfy ALL constraints
valid_rows_query = """
    SELECT COUNT(*) FROM taxi_raw
    WHERE
        tpep_pickup_datetime IS NOT NULL
        AND tpep_dropoff_datetime IS NOT NULL
        AND trip_distance > 0
        AND pickup_longitude IS NOT NULL
        AND pickup_latitude  IS NOT NULL
        AND dropoff_longitude IS NOT NULL
        AND dropoff_latitude  IS NOT NULL
"""
valid_rows = con.execute(valid_rows_query).fetchone()[0]

total_rows = con.execute("SELECT COUNT(*) FROM taxi_raw").fetchone()[0]

print("\n=== Summary ===")
print(f"Total rows in taxi_raw: {total_rows:,}")
print(f"Rows that PASS all filters: {valid_rows:,}")
print(f"Rows that FAIL at least one filter: {total_rows - valid_rows:,}")

con.close()
