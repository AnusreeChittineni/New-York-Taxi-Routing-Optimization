import duckdb

# Connect to your existing data warehouse
con = duckdb.connect("../nyc_routing.duckdb")

# Query taxi_clean for the needed columns
query = """
SELECT
    CAST(PULocationID AS INTEGER) AS PULocationID,
    CAST(DOLocationID AS INTEGER) AS DOLocationID,
    trip_distance,
    tpep_pickup_datetime AS pickup_ts,
    tpep_dropoff_datetime AS dropoff_ts
FROM taxi_clean
ORDER BY tpep_pickup_datetime
LIMIT 100  -- remove or change limit as needed
"""

# Execute query and fetch results
df = con.execute(query).fetchdf()

# Save to CSV
df.to_csv("taxi_clean_extract.csv", index=False)

print("CSV saved: taxi_clean_extract.csv")
print(df.head())

# Query to randomly select 2 rows
query_random_rows = """
SELECT
    PULocationID,
    DOLocationID,
    trip_distance,
    tpep_pickup_datetime AS pickup_ts,
    tpep_dropoff_datetime AS dropoff_ts
FROM taxi_clean
ORDER BY RANDOM()
LIMIT 2
"""

random_rows_df = con.execute(query_random_rows).fetchdf()

print("Random sample of 2 rows from taxi_clean:")
print(random_rows_df)

# Query: average traffic volume per SegmentID
query_avg_volume = """
SELECT
    t.SegmentID,
    ROUND(AVG(CAST(REPLACE(t.Vol, ',', '') AS DOUBLE)), 8) AS Avg_Volume
FROM trips_with_traffic t
GROUP BY t.SegmentID
ORDER BY t.SegmentID
"""

avg_volume_df = con.execute(query_avg_volume).fetchdf()

# Save to CSV
avg_volume_df.to_csv("segment_avg_volume.csv", index=False)

print("CSV saved: segment_avg_volume.csv")
print(avg_volume_df.head())

# Close the connection
con.close()
