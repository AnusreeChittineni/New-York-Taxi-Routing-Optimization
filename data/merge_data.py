import duckdb

# Connect to the same DuckDB database
conn = duckdb.connect("nyc_traffic_2016.duckdb")

# --------------------------
# Merge the tables into a single persistent table
# --------------------------
# Adjust column names to match your CSV headers
conn.execute("""
CREATE TABLE IF NOT EXISTS merged_2016 AS
SELECT 
    t.*,
    tr.*,
    c.*
FROM taxi_2016 t
LEFT JOIN traffic_2016 tr
    ON t.pickup_longitude = tr.longitude
    AND t.pickup_latitude = tr.latitude
    AND DATE(t.pickup_datetime) = DATE(tr.date)
LEFT JOIN collisions_2016 c
    ON tr.longitude = c.longitude
    AND tr.latitude = c.latitude
    AND DATE(tr.date) = DATE(c.crash_date)
""")

# --------------------------
# Test query
# --------------------------
df = conn.execute("SELECT * FROM merged_2016 LIMIT 10").fetchdf()
print("Merged table created successfully. Sample rows:")
print(df)
