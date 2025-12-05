import duckdb

# Connect to DuckDB
con = duckdb.connect(database='nyc_traffic_2016.duckdb') 


# map collision locations (lat, long) to street status (open, closed)
# can check if a closed street is involved in a predcited taxi route

# Step 1: Create a temporal street status table based on collisions
con.execute("""
CREATE OR REPLACE TABLE closed_streets AS
SELECT 
    street_upper AS street_name,
    crash_datetime,
    'closed' AS status
FROM collisions_2016
WHERE street_upper IS NOT NULL
""")

# Verify results

result = con.execute("SELECT * FROM closed_streets where status = 'closed' LIMIT 20").fetchall()
print(result)
