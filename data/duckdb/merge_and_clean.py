import duckdb

con = duckdb.connect("nyc_routing.duckdb")

# ------------------------------------------------------------
# CLEAN TRIPS TABLE
# ------------------------------------------------------------

print("Cleaning taxi trip table...")

con.execute("""
CREATE OR REPLACE TABLE taxi_clean AS
SELECT
    tpep_pickup_datetime,
    tpep_dropoff_datetime,
    passenger_count,
    trip_distance,
    pickup_longitude AS pickup_lon,
    pickup_latitude  AS pickup_lat,
    dropoff_longitude AS dropoff_lon,
    dropoff_latitude  AS dropoff_lat,

FROM taxi_raw
WHERE
    tpep_pickup_datetime IS NOT NULL
    AND tpep_dropoff_datetime IS NOT NULL
    AND trip_distance > 0
    AND pickup_longitude IS NOT NULL
    AND pickup_latitude  IS NOT NULL
    AND dropoff_longitude IS NOT NULL
    AND dropoff_latitude  IS NOT NULL
""")

print("Taxi cleaned rows:", con.execute("SELECT COUNT(*) FROM taxi_clean").fetchone()[0])

# ------------------------------------------------------------
# CREATE TEMPORAL STREET STATUS FROM COLLISIONS
# ------------------------------------------------------------

print("Building closed street table...")

con.execute("""
CREATE OR REPLACE TABLE closed_streets AS
SELECT
    street_upper AS street_name,
    crash_datetime,
    'closed' AS status
FROM collisions_2016
WHERE street_upper IS NOT NULL
""")

print("Closed street samples:")
print(con.execute("SELECT * FROM closed_streets LIMIT 10").fetchdf())

# ------------------------------------------------------------
# JOIN 1 — TAXI TRIPS × ROAD CLOSURES
# ------------------------------------------------------------

print("\nJoining taxi trips with road closures...")

con.execute("""
CREATE OR REPLACE TABLE trips_with_closures AS
SELECT t.*, c.street_name, c.status
FROM taxi_clean t
LEFT JOIN closed_streets c
     ON DATE(t.tpep_pickup_datetime) = DATE(c.crash_datetime)
""")

print("Trip–closure join complete.")
print(con.execute("SELECT COUNT(*) FROM trips_with_closures").fetchone()[0])

# ------------------------------------------------------------
# JOIN 2 — TRAFFIC SPEED / TIME (SPATIO-TEMPORAL)
# ------------------------------------------------------------

print("\nJoining taxi trips with traffic speed/time...")

con.execute("""
CREATE OR REPLACE TABLE trips_with_traffic AS
SELECT t.*, tvc.volume_count, tvc.hour
FROM taxi_clean t
LEFT JOIN (
    SELECT
        Date,
        Hour AS hour,
        Volume AS volume_count
    FROM traffic_2016
) tvc
ON t.pickup_date = tvc.Date
AND t.pickup_hour = tvc.hour
""")

print("Trip–traffic join complete.")
print(con.execute("SELECT COUNT(*) FROM trips_with_traffic").fetchone()[0])

con.close()
print("\nMerging and cleaning pipeline complete.")
