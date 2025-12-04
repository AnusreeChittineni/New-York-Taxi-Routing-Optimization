import duckdb
import pandas as pd
import os
import time

# ---------------- CONFIG ----------------
DB_PATH = "nyc_routing.duckdb"
OUTPUT_DIR = "./model_csvs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

con = duckdb.connect(DB_PATH)
start_time = time.time()

print("\n--- EXPORTING 6 MODEL CSV FILES FROM DATA WAREHOUSE ---")

# ---------------- Traffic Train ----------------
traffic_train_query = """
SELECT
    t.id,
    AVG(t2.volume_count) AS last_hour_avg_traffic_vol
FROM trips_with_traffic t
INNER JOIN trips_with_traffic t2
    ON t.PULocationID = t2.PULocationID
    AND t2.tpep_pickup_datetime >= t.tpep_pickup_datetime - INTERVAL '1 hour'
    AND t2.tpep_pickup_datetime < t.tpep_pickup_datetime
GROUP BY t.id;
WHERE RANDOM() < 0.8;
"""
traffic_train = con.execute(traffic_train_query).fetchdf()
traffic_train.to_csv(os.path.join(OUTPUT_DIR, "traffic_train.csv"), index=False)
print(f"Traffic train CSV saved: {len(traffic_train):,} rows")

# ---------------- Traffic Test ----------------
traffic_test_query = """
SELECT
    t.id,
    AVG(t2.volume_count) AS last_hour_avg_traffic_vol
FROM trips_with_traffic t
INNER JOIN trips_with_traffic t2
    ON t.PULocationID = t2.PULocationID
    AND t2.tpep_pickup_datetime >= t.tpep_pickup_datetime - INTERVAL '1 hour'
    AND t2.tpep_pickup_datetime < t.tpep_pickup_datetime
GROUP BY t.id;
WHERE RANDOM() >= 0.8;
"""
traffic_test = con.execute(traffic_test_query).fetchdf()
traffic_test.to_csv(os.path.join(OUTPUT_DIR, "traffic_test.csv"), index=False)
print(f"Traffic test CSV saved: {len(traffic_test):,} rows")

# ---------------- Train with OSRM ----------------
train_osrm_query = """
SELECT
    t.id,
    t.VendorID AS vendor_id,
    t.tpep_pickup_datetime AS pickup_datetime,
    t.tpep_dropoff_datetime AS dropoff_datetime,
    t.passenger_count,
    t.pickup_lon AS pickup_longitude,
    t.pickup_lat AS pickup_latitude,
    t.dropoff_lon AS dropoff_longitude,
    t.dropoff_lat AS dropoff_latitude,
    t.store_and_fwd_flag,
    t.trip_distance,
    t.trip_duration AS trip_duration,
FROM taxi_clean t
WHERE RANDOM() < 0.8;
"""
train_osrm = con.execute(train_osrm_query).fetchdf()
train_osrm.to_csv(os.path.join(OUTPUT_DIR, "train_with_osrm.csv"), index=False)
print(f"Train with OSRM CSV saved: {len(train_osrm):,} rows")

# ---------------- Test with OSRM ----------------
test_osrm_query = """
SELECT
    t.id,
    t.VendorID AS vendor_id,
    t.tpep_pickup_datetime AS pickup_datetime,
    t.passenger_count,
    t.pickup_lon AS pickup_longitude,
    t.pickup_lat AS pickup_latitude,
    t.dropoff_lon AS dropoff_longitude,
    t.dropoff_lat AS dropoff_latitude,
    t.store_and_fwd_flag,
    t.trip_distance,
FROM taxi_clean t
WHERE RANDOM() >= 0.8;
"""
test_osrm = con.execute(test_osrm_query).fetchdf()
test_osrm.to_csv(os.path.join(OUTPUT_DIR, "test_with_osrm.csv"), index=False)
print(f"Test with OSRM CSV saved: {len(test_osrm):,} rows")

# ---------------- Collision Train ----------------
collision_train_query = """
SELECT
    t.id,
    COUNT(c.crash_datetime) AS last_hour_collisions
FROM trips_with_closures t
LEFT JOIN collisions_2016 c
    ON c.crash_datetime BETWEEN t.tpep_pickup_datetime - INTERVAL '1 hour'
                             AND t.tpep_pickup_datetime
GROUP BY t.id
HAVING RANDOM() < 0.8;
"""
collision_train = con.execute(collision_train_query).fetchdf()
collision_train.to_csv(os.path.join(OUTPUT_DIR, "collision_train.csv"), index=False)
print(f"Collision train CSV saved: {len(collision_train):,} rows")

# ---------------- Collision Test ----------------
collision_test_query = """
SELECT
    t.id,
    COUNT(c.crash_datetime) AS last_hour_collisions
FROM trips_with_closures t
LEFT JOIN collisions_2016 c
    ON c.crash_datetime BETWEEN t.tpep_pickup_datetime - INTERVAL '1 hour'
                             AND t.tpep_pickup_datetime
GROUP BY t.id
HAVING RANDOM() >= 0.8;
"""
collision_test = con.execute(collision_test_query).fetchdf()
collision_test.to_csv(os.path.join(OUTPUT_DIR, "collision_test.csv"), index=False)
print(f"Collision test CSV saved: {len(collision_test):,} rows")

elapsed = time.time() - start_time
print(f"\nAll CSV exports completed in {elapsed:.2f} seconds.")

con.close()

