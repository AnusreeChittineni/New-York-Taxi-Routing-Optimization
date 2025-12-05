import duckdb
import time

con = duckdb.connect("nyc_routing.duckdb")

# ------------------------------------------------------------
# CLEAN TRIPS TABLE
# ------------------------------------------------------------

DISTANCE_TOLERANCE = 0.05 
FARE_TOLERANCE = 0.01 
# ---

def create_clean_enriched_taxi_table(con, distance_tolerance=0.05, fare_tolerance=0.01):
    """
    Replaces the standard cleaning query with an enriched version 
    that INNER JOINs taxi_raw (coordinates) and taxi_raw_parquet (Location IDs)
    to create the final, clean 'taxi_clean' table.
    """
    
    print("\n--- STAGE: CREATING CLEAN AND ENRICHED TABLE ---")
    print("Performing INNER JOIN and cleaning based on 5-part composite key.")
    start_time = time.time()
    
    # --- Execute the CREATE TABLE AS SELECT query ---
    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE taxi_clean AS
        SELECT
            t.tpep_pickup_datetime,
            t.tpep_dropoff_datetime,
            t.passenger_count,
            t.trip_distance,
            t.pickup_longitude AS pickup_lon,
            t.pickup_latitude AS pickup_lat,
            t.dropoff_longitude AS dropoff_lon,
            t.dropoff_latitude AS dropoff_lat,
            t.VendorID,
            t.RatecodeID,
            t.store_and_fwd_flag,
            t.payment_type,
            t.fare_amount,
            t.extra,
            t.mta_tax,
            t.tip_amount,
            t.tolls_amount,
            t.improvement_surcharge,
            t.total_amount,
            
            -- 2. ENRICHMENT: Select the accurate Location IDs from taxi_raw_parquet (p).
            -- We CAST these to VARCHAR/TEXT to match the type derived from your original CSV loader.
            CAST(p.PULocationID AS VARCHAR) AS PULocationID,
            CAST(p.DOLocationID AS VARCHAR) AS DOLocationID

        -- Use INNER JOIN to automatically filter out rows that don't match the clean, reliable Parquet data
        FROM (SELECT * FROM taxi_raw ORDER BY tpep_pickup_datetime) AS t
        INNER JOIN (SELECT * FROM taxi_raw_parquet ORDER BY tpep_pickup_datetime) AS p
            -- The join condition is the 5-part composite key: Vendor, 2xTime, 2xFuzzy-Float
            ON t.VendorID = p.VendorID 
            AND t.tpep_pickup_datetime = p.tpep_pickup_datetime 
            AND t.tpep_dropoff_datetime = p.tpep_dropoff_datetime
            -- Fuzzy match on trip distance
            AND ABS(t.trip_distance - p.trip_distance) <= {distance_tolerance}
            -- Fuzzy match on fare amount
            AND ABS(t.fare_amount - p.fare_amount) <= {fare_tolerance}
        
        -- Additional Cleaning Filter (Ensures only valid, non-zero, non-null data is kept)
        WHERE
            -- Core Time & Passenger Cleansing
            t.tpep_pickup_datetime IS NOT NULL
            AND t.tpep_dropoff_datetime IS NOT NULL
            AND t.passenger_count IS NOT NULL AND t.passenger_count > 0
            
            -- Core Trip Cleansing
            AND t.trip_distance IS NOT NULL AND t.trip_distance >= 0.01 
            
            -- Coordinate Cleansing
            AND t.pickup_longitude IS NOT NULL
            AND t.pickup_latitude IS NOT NULL
            AND t.dropoff_longitude IS NOT NULL
            AND t.dropoff_latitude IS NOT NULL
            
            -- Fare Cleansing
            AND t.fare_amount IS NOT NULL AND t.fare_amount > 0
        ;
        """)

        elapsed_time = time.time() - start_time
        clean_row_count = con.execute("SELECT COUNT(*) FROM taxi_clean").fetchone()[0]
        
        print(f"\nFinal table 'taxi_clean' created successfully.")
        print(f"Total rows in taxi_clean: {clean_row_count:,}")
        print(f"Time taken for cleaning and joining: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred during table creation: {e}")
        print(f"Ensure the global variables DISTANCE_TOLERANCE and FARE_TOLERANCE are defined before calling.")

# --- Execution Example (Assuming you have con connected) ---
#create_clean_enriched_taxi_table(con)
#con.close()

def drop_raw_tables(con):
    """Drops the raw tables to save space after cleaning."""
    print("\nDropping raw tables to save space...")
    con.execute("DROP TABLE IF EXISTS taxi_raw;")
    con.execute("DROP TABLE IF EXISTS taxi_raw_parquet;")
    print("Raw tables dropped.")

#drop_raw_tables(con)
#con.close()

# ------------------------------------------------------------
# CREATE TEMPORAL STREET STATUS FROM COLLISIONS
# ------------------------------------------------------------

print("Building closed street table...")

con.execute("""
CREATE OR REPLACE TABLE closed_streets AS
SELECT
    "ON STREET NAME" AS street_name,
    crash_datetime,
    'closed' AS status,
    LATITUDE,
    LONGITUDE
FROM collisions_2016
WHERE "ON STREET NAME" IS NOT NULL
""")


print("Closed street samples:")
print(con.execute("SELECT * FROM closed_streets LIMIT 10").fetchdf())

# ------------------------------------------------------------
# JOIN 1 — TAXI TRIPS × ROAD CLOSURES
# ------------------------------------------------------------

# Tolerance for spatial proximity in decimal degrees )
COORD_TOLERANCE = 0.01 # (~1 km)
SQUARED_TOLERANCE = COORD_TOLERANCE * COORD_TOLERANCE

def create_trips_with_closures_table(con, tolerance_sq=SQUARED_TOLERANCE):
    """
    Creates the 'trips_with_closures' table by joining taxi trips 
    with relevant road closures, requiring both a temporal (daily) 
    and a spatial (within tolerance) match.
    """
    print("\nJoining taxi trips with relevant road closures (Spatial + Temporal)...")
    start_time = time.time()
    
    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE trips_with_closures AS
        SELECT 
            t.*, 
            c.street_name, 
            c.status,
            -- Calculate and store the squared distance for analysis
            (
                (t.pickup_lat - c.LATITUDE)^2 +
                (t.pickup_lon - c.LONGITUDE)^2
            ) AS pickup_crash_dist_sq
        FROM taxi_clean t
        -- Use INNER JOIN because we only want trips where a relevant crash occurred
        INNER JOIN closed_streets c 
            -- TEMPORAL CRITERIA: Crash occurred on the same day as pickup
            ON t.tpep_pickup_datetime = c.crash_datetime
            
            -- SPATIAL CRITERIA: Pickup location is within the defined coordinate tolerance
            AND (
                (t.pickup_lat - c.LATITUDE)^2 +  -- Squared difference in latitude
                (t.pickup_lon - c.LONGITUDE)^2   -- Squared difference in longitude
            ) <= {tolerance_sq}
        ;
        """)

        elapsed_time = time.time() - start_time
        row_count = con.execute("SELECT COUNT(*) FROM trips_with_closures").fetchone()[0]
        
        print("Trip–closure join complete.")
        print(f"Total relevant rows created (Inner Join): {row_count:,}")
        print(f"Time taken for join: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"An error occurred during the join: {e}")
        print("Ensure 'closed_streets' has LATITUDE and LONGITUDE columns.")


# create_trips_with_closures_table(con)
# con.close()

# ------------------------------------------------------------
# JOIN 2 — TRAFFIC SPEED / TIME (SPATIO-TEMPORAL)
# ------------------------------------------------------------

print("\nJoining taxi trips with traffic speed/volume...")

con.execute("""
CREATE OR REPLACE TABLE trips_with_traffic AS
SELECT
    t.*,
    tvc.volume_count
FROM taxi_clean t
INNER JOIN (
    SELECT
        -- Combine date/time into timestamp
        STRPTIME(
            CAST(Yr AS VARCHAR) || '-' || 
            LPAD(CAST(M AS VARCHAR), 2, '0') || '-' || 
            LPAD(CAST(D AS VARCHAR), 2, '0') || ' ' ||
            LPAD(CAST(HH AS VARCHAR), 2, '0') || ':' ||
            LPAD(CAST(MM AS VARCHAR), 2, '0'),
            '%Y-%m-%d %H:%M'
        ) AS traffic_datetime,
        Vol AS volume_count,
        stz.PULocationID
    FROM traffic_2016 t
    INNER JOIN segment_to_zone_lookup stz
        ON t.SegmentID = stz.SegmentID
) tvc
-- Match taxi trip pickup time (rounded to minute) to traffic datetime
ON STRFTIME(t.tpep_pickup_datetime, '%Y-%m-%d %H:%M') = STRFTIME(tvc.traffic_datetime, '%Y-%m-%d %H:%M')
AND t.PULocationID = tvc.PULocationID
;
""")

print("Trip–traffic join complete.")
print(con.execute("SELECT COUNT(*) FROM trips_with_traffic").fetchone()[0])


con.close()
print("\nMerging and cleaning pipeline complete.")
