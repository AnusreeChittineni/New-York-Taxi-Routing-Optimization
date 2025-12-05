import duckdb
import pandas as pd
import pyproj
from scipy.spatial import cKDTree
import numpy as np
import os

# --- 0. Setup and Connection ---
DB_PATH = "nyc_traffic_2016.duckdb"

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database file not found at {DB_PATH}. Please run the initial data loading script first.")

conn = duckdb.connect(DB_PATH)
print(f"Connected to {DB_PATH}")
print("\n--- Step 0: Rebuilding Traffic Table with Date Columns ---")
# We rebuild this table to ensure it has 'Yr', 'M', 'D' required for the GNN join.
conn.execute("""
CREATE OR REPLACE TABLE segment_congestion_data AS
WITH SegmentAttributes AS (
    SELECT
        SegmentID,
        FIRST(CAST(NULLIF(REGEXP_EXTRACT(WktGeom, 'POINT \\(([0-9]+\\.[0-9]+)', 1), '') AS DOUBLE)) AS X_coord,
        FIRST(CAST(NULLIF(REGEXP_EXTRACT(WktGeom, 'POINT \\([0-9]+\\.[0-9]+ ([0-9]+\\.[0-9]+)\\)', 1), '') AS DOUBLE)) AS Y_coord
    FROM traffic_2016
    GROUP BY 1
),
SegmentVolume AS (
    SELECT 
        SegmentID, 
        -- CRITICAL: Keep Date columns for GNN time-series
        Yr, M, D, 
        CAST(HH AS INTEGER) * 60 + CAST(MM AS INTEGER) - (CAST(MM AS INTEGER) % 15) AS TimeBin_Min,
        AVG(CAST(REPLACE(Vol, ',', '') AS DOUBLE)) AS Avg_Volume
    FROM traffic_2016
    WHERE Yr = 2016
    GROUP BY 1, 2, 3, 4, 5
)
SELECT 
    T1.SegmentID,
    T1.Yr, T1.M, T1.D, 
    T1.TimeBin_Min,
    T1.Avg_Volume,
    T2.X_coord, 
    T2.Y_coord
FROM SegmentVolume T1
JOIN SegmentAttributes T2 ON T1.SegmentID = T2.SegmentID
WHERE T2.X_coord IS NOT NULL AND T2.Y_coord IS NOT NULL;
""")
print("Traffic table ready.")

# Step 1: collision data
print("\n--- Step 1: Aggregating Collision Data by Time Bin ---")

collision_sql_temp = """
CREATE OR REPLACE TABLE collision_counts_temp AS
SELECT
    STRPTIME(
        REGEXP_REPLACE(
             CAST("CRASH DATE" AS VARCHAR) || ' ' || CAST("CRASH TIME" AS VARCHAR), 
             '/', '-'
        ),
        '%Y-%m-%d %H:%M:%S'
    ) AS crash_datetime,
    
    (EXTRACT(HOUR FROM crash_datetime) * 60 + EXTRACT(MINUTE FROM crash_datetime)) - 
    ((EXTRACT(MINUTE FROM crash_datetime)) % 15) AS TimeBin_Min,
    
    EXTRACT(YEAR FROM crash_datetime) AS Yr,
    EXTRACT(MONTH FROM crash_datetime) AS M,
    EXTRACT(DAY FROM crash_datetime) AS D,
    
    LATITUDE,
    LONGITUDE,
    
    CASE WHEN "NUMBER OF PERSONS INJURED" > 0 OR "NUMBER OF PERSONS KILLED" > 0 THEN 1 ELSE 0 END AS Is_Severe
    
FROM collisions_2016
WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL;
"""
conn.execute(collision_sql_temp)
print("collision_counts_temp created.")

# STEP 2: SPATIAL MATCHING: Collisions -> Segments
print("\n--- Step 2: Spatial Matching of Collisions to Segments ---")

# A. Load Data
df_collisions = conn.execute("SELECT * FROM collision_counts_temp").fetchdf()
df_segments = conn.execute("SELECT DISTINCT SegmentID, X_coord, Y_coord FROM segment_congestion_data").fetchdf()

if df_segments.empty:
    raise ValueError("ERROR: Segment data is empty. Check Step 0.")

# B. Transform Coordinates (GPS -> State Plane)
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
lon, lat = df_collisions['LONGITUDE'].values, df_collisions['LATITUDE'].values
x_coll, y_coll = transformer.transform(lon, lat)

df_collisions['X_coll'] = x_coll
df_collisions['Y_coll'] = y_coll

# C. Nearest Neighbor Matching
segment_coords = df_segments[['X_coord', 'Y_coord']].values
tree = cKDTree(segment_coords)

collision_coords = df_collisions[['X_coll', 'Y_coll']].values
distance, index = tree.query(collision_coords)

# Map back to IDs
df_collisions['SegmentID'] = df_segments.iloc[index]['SegmentID'].values

# D. Save Matched Data
conn.register('df_collisions_matched', df_collisions)

final_collision_sql = """
CREATE OR REPLACE TABLE collision_target AS
SELECT
    SegmentID,
    Yr, M, D,
    TimeBin_Min,
    COUNT(*) AS Collision_Count,
    SUM(Is_Severe) AS Severe_Collision_Count
FROM df_collisions_matched
GROUP BY 1, 2, 3, 4, 5;
"""
conn.execute(final_collision_sql)
print("Collision target table `collision_target` created.")

# STEP 3: MASTER JOIN (The GNN Dataset)
print("\n--- Step 3: Creating Final GNN Master Table ---")

master_join_sql = """
CREATE OR REPLACE TABLE gnn_master_features AS
SELECT
    T1.SegmentID,
    T1.Yr, T1.M, T1.D,
    T1.TimeBin_Min,
    T1.Avg_Volume,
    T1.X_coord, T1.Y_coord,
    COALESCE(T2.Collision_Count, 0) AS Collision_Count,
    COALESCE(T2.Severe_Collision_Count, 0) AS Severe_Collision_Count
FROM segment_congestion_data AS T1
LEFT JOIN collision_target AS T2
    ON T1.SegmentID = T2.SegmentID
    AND T1.Yr = T2.Yr
    AND T1.M = T2.M
    AND T1.D = T2.D
    AND T1.TimeBin_Min = T2.TimeBin_Min;
"""

conn.execute(master_join_sql)
print("\n GNN Master Feature table `gnn_master_features` successfully created.")
print(conn.execute("SELECT * FROM gnn_master_features WHERE Collision_Count > 0 LIMIT 5").fetchdf())

conn.close()