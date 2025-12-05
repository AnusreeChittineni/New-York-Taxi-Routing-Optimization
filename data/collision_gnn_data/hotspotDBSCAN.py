import duckdb
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
from scipy.spatial import cKDTree

# --- 1. Setup ---
DB_PATH = "../duckdb/nyc_routing.duckdb"
SHAPEFILE_PATH = "taxi_zones/taxi_zones.shp"

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database not found at {DB_PATH}")

conn = duckdb.connect(DB_PATH)
print(f"Connected to {DB_PATH}")

# --- 2. Load Shapefile ---
print("\n--- Step 1: Processing Local Shapefile ---")
try:
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if gdf.crs.to_string() != "EPSG:2263":
        gdf = gdf.to_crs("EPSG:2263")
    gdf['centroid'] = gdf.geometry.centroid
    gdf['Zone_X'] = gdf.centroid.x
    gdf['Zone_Y'] = gdf.centroid.y
    gdf['LocationID'] = gdf['LocationID'].astype(int)
    
    df_zone_lookup = gdf[['LocationID', 'Zone_X', 'Zone_Y']].copy()
    conn.register('zone_lookup_df', df_zone_lookup)
except Exception as e:
    print(f"Error loading shapefile: {e}")
    exit()

# --- 3. Prepare Traffic Segments ---
print("\n--- Step 2: Preparing Traffic Segments ---")
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
        CAST(HH AS INTEGER) * 60 + CAST(MM AS INTEGER) - (CAST(MM AS INTEGER) % 15) AS TimeBin_Min,
        AVG(CAST(REPLACE(Vol, ',', '') AS DOUBLE)) AS Avg_Volume
    FROM traffic_2016
    WHERE Yr = 2016
    GROUP BY 1, 2
)
SELECT T1.*, T2.X_coord, T2.Y_coord
FROM SegmentVolume T1
JOIN SegmentAttributes T2 ON T1.SegmentID = T2.SegmentID
WHERE T2.X_coord IS NOT NULL AND T2.Y_coord IS NOT NULL;
""")

df_segments = conn.execute("SELECT DISTINCT SegmentID, X_coord, Y_coord FROM segment_congestion_data").fetchdf()
df_segments = df_segments.dropna(subset=['X_coord', 'Y_coord'])
df_segments = df_segments[np.isfinite(df_segments['X_coord']) & np.isfinite(df_segments['Y_coord'])]

# --- 4. Process Taxi Data ---
print("\n--- Step 3: Processing Taxi Data ---")
taxi_query = """
SELECT t.PULocationID, z.Zone_X, z.Zone_Y
FROM taxi_data t
JOIN zone_lookup_df z ON t.PULocationID = z.LocationID
WHERE EXTRACT(HOUR FROM tpep_pickup_datetime) = 8
AND t.PULocationID IS NOT NULL
"""
df_taxi = conn.execute(taxi_query).fetchdf()

# --- 5. Spatial Matching ---
print("\n--- Step 4: Spatial Matching ---")
segment_coords = df_segments[['X_coord', 'Y_coord']].values
tree = cKDTree(segment_coords)
taxi_coords = df_taxi[['Zone_X', 'Zone_Y']].values
dists, indices = tree.query(taxi_coords)
df_taxi['SegmentID'] = df_segments.iloc[indices]['SegmentID'].values

# --- 6. Fusion ---
print("\n--- Step 5: Creating Congestion Index ---")
taxi_counts = df_taxi.groupby('SegmentID').size().reset_index(name='Taxi_Count')
traffic_vol = conn.execute("""
    SELECT SegmentID, AVG(Avg_Volume) as Traffic_Volume
    FROM segment_congestion_data
    WHERE TimeBin_Min BETWEEN 480 AND 540
    GROUP BY SegmentID
""").fetchdf()

df_merged = pd.merge(traffic_vol, taxi_counts, on='SegmentID', how='left').fillna(0)
scaler = MinMaxScaler()
df_merged[['Vol_Norm', 'Taxi_Norm']] = scaler.fit_transform(df_merged[['Traffic_Volume', 'Taxi_Count']])
df_merged['Congestion_Index'] = (0.6 * df_merged['Vol_Norm']) + (0.4 * df_merged['Taxi_Norm'])

# --- 7. Tuning DBSCAN ---
print("\n--- Step 6: Tuning DBSCAN Parameters ---")
threshold = df_merged['Congestion_Index'].quantile(0.50) 
df_hot = df_merged[df_merged['Congestion_Index'] >= threshold].copy()
df_hot = pd.merge(df_hot, df_segments[['SegmentID', 'X_coord', 'Y_coord']], on='SegmentID', how='inner')

X_spatial = df_hot[['X_coord', 'Y_coord']].values
X_scaled = StandardScaler().fit_transform(X_spatial)

# Tuning Logic
best_score = -1
best_eps = 0
best_min = 0
best_labels = None

print("Grid Search for Optimal Parameters...")
for eps in np.arange(0.15, 0.5, 0.05):
    for min_samples in range(3, 7):
        db_test = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db_test.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Avoid scenarios where everything is noise or everything is 1 cluster
        if n_clusters > 2 and n_noise / len(labels) < 0.7:
            score = n_clusters * (1 - n_noise/len(labels))
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min = min_samples
                best_labels = labels

if best_labels is not None:
    print(f"\nOptimal Parameters: Eps={best_eps:.2f}, MinSamples={best_min}")
    df_hot['cluster_label'] = best_labels
else:
    print("Using defaults (Grid search failed to find optimal spread).")
    db = DBSCAN(eps=0.3, min_samples=3).fit(X_scaled)
    df_hot['cluster_label'] = db.labels_

# --- 8. Save DBSCAN Results to DuckDB to use for Collision Risk analysis ---
num_clusters = len(set(df_hot['cluster_label'])) - (1 if -1 in df_hot['cluster_label'] else 0)

if num_clusters > 0:
    print("\n--- Step 7: Saving Clusters to Database ---")
    
    # 1. Select relevant columns - creating a binary flag: Is this segment part of a systemic cluster?
    df_hot['Is_In_Cluster'] = df_hot['cluster_label'].apply(lambda x: 1 if x != -1 else 0)
    
    # 2. Save to DuckDB - Only need the SegmentID and the Cluster Flag
    df_export = df_hot[['SegmentID', 'Is_In_Cluster', 'Congestion_Index']]
    
    conn.register('df_dbscan_results', df_export)
    conn.execute("DROP TABLE IF EXISTS feature_congestion_clusters")
    conn.execute("CREATE TABLE feature_congestion_clusters AS SELECT * FROM df_dbscan_results")
    
    print("Saved table 'feature_congestion_clusters'.")

# --- 9. Close Connection ---
conn.close()
print("Database connection closed.")

# --- 10. Visualization ---
if num_clusters > 0:
    print("\n--- Step 8: Visualizing Results ---")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = set(df_hot['cluster_label'])
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # approx 500 feet (about 2 NYC blocks) to represent the reach of congestion
    VISUAL_RADIUS_FEET = 500 
    
    for k, col in zip(unique_labels, colors):
        class_member_mask = (df_hot['cluster_label'] == k)
        xy = df_hot[class_member_mask]
        
        if k == -1:
            # Noise: Just small dots, no boundary circles
            ax.scatter(xy['X_coord'], xy['Y_coord'], c='gray', s=10, alpha=0.3, label='Noise')
            continue
            
        # CLUSTERS: Draw Boundary Circles
        for _, row in xy.iterrows():
            circle = mpatches.Circle(
                (row['X_coord'], row['Y_coord']), 
                VISUAL_RADIUS_FEET, 
                color=col, 
                alpha=0.15,
                linewidth=0
            )
            ax.add_patch(circle)
            
        # 2. Draw the core point (The center of the road segment)
        ax.scatter(xy['X_coord'], xy['Y_coord'], c=[col], s=20, edgecolors='k', zorder=2, label=f'Cluster {k}')

    plt.title(f'Multi-Modal Hotspots (Eps Circles)\nRadius indicates extent of influence (~500ft)')
    plt.xlabel('X Coordinate (State Plane Feet)')
    plt.ylabel('Y Coordinate (State Plane Feet)')
    
    # Handle Legend (Remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal') # Important so circles don't look like ovals
    plt.show()
else:
    print("No clusters to visualize.")