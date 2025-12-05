import duckdb
import pandas as pd

# --- 1. Connect to DuckDB ---
DB_PATH = "../nyc_routing.duckdb"
conn = duckdb.connect(DB_PATH)
print("Connected to DuckDB.")

# --- 2. Load CSV ---
csv_path = "../../bad_edges.csv"
edges_df = pd.read_csv(csv_path)

tolerance = 0.0005  # ~50 meters

# --- 3. Map u_lat/u_lon to PULocationID ---
def get_pulocation(lat, lon):
    query = f"""
        SELECT DISTINCT PULocationID
        FROM taxi_clean
        WHERE pickup_lat BETWEEN {lat - tolerance} AND {lat + tolerance}
          AND pickup_lon BETWEEN {lon - tolerance} AND {lon + tolerance}
    """
    res = conn.execute(query).fetchall()
    return res[0][0] if res else None

edges_df['PULocationID'] = edges_df.apply(lambda row: get_pulocation(row['u_lat'], row['u_lon']), axis=1)

# --- 4. Map PULocationID to SegmentID ---
pu_ids = edges_df['PULocationID'].dropna().unique()
pu_ids_str = ','.join(str(int(p)) for p in pu_ids)

query = f"""
    SELECT SegmentID, PULocationID
    FROM segment_to_zone_lookup
    WHERE PULocationID IN ({pu_ids_str})
"""
pu_to_segment = conn.execute(query).fetchdf()

# ---  Ensure consistent types for merging ---
edges_df['PULocationID'] = edges_df['PULocationID'].astype('Int64')
pu_to_segment['PULocationID'] = pu_to_segment['PULocationID'].astype('Int64')

# --- Merge ---
edges_df = edges_df.merge(pu_to_segment, on='PULocationID', how='left')


# --- 5. Check SegmentID in trips_with_traffic ---
def puid_in_traffic(pu_id):
    if pd.isna(pu_id):
        return False
    count = conn.execute(f"SELECT COUNT(*) FROM trips_with_traffic WHERE PULocationID={int(pu_id)}").fetchone()[0]
    return count > 0

edges_df['puid_in_traffic'] = edges_df['PULocationID'].apply(puid_in_traffic)

# --- 6. Check if coordinates are in trips_with_closures ---
def in_collision(lat, lon):
    query = f"""
        SELECT COUNT(*)
        FROM trips_with_closures
        WHERE pickup_lat BETWEEN {lat - tolerance} AND {lat + tolerance}
          AND pickup_lon BETWEEN {lon - tolerance} AND {lon + tolerance}
    """
    count = conn.execute(query).fetchone()[0]
    return count > 0

edges_df['Collision_at_start'] = edges_df.apply(lambda row: in_collision(row['u_lat'], row['u_lon']), axis=1)


# --- 7. Summary ---
print("Total edges:", len(edges_df))
print("Pick up Location IDs found in trips_with_traffic:", edges_df['Segment_in_traffic'].sum())
print("Edges starting in trips_with_closures:", edges_df['Collision_at_start'].sum())


# Optional: save
edges_df.to_csv("bad_edges_segment_check.csv", index=False)

# --- 7. Close connection ---
conn.close()
print("DuckDB connection closed.")
