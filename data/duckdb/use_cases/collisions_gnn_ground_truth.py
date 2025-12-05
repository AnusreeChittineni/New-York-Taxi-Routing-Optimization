import duckdb
import pandas as pd

# --- 1. Connect to DuckDB ---
DB_PATH = "../nyc_routing.duckdb"  # adjust your path
conn = duckdb.connect(DB_PATH)
print("Connected to DuckDB.")

# --- 2. Load GNN output CSV ---
gnn_csv_path = "../../nyc_road_safety_analysis.csv"
road_safety_df = pd.read_csv(gnn_csv_path)

# Ensure SegmentID column exists
if 'SegmentID' not in road_safety_df.columns:
    raise ValueError("CSV must contain a 'SegmentID' column")

# --- 3. Map SegmentID to PULocationID using segment_to_zone_lookup ---
segment_ids_str = ','.join(str(s) for s in road_safety_df['SegmentID'].unique())
query = f"""
    SELECT SegmentID, PULocationID
    FROM segment_to_zone_lookup
    WHERE SegmentID IN ({segment_ids_str})
"""
segment_to_pu = conn.execute(query).fetchdf()

# Merge PULocationID into road_safety_df
road_safety_df = road_safety_df.merge(segment_to_pu, on='SegmentID', how='left')

# --- 4. Check if each PULocationID exists in taxi_clean ---
def check_pulocation_exists(pu_id):
    if pd.isna(pu_id):
        return False
    result = conn.execute(f"""
        SELECT COUNT(*) 
        FROM taxi_clean
        WHERE PULocationID = {int(pu_id)}
    """).fetchone()[0]
    return result > 0

road_safety_df['PU_in_taxi_clean'] = road_safety_df['PULocationID'].apply(check_pulocation_exists)

# --- 5. Optional: check if SegmentID exists in traffic_2016 ---
def check_segment_exists(segment_id):
    result = conn.execute(f"""
        SELECT COUNT(*)
        FROM traffic_2016
        WHERE SegmentID = {int(segment_id)}
    """).fetchone()[0]
    return result > 0

road_safety_df['Segment_in_traffic'] = road_safety_df['SegmentID'].apply(check_segment_exists)

# --- 6. Output summary ---
total_segments = len(road_safety_df)
pu_present_count = road_safety_df['PU_in_taxi_clean'].sum()
segment_present_count = road_safety_df['Segment_in_traffic'].sum()

print(f"Total segments in CSV: {total_segments}")
print(f"PULocationIDs present in taxi_clean: {pu_present_count}")
print(f"SegmentIDs present in traffic_2016: {segment_present_count}")

# Optional: save full mapping table
road_safety_df.to_csv("segment_pulocation_check.csv", index=False)

# --- 7. Close connection ---
conn.close()
print("DuckDB connection closed.")