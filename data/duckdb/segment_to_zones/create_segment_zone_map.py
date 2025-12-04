import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkt

# --------------------------------------------------------
# CONNECT TO DUCKDB
# --------------------------------------------------------
DB_PATH = "../nyc_routing.duckdb"
con = duckdb.connect(DB_PATH)

# --------------------------------------------------------
# LOAD NYC TAXI ZONES FROM DUCKDB
# --------------------------------------------------------
print("Loading Taxi Zones from DuckDB...")

taxi_zones_df = con.execute("SELECT PULocationID, shape_geometry FROM taxi_zones").fetchdf()

# Convert WKT to geometry
taxi_zones_df["geometry"] = taxi_zones_df["shape_geometry"].apply(wkt.loads)
taxi_zones = gpd.GeoDataFrame(taxi_zones_df, geometry="geometry", crs="EPSG:4326")

# Reproject to EPSG:2263 (matches ATVC points)
taxi_zones_2263 = taxi_zones.to_crs("EPSG:2263")

print(f"Taxi Zones Loaded: {len(taxi_zones_2263)}")

# --------------------------------------------------------
# LOAD ATVC POINTS FROM DUCKDB
# --------------------------------------------------------
print("Loading ATVC points from DuckDB...")

atvc_df = con.execute("SELECT SegmentID, WktGeom FROM traffic_2016 WHERE WktGeom IS NOT NULL").fetchdf()
atvc_df["geometry"] = atvc_df["WktGeom"].apply(wkt.loads)
atvc_points = gpd.GeoDataFrame(atvc_df, geometry="geometry", crs="EPSG:2263")

print(f"ATVC Points Loaded: {len(atvc_points)}")

# --------------------------------------------------------
# SPATIAL JOIN
# --------------------------------------------------------
print("Performing spatial join (Point-in-Polygon)...")

joined = gpd.sjoin(
    atvc_points,
    taxi_zones_2263[["PULocationID", "geometry"]],
    how="left",
    predicate="within"
)

matched_count = joined["PULocationID"].notna().sum()
print(f"Matched points: {matched_count} / {len(joined)}")

# --------------------------------------------------------
# CLEAN OUTPUT
# --------------------------------------------------------
lookup = joined[["SegmentID", "PULocationID"]].drop_duplicates()

print("Example mappings:")
print(lookup.head())

# --------------------------------------------------------
# SAVE RESULT
# --------------------------------------------------------
lookup.to_csv("segment_to_pulocationid_lookup.csv", index=False)
print("\nSaved lookup table → segment_to_pulocationid_lookup.csv")

