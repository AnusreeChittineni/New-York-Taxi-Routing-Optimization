import duckdb
import time
import os

# --- Configuration ---
DB_PATH = "../nyc_routing.duckdb"
TAXI_ZONES_FILE = "NYC_Taxi_Zones_20251204.csv" 

def load_taxi_zones(con, file_path):
    """
    Loads the Taxi Zone geometry data from a CSV file into the 'taxi_zones' table.
    """
    
    if not os.path.exists(file_path):
        print(f"Error: Taxi Zones file not found at path: {file_path}")
        return

    print("\n--- STAGE: LOADING TAXI ZONE GEOMETRY ---")
    start_time = time.time()
    
    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE taxi_zones AS
        SELECT
            "Location ID" AS PULocationID, -- Use consistent naming for the ID field
            "Zone" AS zone_name,
            "Borough" AS borough,
            "Shape Geometry" AS shape_geometry, -- Contains the MULTIPOLYGON WKT
            "Shape Area" AS shape_area
        FROM read_csv_auto('{file_path}')
        -- Filter out any non-standard zone (like the airport geometry rows if present)
        WHERE "Location ID" IS NOT NULL AND "Zone" IS NOT NULL;
        """)

        elapsed_time = time.time() - start_time
        row_count = con.execute("SELECT COUNT(*) FROM taxi_zones").fetchone()[0]
        examples = con.execute("SELECT * FROM taxi_zones LIMIT 2").fetchdf()
        
        print(f"Table 'taxi_zones' loaded successfully.")
        print(f"Total taxi zones loaded: {row_count:,}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print("Example rows:", examples)

    except Exception as e:
        print(f"An error occurred during table creation: {e}")
        print("Check if the column headers in the CSV exactly match the query.")

    return row_count

con = duckdb.connect(DB_PATH)
load_taxi_zones(con, TAXI_ZONES_FILE)
con.close()