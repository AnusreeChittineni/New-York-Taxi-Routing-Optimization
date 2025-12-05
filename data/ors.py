import openrouteservice as ors
import json
import geopandas as gpd
from shapely.geometry import Point
import duckdb       # Added
import pandas as pd # Added
import sys          # Added
import time

# --- Configuration ---

API_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjJhY2ViOTMwMGYxNTQ0NmNhNzdlYjdiMjcwNDFlN2M0IiwiaCI6Im11cm11cjY0In0='

# 1. Download the shapefile from:
#    https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-b9aa
# 2. Export it as a "Shapefile" (it will be a .zip).
# 3. Unzip it and put the path to the '.shp' file here.
TAXI_ZONES_SHAPEFILE = "/Users/smrithinarayan/Downloads/taxi_zones"
DB_PATH = "nyc_traffic_2016.duckdb"

def get_ors_route(api_key, coords, profile='driving-car'):
    """
    Calculates a route between two or more coordinates using the OpenRouteService API.
    """
    try:
        client = ors.Client(key=api_key)
        
        # print(f"Requesting route for {coords}...") # Commented out for cleaner logs
        route_response = client.directions(
            coordinates=coords,
            profile=profile,
            geometry=True,
            format='json' # Explicitly ask for JSON
        )
        # Add a small delay to respect API rate limits
        time.sleep(1.5) # ~40 requests per minute
        return route_response
        
    except Exception as e:
        print(f"  > API Error: {e}")
        # If we hit a rate limit, wait and retry once
        if "429" in str(e):
            print("  > Hit rate limit, sleeping for 20 seconds...")
            time.sleep(20)
            return get_ors_route(api_key, coords, profile) # Retry
        return None

def load_taxi_zones(shapefile_path):
    """
    Loads the Taxi Zone Shapefile into a GeoDataFrame.
    """
    print(f"Loading taxi zone shapefile from: {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path)
        gdf['LocationID'] = gdf['LocationID'].astype(int)
        gdf = gdf.set_index('LocationID')
        gdf = gdf.to_crs(epsg=4326) 
        print("Shapefile loaded and indexed successfully.")
        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {e}", file=sys.stderr)
        print("Please check the 'TAXI_ZONES_SHAPEFILE' path.", file=sys.stderr)
        return None

def get_centroid_for_id(gdf, location_id):
    """
    Finds the center (centroid) coordinates for a given LocationID.
    """
    try:
        zone = gdf.loc[location_id]
        geometry = zone.geometry
        centroid = geometry.centroid
        return (centroid.x, centroid.y)
    except KeyError:
        # This will happen for IDs not in the file
        return None
    except Exception as e:
        print(f"Error getting centroid: {e}", file=sys.stderr)
        return None


# --- NEW FUNCTION TO PROCESS UNIQUE JANUARY ROUTES ---

def generate_routes(api_key, db_path, taxi_zones_gdf, process_limit=50, offset=0):
    """
    Reads from 'unique_routes', gets routes from ORS,
    and saves them to a new table 'unique_routes'.
    """
    
    print(f"\n--- Starting route generation for {process_limit} unique routes ---")
    conn = None
    try:
        # 1. Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # 2. Create the new table to store the results
        # conn.execute("DROP TABLE IF EXISTS unique_routes_info;")
        # conn.execute("""
        #     CREATE TABLE unique_routes_info (
        #         PULocationID INTEGER,
        #         DOLocationID INTEGER,
        #         route_coordinates_json TEXT,   -- Stores a JSON list of [lon, lat] pairs
        #         route_distance_meters FLOAT,   -- Stores the distance
        #         route_duration_seconds FLOAT,  -- Stores the travel time
        #         PRIMARY KEY (PULocationID, DOLocationID)
        #     );
        # """)
        # print("Created new table 'unique_routes_info'.")
        
        # 3. Get the unique routes to process
        query = f"SELECT PULocationID, DOLocationID FROM nyc_traffic_2016.unique_routes LIMIT {process_limit} OFFSET {offset};"
        routes_to_process = conn.execute(query).fetchall()
        
        if not routes_to_process:
            print("No data found in 'unique_routes'. Aborting.")
            return

        print(f"Fetched {len(routes_to_process)} unique routes to process.")
        
        # 4. Loop, get routes, and insert into the new table
        insert_count = 0
        for i, (pu_id, do_id) in enumerate(routes_to_process):
            print(f"Processing route {i+1}/{len(routes_to_process)}: {pu_id} -> {do_id}", end="... ")
            
            # Get coordinates from the loaded shapefile
            start_coord = get_centroid_for_id(taxi_zones_gdf, pu_id)
            end_coord = get_centroid_for_id(taxi_zones_gdf, do_id)
            
            # Skip if either ID was invalid (e.g., no centroid)
            if not start_coord or not end_coord:
                print("SKIPPED (invalid ID)")
                continue
                
            coordinates = (start_coord, end_coord)
            
            # Get route from ORS
            route_info = get_ors_route(api_key, coordinates)
            
            if route_info and 'routes' in route_info and route_info['routes']:
                # Extract the *encoded* polyline (more efficient for storage)
                encoded_polyline = route_info['routes'][0]['geometry']
                decoded_geometry = ors.convert.decode_polyline(encoded_polyline)
                route_coordinates = decoded_geometry['coordinates']

                downsampled_coords = route_coordinates[::5]
                coords_json = json.dumps(downsampled_coords)

                
                # Extract summary data
                summary = route_info['routes'][0]['summary']
                distance = summary.get('distance', None) # in meters
                duration = summary.get('duration', None) # in seconds
                
                # Insert all data into the new 'unique_routes_info' table
                conn.execute("""
                    INSERT INTO unique_routes_info VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (PULocationID, DOLocationID) DO NOTHING;
                """, (
                    pu_id,
                    do_id,
                    coords_json,
                    distance,
                    duration
                ))
                insert_count += 1
                print("DONE")
            else:
                print("SKIPPED (no route found)")

        print(f"\n--- Process complete. ---")
        print(f"Successfully inserted {insert_count} routes into 'unique_routes_info'.")
        
        # 5. Show a sample from the new table
        print("\nSample from new 'unique_routes_info' table:")
        print(conn.execute("SELECT * FROM unique_routes_info;").fetchdf())

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
            print("\nDuckDB connection closed.")


# --- Main execution (Updated) ---
if __name__ == "__main__":
    if API_KEY == 'YOUR_ORS_API_KEY' or TAXI_ZONES_SHAPEFILE == "path/to/your/folder/taxi_zones.shp":
        print("Please set your 'API_KEY' and 'TAXI_ZONES_SHAPEFILE' path at the top of the script.", file=sys.stderr)
    else:
        
        # Step 1: Load the geographic data (this is loaded into memory once)
        taxi_zones_gdf = load_taxi_zones(TAXI_ZONES_SHAPEFILE)
        
        if taxi_zones_gdf is not None:
            
            # Step 2: Call the new function to build the table
            #         This will process the top 50 routes by default.
            #         You can change this number, but be careful of API limits!
            generate_routes(
                api_key=API_KEY,
                db_path=DB_PATH,
                taxi_zones_gdf=taxi_zones_gdf,
                process_limit=1000, 
                offset = 200
            )