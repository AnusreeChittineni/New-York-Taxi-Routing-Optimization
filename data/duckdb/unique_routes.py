import duckdb
import sys

# Path to your existing DuckDB database
DB_PATH = "nyc_traffic_2016.duckdb"

def build_unique_routes():
    """
    Connects to the DuckDB database and builds a new table
    containing all unique PU/DO location ID pairs and their counts
    """
    conn = None
    try:
        # 1. Connect to the database
        conn = duckdb.connect(DB_PATH)
        print(f"Connected to {DB_PATH}")

        # 2. Run the query to create the new table
        print("Building 'unique_routes' table...")
        
        # Drop the table if it already exists so we can re-run this script
        conn.execute("DROP TABLE IF EXISTS unique_routes;")

        conn.execute("""
            CREATE TABLE unique_routes AS
            SELECT 
                PULocationID, 
                DOLocationID, 
                COUNT(*) AS trip_count
            FROM taxi_data
            WHERE                 
                -- Existing filters for valid routes
                PULocationID IS NOT NULL 
                AND DOLocationID IS NOT NULL 
                AND PULocationID < 264  -- Filter out 'Unknown' zone (264 & 265)
                AND DOLocationID < 264  -- Filter out 'Unknown' zone (264 & 265)
                AND PULocationID != DOLocationID -- Filter out trips that start/end in the same zone
            GROUP BY 
                PULocationID, DOLocationID
            ORDER BY 
                trip_count DESC;
        """)

        print("Successfully created 'unique_routes' table.")

        # 3. Print verification info
        total_unique_routes = conn.execute("SELECT COUNT(*) FROM unique_routes;").fetchone()[0]
        print(f"Total unique routes: {total_unique_routes:,}")

        print("\n--- Sample of most popular unique routes ---")
        sample_df = conn.execute("SELECT * FROM unique_routes LIMIT 10;").fetchdf()
        print(sample_df)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
            print(f"\nDatabase connection closed. 'unique_routes' is saved in {DB_PATH}")

# --- Main execution ---
if __name__ == "__main__":
    build_unique_routes()