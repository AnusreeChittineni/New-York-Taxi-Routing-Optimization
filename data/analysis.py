import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
DB_PATH = "nyc_traffic_2016.duckdb"

def run_analysis():
    """
    Connects to the DuckDB database, runs all analyses, 
    and generates plots.
    """
    conn = None
    try:
        # 1. Connect to the database
        conn = duckdb.connect(DB_PATH, read_only=True)
        print(f"Connected to {DB_PATH} (read-only)")

        
        # --- 1. Headline Stats ---
        print("\n--- 📊 Headline Statistics (Full 2016) ---")
        
        # Get Total Trips from taxi_data
        try:
            total_trips = conn.execute("SELECT COUNT(*) FROM taxi_data;").fetchone()[0]
            print(f"Total Trips: {total_trips:,}")
        except Exception as e:
            print(f"Error fetching total trips (is 'taxi_data' table present?): {e}")

        # Get Total Unique Routes from unique_routes
        try:
            unique_routes_count = conn.execute("SELECT COUNT(*) FROM unique_routes;").fetchone()[0]
            print(f"Total Unique Routes (A->B): {unique_routes_count:,}")
        except Exception as e:
            print(f"Error fetching unique routes (run 'create_unique_routes.py' first?): {e}")


        # --- 2. Top 10 Route Distribution Plot ---
        print("\n--- 📈 Generating Top 50 Route Distribution Plot ---")
        
        # Get Top 10 routes
        df_top10 = conn.execute("SELECT * FROM unique_routes LIMIT 50").fetchdf()

        # Get the sum of all other routes
        rest_count_df = conn.execute("""
            SELECT SUM(trip_count) 
            FROM (
                SELECT trip_count 
                FROM unique_routes 
                OFFSET 50
            );
        """).fetchdf()
        
        # Add a safeguard for the 'Rest' count
        if rest_count_df.empty:
            rest_count = 0
        else:
            rest_count_value = rest_count_df.iloc[0, 0]
            if rest_count_value is None:
                rest_count = 0
            else:
                rest_count = int(rest_count_value)
        
        # Create a "Rest" row
        rest_row = pd.DataFrame([{'PULocationID': 'Rest', 'DOLocationID': '', 'trip_count': rest_count}])
        
        # Combine for plotting
        df_plot_dist = pd.concat([df_top10, rest_row], ignore_index=True)
        
        # Create a text label
        df_plot_dist['route_label'] = df_plot_dist['PULocationID'].astype(str) + ' -> ' + df_plot_dist['DOLocationID'].astype(str)
        df_plot_dist.loc[df_plot_dist['PULocationID'] == 'Rest', 'route_label'] = 'Rest of Routes'
        
        # Plot
        plt.figure(figsize=(12, 7))
        bars = plt.bar(df_plot_dist['route_label'], df_plot_dist['trip_count'])
        plt.title('Top 50 Route Distribution (vs. The Rest)', fontsize=16)
        plt.ylabel('Total Number of Trips (Log Scale)')
        plt.xlabel('Route (Pickup ID -> Dropoff ID)')
        plt.xticks(rotation=45, ha='right')
        plt.yscale('log') # Use a log scale, as the "Rest" bar is huge
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        print("... Plot 1 generated.")
        
        
        # --- 3. Real vs. Potential Time Comparison ---
        print("\n--- ⏱️ Generating Real vs. Potential Time Comparison ---")
        
        comparison_query = """
            WITH Top10Routes AS (
                -- Get the Top 10 most popular routes
                SELECT PULocationID, DOLocationID, trip_count
                FROM unique_routes 
                LIMIT 10
            )
            SELECT 
                t.PULocationID,
                t.DOLocationID,
                ANY_VALUE(top.trip_count) AS num_trips,
                
                -- Calculate average real duration in seconds
                AVG(epoch(t.tpep_dropoff_datetime) - epoch(t.tpep_pickup_datetime)) AS avg_real_duration_sec,
                
                -- Get the single ORS duration value from your info table
                ANY_VALUE(info.route_duration_seconds) AS ors_potential_duration_sec
                
            FROM taxi_data AS t
            
            JOIN Top10Routes AS top 
                ON t.PULocationID = top.PULocationID AND t.DOLocationID = top.DOLocationID
            
            LEFT JOIN unique_routes_info AS info 
                ON t.PULocationID = info.PULocationID AND t.DOLocationID = info.DOLocationID
            
            WHERE
                -- Filter out bad data: trips < 30 sec or > 4 hours
                (epoch(t.tpep_dropoff_datetime) - epoch(t.tpep_pickup_datetime)) > 30
                AND (epoch(t.tpep_dropoff_datetime) - epoch(t.tpep_pickup_datetime)) < (4 * 3600)
            
            GROUP BY 
                t.PULocationID, t.DOLocationID
            
            ORDER BY 
                num_trips DESC;
        """
        
        try:
            df_comparison = conn.execute(comparison_query).fetchdf()
        except Exception as e:
            print(f"Error running comparison query (is 'unique_routes_info' present?): {e}")
            return # Exit if this fails

        # Convert seconds to minutes for plotting
        df_comparison['avg_real_min'] = df_comparison['avg_real_duration_sec'] / 60
        df_comparison['ors_potential_min'] = df_comparison['ors_potential_duration_sec'] / 60
        df_comparison['route_label'] = df_comparison['PULocationID'].astype(str) + ' -> ' + df_comparison['DOLocationID'].astype(str)

        # Plot
        df_plot_time = df_comparison[['route_label', 'avg_real_min', 'ors_potential_min']]
        df_plot_time.set_index('route_label').plot(kind='bar', figsize=(14, 8))
        
        plt.title('Real Trip Time vs. Potential (Ideal) Time', fontsize=16)
        plt.ylabel('Average Trip Duration (Minutes)')
        plt.xlabel('Route (Pickup ID -> Dropoff ID)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(['Average Real Trip Time', 'ORS Potential (Ideal) Time'], fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        print("... Plot 2 generated.")

        
# --- 4. Inefficiency Score (Bonus Insight) ---
        print("\n--- 🚦 Top 10 Routes by Inefficiency Score ---")
        
        # An Inefficiency Score of 1.5x means the real trip
        # takes 50% longer than the ideal ORS route.
        df_comparison['inefficiency_score'] = (
            df_comparison['avg_real_duration_sec'] / df_comparison['ors_potential_duration_sec']
        )
        
        # Sort by the most inefficient routes
        # --- THIS IS THE FIX ---
        df_inefficient = df_comparison.sort_values(by='inefficiency_score', ascending=False)
        
        print(df_inefficient[['route_label', 'avg_real_min', 'ors_potential_min', 'inefficiency_score']].to_string(index=False))

    # --- THIS IS THE FIX: All code above is now inside the 'try' block ---

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")
        
        # Show both plots at the end (must be inside the function)
        plt.show()

# --- Main execution ---
if __name__ == "__main__":
    run_analysis()