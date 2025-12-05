import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
DB_PATH = "nyc_traffic_2016.duckdb"

def run_hourly_analysis():
    """
    Connects to the DuckDB database, analyzes the inefficiency
    by hour for the #1 most popular route, and generates a plot.
    """
    conn = None
    try:
        # 1. Connect to the database
        conn = duckdb.connect(DB_PATH, read_only=True)
        print(f"Connected to {DB_PATH} (read-only)")
        
        # --- 1. Find the #1 Most Popular Route ---
        print("\n--- 🚦 Finding most popular route to analyze ---")
        
        target_route = conn.execute("""
            SELECT PULocationID, DOLocationID 
            FROM unique_routes 
            ORDER BY trip_count DESC 
            LIMIT 1;
        """).fetchone()
        
        if not target_route:
            print("Error: Could not find any routes in 'unique_routes'.", file=sys.stderr)
            return

        pu_id, do_id = target_route
        print(f"Analyzing hourly traffic for route: {pu_id} -> {do_id}")

        # --- 2. Get the "Ideal" (ORS) Time for this route ---
        potential_time_row = conn.execute("""
            SELECT route_duration_seconds 
            FROM unique_routes_info 
            WHERE PULocationID = ? AND DOLocationID = ?;
        """, [pu_id, do_id]).fetchone()
        
        if not potential_time_row or not potential_time_row[0]:
            print(f"Error: No ORS route info found for {pu_id} -> {do_id}.", file=sys.stderr)
            print("Please run the 'openrouteservice' script for this route.", file=sys.stderr)
            return
            
        potential_time_sec = potential_time_row[0]
        print(f"Ideal (ORS) Time for this route: {potential_time_sec / 60:.2f} minutes")

        # --- 3. Get Real Average Time, Grouped by Hour ---
        print("... Analyzing all 2016 trips for this route by hour...")
        
        hourly_query = f"""
            SELECT 
                EXTRACT(hour FROM tpep_pickup_datetime) AS hour_of_day,
                AVG(epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime)) AS avg_real_duration_sec,
                COUNT(*) as num_trips
            FROM taxi_data
            WHERE 
                PULocationID = {pu_id} AND DOLocationID = {do_id}
                -- Filter out bad data
                AND (epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime)) > 30
                AND (epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime)) < (4 * 3600)
            GROUP BY 
                hour_of_day
            ORDER BY 
                hour_of_day;
        """
        df_hourly = conn.execute(hourly_query).fetchdf()

        # --- 4. Calculate Inefficiency Score ---
        # Create a full 24-hour DataFrame to ensure all hours are plotted
        all_hours = pd.DataFrame({'hour_of_day': range(24)})
        df_hourly = pd.merge(all_hours, df_hourly, on='hour_of_day', how='left')

        # Fill in the ideal time for all hours
        df_hourly['ors_potential_sec'] = potential_time_sec
        
        # For any hours with no trips (e.g., 4am), fill real-time with the ideal time
        # This gives them a "perfect" score of 1.0 instead of NaN (Not a Number)
        df_hourly['avg_real_duration_sec'] = df_hourly['avg_real_duration_sec'].fillna(potential_time_sec)
        
        # Calculate the final score
        df_hourly['inefficiency_score'] = df_hourly['avg_real_duration_sec'] / df_hourly['ors_potential_sec']
        
        print("\n--- Hourly Inefficiency Report ---")
        print(df_hourly[['hour_of_day', 'avg_real_duration_sec', 'inefficiency_score', 'num_trips']].to_string(index=False, na_rep="0"))

        # --- 5. Generate Plot ---
        plt.figure(figsize=(14, 7))
        colors = ['red' if x > 1.5 else 'orange' if x > 1.2 else 'green' for x in df_hourly['inefficiency_score']]
        
        plt.bar(df_hourly['hour_of_day'], df_hourly['inefficiency_score'], color=colors)
        
        # Add a baseline
        plt.axhline(y=1.0, color='blue', linestyle='--', label='Ideal (Score = 1.0)')
        
        plt.title(f'Hourly Inefficiency for Route {pu_id} -> {do_id}', fontsize=16)
        plt.xlabel('Hour of Day (0 = Midnight)')
        plt.ylabel('Inefficiency Score (Real Time / Ideal Time)')
        plt.xticks(range(24))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        print("\n... Plot generated.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")
        
        # Show the plot
        plt.show()

# --- Main execution ---
if __name__ == "__main__":
    run_hourly_analysis()