import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DB_PATH = "nyc_routing.duckdb"

# --- DuckDB Aggregation Functions ---

def get_temporal_demand(con):
    """Queries trips aggregated by hour and day of week."""
    print("Querying temporal demand distributions...")
    query = """
    SELECT 
        EXTRACT(HOUR FROM tpep_pickup_datetime) AS hour,
        EXTRACT(DOW FROM tpep_pickup_datetime) AS day_of_week_num,
        CASE EXTRACT(DOW FROM tpep_pickup_datetime)
            WHEN 0 THEN 'Sun' WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue' 
            WHEN 3 THEN 'Wed' WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri' 
            WHEN 6 THEN 'Sat' END AS day_of_week,
        COUNT(*) AS trips
    FROM taxi_clean
    GROUP BY ALL
    ORDER BY day_of_week_num, hour;
    """
    return con.execute(query).fetchdf()

def get_spatial_hotspots(con, sample_size=100000):
    """Queries a sample of coordinates for heatmap/scatter plotting."""
    print(f"Sampling {sample_size:,} rows for spatial analysis...")
    query = f"""
    SELECT 
        pickup_lon, 
        pickup_lat, 
        PULocationID, 
        DOLocationID 
    FROM taxi_clean
    LIMIT {sample_size};
    """
    return con.execute(query).fetchdf()

def get_top_od_pairs(con, top_n=10):
    """Queries the top N Origin-Destination pairs based on location IDs."""
    print(f"Querying Top {top_n} Origin-Destination Pairs...")
    query = f"""
    SELECT 
        PULocationID, 
        DOLocationID, 
        COUNT(*) AS trip_count
    FROM taxi_clean
    WHERE PULocationID IS NOT NULL AND DOLocationID IS NOT NULL
    GROUP BY 1, 2
    ORDER BY trip_count DESC
    LIMIT {top_n};
    """
    return con.execute(query).fetchdf()

def get_vendor_market_share(con):
    """Queries the total trip count per VendorID."""
    print("Querying Vendor Market Share...")
    query = """
    SELECT 
        CAST(VendorID AS VARCHAR) AS vendor,
        COUNT(*) AS trips
    FROM taxi_clean
    WHERE VendorID IS NOT NULL
    GROUP BY 1
    ORDER BY trips DESC;
    """
    return con.execute(query).fetchdf()

# --- Plotting Functions ---

def plot_temporal(df):
    """Plots Hourly Demand and Day of Week Demand."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Hourly Demand
    hourly_df = df.groupby('hour')['trips'].sum().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='hour', y='trips', data=hourly_df, marker='o', color='darkblue')
    plt.title("Taxi Trip Demand by Hour of Day")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Total Trips")
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()

    # Plot 2: Demand Heatmap (Hour vs. Day of Week)
    # Pivot for heatmap
    heatmap_data = df.pivot_table(index='hour', columns='day_of_week', values='trips')
    
    # Order the columns correctly (Mon-Sun)
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heatmap_data = heatmap_data[day_order]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        cmap='rocket_r', 
        linecolor='white', 
        linewidths=0.5, 
        cbar_kws={'label': 'Total Trips'}
    )
    plt.title("Demand Heatmap: Hour of Day vs. Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Hour of Day")
    plt.yticks(rotation=0)
    plt.tight_layout()

def plot_spatial(df_coords, df_top_od):
    """Plots Spatial Heatmap and Top O-D Pairs."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 3: Pickup Location Heatmap (Density)
    # Filter coordinates to typical NYC boundaries for a clean map
    nyc_coords = df_coords[
        (df_coords['pickup_lon'] > -74.05) & (df_coords['pickup_lon'] < -73.85) &
        (df_coords['pickup_lat'] > 40.7) & (df_coords['pickup_lat'] < 40.9)
    ]
    
    plt.figure(figsize=(8, 8))
    sns.kdeplot(
        x=nyc_coords['pickup_lon'], 
        y=nyc_coords['pickup_lat'], 
        cmap='magma', 
        fill=True, 
        thresh=0.05, 
        levels=100
    )
    plt.title("Density Map of Taxi Pickup Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    

    # Plot 4: Top Origin-Destination Pairs
    # Create a unique label for O-D pair
    df_top_od['OD_Pair'] = df_top_od['PULocationID'].astype(str) + ' -> ' + df_top_od['DOLocationID'].astype(str)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='trip_count', y='OD_Pair', data=df_top_od, palette='viridis')
    plt.title(f"Top {len(df_top_od)} Most Frequent Origin-Destination (O-D) Trips")
    plt.xlabel("Total Trip Count")
    plt.ylabel("PU Location ID -> DO Location ID")
    plt.tight_layout()
    


def plot_impact_and_correlation(df_vendor):
    """Plots Vendor Market Share and Trip Distance Distribution (as an example)."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 5: Vendor Market Share (Pie Chart)
    plt.figure(figsize=(7, 7))
    plt.pie(
        df_vendor['trips'], 
        labels=df_vendor['vendor'], 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=sns.color_palette('Pastel1')
    )
    plt.title("Vendor Market Share by Trip Count")
    plt.tight_layout()

def main():
    try:
        con = duckdb.connect(DB_PATH)

        # --- Aggregation Steps ---
        df_temporal = get_temporal_demand(con)
        df_coords = get_spatial_hotspots(con)
        df_top_od = get_top_od_pairs(con)
        df_vendor = get_vendor_market_share(con)
        
        con.close()
        print("\nData aggregation complete. Generating plots...")

        # --- Visualization Steps ---
        
        # Option 1: Temporal Analysis
        plot_temporal(df_temporal)

        # Option 3: Spatial Hotspot Analysis
        plot_spatial(df_coords, df_top_od)

        # Option 4: Correlation & Impact Analysis (Vendor Share)
        plot_impact_and_correlation(df_vendor)

        plt.show()

    except Exception as e:
        print(f"\nAn unrecoverable error occurred: {e}")
        print("Please ensure your 'taxi_clean' table exists and the DuckDB connection is valid.")
    

if __name__ == "__main__":
    main()