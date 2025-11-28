import duckdb

# ---------------------------------------
# Connect to the DuckDB warehouse
# ---------------------------------------
DB_PATH = "nyc_routing.duckdb"
con = duckdb.connect(DB_PATH)
print(f"Connected to {DB_PATH}\n")

print("=== TABLES PRESENT IN WAREHOUSE ===")
tables = con.execute("SHOW TABLES").fetchall()
table_list = [t[0] for t in tables]
print(table_list, "\n")

# ---------------------------------------
# Helper function for table summaries
# ---------------------------------------
def explore_table(table_name):
    print(f"=== TABLE: {table_name} ===")

    # Check if table exists
    exists = con.execute(f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_name = '{table_name.lower()}'
    """).fetchone()[0]

    if exists == 0:
        print("Table does not exist.\n")
        return
    
    # Columns
    try:
        df = con.execute(f"SELECT * FROM {table_name} LIMIT 0").df()
        print("Columns:", list(df.columns))
    except Exception as e:
        print(f"(Could not read columns: {e})")

    # Row count
    try:
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"Total rows: {count:,}")
    except:
        print("(Could not compute row count.)")

    # Example rows
    try:
        examples = con.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
        print("\nExample rows:")
        print(examples)
    except:
        print("(Could not fetch example rows.)")

    print("\n")


# ---------------------------------------
# Explore all tables
# ---------------------------------------
tables = [
    "taxi_raw",
    "traffic_2016",
    "collisions_2016",
    "closed_streets"
]

for t in tables:
    explore_table(t)

# ---------------------------------------
# Additional Stats / Useful Exploration
# ---------------------------------------

print("=== ADDITIONAL ANALYSIS ===\n")

# --- Taxi Stats ---
try:
    print("Taxi Stats:")
    print("Unique pickup locations:",
        con.execute("SELECT COUNT(DISTINCT pulocationid) FROM taxi_raw").fetchone()[0])

    print("Unique dropoff locations:",
        con.execute("SELECT COUNT(DISTINCT dolocationid) FROM taxi_raw").fetchone()[0])

    print("Average trip distance:",
        con.execute("SELECT AVG(trip_distance) FROM taxi_raw").fetchone()[0])

    print()
except Exception as e:
    print("Taxi stats unavailable:", e, "\n")

# --- Traffic Stats ---
try:
    print("Traffic Stats:")
    print("Unique traffic links:",
        con.execute("SELECT COUNT(DISTINCT link_id) FROM traffic_2016").fetchone()[0])

    print("Average volume:",
        con.execute("SELECT AVG(volume) FROM traffic_2016").fetchone()[0])

    print()
except Exception as e:
    print("Traffic stats unavailable:", e, "\n")

# --- Collision Stats ---
try:
    print("Collision Stats:")
    print("Unique streets involved:",
        con.execute("SELECT COUNT(DISTINCT street_upper) FROM collisions_2016").fetchone()[0])

    print("Total injuries:",
        con.execute("SELECT SUM(number_of_persons_injured) FROM collisions_2016").fetchone()[0])

    print("Total fatalities:",
        con.execute("SELECT SUM(number_of_persons_killed) FROM collisions_2016").fetchone()[0])

    print()
except Exception as e:
    print("Collision stats unavailable:", e, "\n")


print("\n=== Exploration Complete ===")

