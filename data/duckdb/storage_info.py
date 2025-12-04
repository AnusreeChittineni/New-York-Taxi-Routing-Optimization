import duckdb
import pandas as pd

DB_PATH = "nyc_routing.duckdb"
con = duckdb.connect(DB_PATH)

def get_storage_info(con):
    """
    Prints the total disk size of the DuckDB file and lists all tables 
    with their respective row counts.
    """
    
    print("\n--- DUCKDB STORAGE AND TABLE INFORMATION ---")
    
    # 1. Get total database size on disk
    try:
        total_size = con.execute("PRAGMA database_size;").fetchone()[0]
        print(f"Total Database File Size: {total_size}")
    except Exception as e:
        print(f"Error getting database size: {e}")

    # 2. Get list of all tables
    tables_df = con.execute("""
        SELECT 
            table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main' 
        ORDER BY table_name;
    """).fetchdf()

    if tables_df.empty:
        print("No tables found in the main schema.")
        return

    # 3. Iterate and get row count for each table
    table_stats = []
    print("\nTable Row Counts:")
    for index, row in tables_df.iterrows():
        table_name = row['table_name']
        try:
            row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            table_stats.append({'Table': table_name, 'Rows': f"{row_count:,}"})
        except Exception as e:
            table_stats.append({'Table': table_name, 'Rows': f"Error: {e}"})

    print(pd.DataFrame(table_stats))

get_storage_info(con)
con.close()