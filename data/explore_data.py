import duckdb

# Connect to a DuckDB database (in-memory or file-based)
con = duckdb.connect(database='nyc_traffic_2016.duckdb')  # use a file path for a persistent DB

tables = con.execute("SHOW TABLES").fetchall()
print("Tables:", [t[0] for t in tables])

df = con.execute("SELECT * FROM taxi_data LIMIT 0").df()
column_names = df.columns.tolist()
print("Column names from taxi_data:", column_names)

result = con.execute("SELECT * FROM taxi_data LIMIT 5").fetchall()
print(result)

df = con.execute("SELECT * FROM traffic_2016 LIMIT 0").df()
column_names = df.columns.tolist()
print("Column names from traffic_2016:", column_names)

df = con.execute("SELECT * FROM collisions_2016 LIMIT 0").df()
column_names = df.columns.tolist()
print("Column names from collisions_2016:", column_names)

result = con.execute("SELECT * FROM collisions_2016 LIMIT 5").fetchall()
print(result)

result = con.execute("SELECT * FROM street_status where status = 'closed' LIMIT 20").fetchall()
print(result)
