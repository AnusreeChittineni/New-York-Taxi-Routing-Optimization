import time
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# ----------------------------------------------------------
# Setup
# ----------------------------------------------------------
paths = {
    "1M": "bench/trips_1M.parquet",
    "5M": "bench/trips_5M.parquet",
}

results = []

# ----------------------------------------------------------
# DuckDB Benchmark
# ----------------------------------------------------------
def benchmark_duckdb(path):
    con = duckdb.connect()
    t0 = time.time()
    con.execute(f"SELECT COUNT(*) FROM '{path}'").fetchall()
    return time.time() - t0

print("Running DuckDB benchmarks...")
for label, path in paths.items():
    dur = benchmark_duckdb(path)
    print(f"DuckDB {label}: {dur:.3f} sec")
    results.append(["DuckDB", label, dur])


# ----------------------------------------------------------
# Spark Benchmark
# ----------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("SparkDuckDBBenchmark")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

def benchmark_spark(path):
    t0 = time.time()
    df = spark.read.parquet(path)
    _ = df.count()   # force full scan
    return time.time() - t0

# spark caches metadata
print("\nRunning Spark benchmarks...")
for label, path in paths.items():
    dur = benchmark_spark(path)
    print(f"Spark {label}: {dur:.3f} sec")
    results.append(["Spark", label, dur])

# ----------------------------------------------------------
# Results Table
# ----------------------------------------------------------
df_results = pd.DataFrame(results, columns=["Engine", "Rows", "Time (sec)"])
print("\n=== Benchmark Results ===")
print(df_results)

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
plt.figure(figsize=(7, 4))
for engine in df_results.Engine.unique():
    sub = df_results[df_results.Engine == engine]
    plt.plot(sub["Rows"], sub["Time (sec)"], marker="o", label=engine)

plt.xlabel("Dataset Size")
plt.ylabel("Time (seconds)")
plt.title("Spark vs DuckDB Read+Scan Benchmark")
plt.legend()
plt.tight_layout()
plt.show()
