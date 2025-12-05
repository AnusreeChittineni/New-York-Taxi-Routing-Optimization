import duckdb
import pandas as pd
from datetime import timedelta
from geopy.distance import geodesic
from itertools import product

# ----------------------------
# CONFIGURATION
# ----------------------------
MAX_CAPACITY = 4  # max passengers per taxi

# Scenarios to test
PICKUP_RADII_KM = [0.2, 0.4, 0.6]     # 200m, 400m, 600m
DROPOFF_RADII_KM = [0.2, 0.4, 0.6]
TIME_WINDOWS_MIN = [3, 5, 7]           # time difference in minutes

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def load_trips(conn, limit=50000):
    query = """
        SELECT
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            pickup_lat, pickup_lon,
            dropoff_lat, dropoff_lon,
            passenger_count
        FROM taxi_clean
        LIMIT ?
    """
    df = conn.execute(query, [limit]).fetchdf()
    df = df.reset_index(drop=True)
    df["trip_id"] = df.index
    return df

def km(p1, p2):
    return geodesic(p1, p2).km

# ----------------------------
# MATCHING ALGORITHM
# ----------------------------
def match_carpools(df, pickup_radius_km, dropoff_radius_km, time_window_min):
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    N = len(df)

    matches = []
    used = set()

    for i in range(N):
        if i in used:
            continue

        tripA = df.iloc[i]
        A_pick = (tripA.pickup_lat, tripA.pickup_lon)
        A_drop = (tripA.dropoff_lat, tripA.dropoff_lon)
        A_time = tripA.tpep_pickup_datetime
        A_passengers = tripA.passenger_count

        # Scan forward only within time window
        j = i + 1
        while j < N:
            tripB = df.iloc[j]
            B_time = tripB.tpep_pickup_datetime
            B_pick = (tripB.pickup_lat, tripB.pickup_lon)
            B_drop = (tripB.dropoff_lat, tripB.dropoff_lon)
            B_passengers = tripB.passenger_count

            # prune by time
            time_diff_min = abs((B_time - A_time).total_seconds() / 60)
            if time_diff_min > time_window_min:
                break

            # distance thresholds + passenger capacity
            total_passengers = A_passengers + B_passengers
            if (
                km(A_pick, B_pick) <= pickup_radius_km
                and km(A_drop, B_drop) <= dropoff_radius_km
                and total_passengers <= MAX_CAPACITY
            ):
                matches.append({
                    "tripA": tripA.trip_id,
                    "tripB": tripB.trip_id,
                    "pickup_distance_km": km(A_pick, B_pick),
                    "dropoff_distance_km": km(A_drop, B_drop),
                    "time_diff_min": time_diff_min,
                    "total_passengers": total_passengers
                })
                used.add(i)
                used.add(j)
                break

            j += 1

    return pd.DataFrame(matches)

# ----------------------------
# CONGESTION REDUCTION METRIC
# ----------------------------
def estimate_congestion(matches, total_trips):
    paired = len(matches)
    reduced = paired   # every pair removes 1 car from the road
    percent = reduced / total_trips * 100
    return {
        "total_trips": total_trips,
        "shared_pairs": paired,
        "reduced_trips": reduced,
        "percent_reduction": percent,
    }

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def run_carpooling(db_path, limit=50000):
    conn = duckdb.connect(db_path)
    df = load_trips(conn, limit)
    results = []

    # Iterate over all scenario combinations
    for pickup_radius, dropoff_radius, time_window in product(PICKUP_RADII_KM, DROPOFF_RADII_KM, TIME_WINDOWS_MIN):
        print(f"\nRunning scenario: pickup {pickup_radius} km, dropoff {dropoff_radius} km, time window {time_window} min")
        matches = match_carpools(df, pickup_radius, dropoff_radius, time_window)
        stats = estimate_congestion(matches, len(df))
        stats.update({
            "pickup_radius_km": pickup_radius,
            "dropoff_radius_km": dropoff_radius,
            "time_window_min": time_window
        })
        results.append(stats)
    
    return pd.DataFrame(results)

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    scenario_results = run_carpooling("../../data/duckdb/nyc_routing.duckdb", limit=50000)
    print("\n=== Scenario Results ===")
    print(scenario_results)
    scenario_results.to_csv("carpooling_scenario_results.csv", index=False)
