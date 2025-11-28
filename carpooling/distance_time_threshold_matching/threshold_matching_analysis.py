import duckdb
import pandas as pd
from datetime import timedelta
from geopy.distance import geodesic

# ----------------------------
# CONFIGURATION
# ----------------------------
PICKUP_RADIUS_KM = 0.4      # 400 meters
DROPOFF_RADIUS_KM = 0.4
TIME_WINDOW_MIN = 5         # allowable time difference between trips

# ----------------------------
# LOAD TRIP DATA FROM DUCKDB
# ----------------------------
def load_trips(conn, limit=50000):
    query = """
        SELECT
            trip_id,
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            pickup_lat, pickup_lon,
            dropoff_lat, dropoff_lon
        FROM taxi_clean
        LIMIT ?
    """
    return conn.execute(query, [limit]).fetchdf()


# ----------------------------
# DISTANCE CALCULATIONS
# ----------------------------
def km(p1, p2):
    return geodesic(p1, p2).km


# ----------------------------
# MATCHING ALGORITHM
# ----------------------------
def match_carpools(df):
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

        # Scan forward only within time window
        j = i + 1
        while j < N:
            tripB = df.iloc[j]
            B_time = tripB.tpep_pickup_datetime

            # prune by time
            if abs((B_time - A_time).total_seconds() / 60) > TIME_WINDOW_MIN:
                break

            B_pick = (tripB.pickup_lat, tripB.pickup_lon)
            B_drop = (tripB.dropoff_lat, tripB.dropoff_lon)

            # distance thresholds
            if km(A_pick, B_pick) < PICKUP_RADIUS_KM and km(A_drop, B_drop) < DROPOFF_RADIUS_KM:
                matches.append({
                    "tripA": tripA.trip_id,
                    "tripB": tripB.trip_id,
                    "pickup_distance_km": km(A_pick, B_pick),
                    "dropoff_distance_km": km(A_drop, B_drop),
                    "time_diff_min": abs((B_time - A_time).total_seconds() / 60)
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
def run_carpooling(db_path):
    conn = duckdb.connect(db_path)

    print("Loading trip sample...")
    df = load_trips(conn)

    print("Running distance threshold matching...")
    matches = match_carpools(df)

    print("Computing congestion reduction...")
    stats = estimate_congestion(matches, len(df))

    return matches, stats


if __name__ == "__main__":
    matches, stats = run_carpooling("../../data/duckdb/nyc_routing.duckdb")
    print("\n=== Carpool Matches ===")
    print(matches.head())

    print("\n=== Congestion Reduction ===")
    print(stats)
