import pandas as pd
import argparse
from osrm import OsrmClient
from tqdm import tqdm
import os
from os import path
from multiprocessing import Pool, cpu_count


def _compute_route_for_trip(args):
    """
    Worker function for a single trip.

    args: (osrm_server_base_url,
           pickup_longitude, pickup_latitude,
           dropoff_longitude, dropoff_latitude)
    """
    (
        osrm_server_base_url,
        pickup_longitude,
        pickup_latitude,
        dropoff_longitude,
        dropoff_latitude,
    ) = args

    with OsrmClient(base_url=osrm_server_base_url) as client: 
        try:

            coordinates = [
                (pickup_longitude, pickup_latitude),
                (dropoff_longitude, dropoff_latitude),
            ]

            osrm_route = client.route(coordinates, steps=False)
            route = osrm_route.routes[0]

            trip_distance = route.distance        # meters
            trip_duration = route.duration        # seconds

            return trip_distance, trip_duration
        except:
            return 0, 0

def compute_osrm_fets(
    osrm_server_base_url: str,
    trips: pd.DataFrame,
    num_workers: int | None = None,
    chunksize: int = 100,
):
    """
    Compute travel time in seconds and travel distance in meters for the fastest
    route for each trip in trips, in parallel.

    New columns:
    - trip_duration
    - trip_distance
    - steps
    """

    if trips.empty:
        return {
            "trip_distance": [],
            "trip_duration": [],
            "steps": [],
        }

    if num_workers is None:
        num_workers = cpu_count()

    # Build an iterator of arguments for each worker
    args_iter = (
        (
            osrm_server_base_url,
            row["pickup_longitude"],
            row["pickup_latitude"],
            row["dropoff_longitude"],
            row["dropoff_latitude"],
        )
        for _, row in trips.iterrows()
    )

    n_trips = len(trips)

    osrm_fets = {
        "trip_distance": [],
        "trip_duration": [],
        "steps": [],
    }

    with Pool(processes=num_workers) as pool:
        for trip_distance, trip_duration  in tqdm(
            pool.imap(_compute_route_for_trip, args_iter, chunksize=chunksize),
            total=n_trips,
        ):
            osrm_fets["trip_distance"].append(trip_distance)
            osrm_fets["trip_duration"].append(trip_duration)
    return osrm_fets


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="extract_osrm_feats.py",
        description=(
            "Compute trip time and distance for every trip in a CSV file "
            "containing taxi trips, using an OSRM server."
        ),
    )
    parser.add_argument(
        "--input_csv",
        help="path to input csv file storing trip pickup and dropoff coordinates",
        required=True,
    )
    parser.add_argument(
        "--output_csv",
        help="path to the output csv",
        required=True,
    )
    parser.add_argument(
        "--osrm_server_base_url",
        help="self hosted osrm server base url",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100,
        help="Chunksize for multiprocessing.imap (default: 100)",
    )

    args = parser.parse_args()

    trips_df = pd.read_csv(args.input_csv)

    if not path.exists(path.dirname(args.output_csv)):
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    osrm_fets = compute_osrm_fets(
        args.osrm_server_base_url,
        trips_df,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )

    osrm_fets_df = pd.DataFrame.from_dict(osrm_fets)
    trips_osrm_df = pd.concat((trips_df, osrm_fets_df), axis=1)

    trips_osrm_df.to_csv(args.output_csv, index=False)
