import pandas as pd
import argparse
from osrm import OsrmClient
from tqdm import tqdm
import os
from os import path
def compute_osrm_fets(osrm_server_base_url: str, trips: pd.DataFrame):
    """
    Compute travel time in seconds and travel distance in meters for the fastest route for each trip in trips

    New columns:
    - trip_duration
    - trip_distance
    - steps

    """
    osrm_fets = {'total_distance': [], 'total_travel_time': [], 'number_of_steps': []}

    with OsrmClient(base_url=osrm_server_base_url) as osrm:
        for i, row in tqdm(trips.iterrows(), total=trips.shape[0]):
            coordinates = [(row['pickup_longitude'], row['pickup_latitude']), (row['dropoff_longitude'], row['dropoff_latitude'])]

            print(coordinates)
            osrm_route = osrm.route(coordinates, steps=False)
            osrm_fets['total_distance'].append(osrm_route.routes[0].distance)
            osrm_fets['total_travel_time'].append(osrm_route.routes[0].duration)
            # osrm_fets['number_of_steps'].append(sum([len(leg.steps) for leg in osrm_route.routes[0].legs]))

    return osrm_fets


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='extract_osrm_feats.py',
                    description='compute trip time and duration for every trip in a CSV file containing taxi trips')
    parser.add_argument('--input_csv', help='path to input csv file storing trip pickup and dropoff coordinates')      
    parser.add_argument('--output_csv', help='path to the output csv')
    parser.add_argument('--osrm_server_base_url', help='self hosted osrm server base url')
    
    args = parser.parse_args()

    trips_df = pd.read_csv(args.input_csv)

    if not path.exists(path.dirname(args.output_csv)):
        os.makedirs(os.path.dirname(args.output_csv))
    
    osrm_fets =  compute_osrm_fets(args.osrm_server_base_url, trips_df)
    osrm_fets_df = pd.DataFrame.from_dict(osrm_fets)
    trips_osrm_df = pd.concat((trips_df, osrm_fets_df), axis=1)
    
    trips_osrm_df.to_csv(args.output_csv)
 
