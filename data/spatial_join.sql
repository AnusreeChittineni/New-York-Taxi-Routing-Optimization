-- Spatial and temporal join of three datasets - taxi trips, collisons, traffic counts

INSTALL spatial;
LOAD spatial;

CREATE OR REPLACE VIEW taxi_data_geom AS SELECT
    strptime(tpep_pickup_datetime, '%m/%d/%Y %-H:%-M:%-S %p') AS PICKUP_DATETIME,
     strptime(tpep_dropoff_datetime, '%m/%d/%Y %-H:%-M:%-S %p') AS DROPOFF_DATETIME,
     VendorID AS vendor_id,
     passenger_count,
     PULocationID as pu_location_id,
     DOLocationID as do_location_id,
     pickup_longitude,
     pickup_latitude,
     dropoff_longitude,
     dropoff_latitude,
     ST_Point(pickup_latitude, pickup_longitude) AS pickup_location,
     ST_Point(dropoff_latitude, dropoff_longitude) as dropoff_location,
     ST_Point((ST_X(pickup_location) + ST_X(dropoff_location)) / 2, (ST_Y(pickup_location) + ST_Y(dropoff_location)) / 2) as trip_midpoint,
     store_and_fwd_flag,
     trip_distance,
     dropoff_datetime - pickup_datetime AS trip_duration
  FROM nyc_traffic_2016.taxi_data;

CREATE OR REPLACE VIEW traffic_2016_std AS SELECT
  make_timestamp(
	    CAST(Yr AS INTEGER),
	    CAST(M  AS INTEGER),
	    CAST(D  AS INTEGER),
	    CAST(HH AS INTEGER),
	    CAST(MM AS INTEGER),
	    0  -- seconds assumed 0
	) AS event_time,
  Boro as traffic_borough,
  Vol as traffic_vol,
  SegmentID as traffic_segment_id,
  ST_GeomFromText(WktGeom) as traffic_location,
  street as traffic_street,
  Direction as traffic_direction
  FROM nyc_traffic_2016.traffic_2016 WHERE traffic_borough='Manhattan';

CREATE OR REPLACE VIEW collisions_2016_std AS SELECT
   borough as collision_borough,
   ST_Point(latitude, longitude) AS collision_location,
   crash_datetime as collision_datetime
   FROM nyc_traffic_2016.collisions_2016 WHERE collision_borough='Manhattan';

-- fast query to pair each trip with the most recent collision before trip started
CREATE OR REPLACE VIEW trips_recent_collisions AS SELECT * FROM taxi_data_std t ASOF JOIN collisions_2017_std c ON t.pickup_datetime >= c.collision_datetime;


-- CREATE OR REPLACE VIEW trips_nearby_collisions AS SELECT * FROM taxi_data_std t JOIN collisions_2016 c ON ST_DISTANCE_SPHEROID(t.trip_midpoint, c.collision_location) < 50;

-- CREATE OR REPLACE VIEW trips_recent_collisions AS SELECT * FROM taxi_data_std t JOIN collisions_2016_std c ON t.pickup_datetime - c.collision_datetime < interval '1 minute';



-- Pair each trip record with the last collision that happened before the trip started
-- Filter the pairs down to trips where the most recent collision happened in a 50-meter radius of the pickup point or dropoff point AND
-- SELECT * FROM trips_recent_closures WHERE (ST_DISTANCE_SPHEROID(pickup_location, collision_location) < 10);


