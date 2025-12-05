-- Spatial and temporal join of three datasets - taxi trips, collisons, traffic counts

INSTALL spatial;
LOAD spatial;

CREATE OR REPLACE VIEW trip_geom AS SELECT
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


CREATE OR REPLACE VIEW traffic_geom AS
SELECT
    make_timestamp(
        CAST(Yr AS INTEGER),
        CAST(M  AS INTEGER),
        CAST(D  AS INTEGER),
        CAST(HH AS INTEGER),
        CAST(MM AS INTEGER),
        0  -- seconds assumed 0
    ) AS event_time,
    Boro AS traffic_borough,
    CAST(REPLACE(Vol, ',', '') AS INTEGER) AS traffic_vol,
    SegmentID AS traffic_segment_id,
    ST_Point2D(
        ST_Y(
            ST_Transform(
                ST_GeomFromText(WktGeom),     -- parse WKT -> GEOMETRY
                'EPSG:2263',                  -- source CRS (projected)
                'EPSG:4326',                  -- target CRS (WGS84)
                TRUE                          -- always_xy: treat as lon/lat
            )
        ),                                   -- latitude
        ST_X(
            ST_Transform(
                ST_GeomFromText(WktGeom),
                'EPSG:2263',
                'EPSG:4326',
                TRUE
            )
        )                                    -- longitude
    ) AS traffic_location,                   -- POINT_2D
    street AS traffic_street,
    Direction AS traffic_direction
FROM nyc_traffic_2016.traffic_2016;

CREATE OR REPLACE VIEW collision_geom AS SELECT
   borough as collision_borough,
   ST_Point(latitude, longitude) AS collision_location,
   crash_datetime as collision_datetime
   FROM nyc_traffic_2016.collisions_2016;

-- select trips in the second week of january
CREATE OR REPLACE VIEW trip_january AS SELECT * FROM trip_geom  WHERE month(pickup_datetime)=1 AND day(pickup_datetime)=2;

CREATE OR REPLACE VIEW collision_january AS SELECT * FROM collision_geom WHERE month(collision_datetime)= 1 AND day(collision_datetime)=2;

CREATE TABLE IF NOT EXISTS train_trips AS SELECT * FROM 'input/train.csv';

CREATE OR REPLACE VIEW train_trip_geom AS SELECT
     id,
     vendor_id
     passenger_count,
     pickup_datetime,
     dropoff_datetime,
     pickup_longitude,
     pickup_latitude,
     dropoff_longitude,
     dropoff_latitude,
     ST_Point(pickup_latitude, pickup_longitude) AS pickup_location,
     ST_Point(dropoff_latitude, dropoff_longitude) as dropoff_location,
     ST_Point((ST_X(pickup_location) + ST_X(dropoff_location)) / 2, (ST_Y(pickup_location) + ST_Y(dropoff_location)) / 2) as trip_midpoint,
     store_and_fwd_flag,
     trip_duration
  FROM nyc_traffic_2016.train_trips; 
  
  

CREATE TABLE IF NOT EXISTS test_trips AS SELECT * FROM 'input/test.csv';


CREATE OR REPLACE VIEW test_trip_geom AS SELECT
     id,
     vendor_id
     passenger_count,
     pickup_datetime,
     pickup_longitude,
     pickup_latitude,
     dropoff_longitude,
     dropoff_latitude,
     ST_Point(pickup_latitude, pickup_longitude) AS pickup_location,
     ST_Point(dropoff_latitude, dropoff_longitude) as dropoff_location,
     ST_Point((ST_X(pickup_location) + ST_X(dropoff_location)) / 2, (ST_Y(pickup_location) + ST_Y(dropoff_location)) / 2) as trip_midpoint,
     store_and_fwd_flag,
  FROM nyc_traffic_2016.test_trips;


CREATE OR REPLACE VIEW train_trips_collisions
AS
SELECT * FROM train_trip_geom t 
    LEFT JOIN collision_geom c ON
    month(t.pickup_datetime) = month(c.collision_datetime) AND 
    day(t.pickup_datetime) = day(c.collision_datetime);
    
CREATE OR REPLACE VIEW train_trips_last_hour_collisions
AS SELECT * 
FROM trips_collisions 
WHERE pickup_datetime - collision_datetime < interval '1 hour';

CREATE OR REPLACE VIEW train_trips_nearby_last_hour_collisions
AS SELECT *
FROM trips_last_hour_collisions
WHERE st_distance_spheroid(trip_midpoint, collision_location) <  st_distance_spheroid(pickup_location, dropoff_location) / 2;


CREATE OR REPLACE VIEW train_trips_traffic
AS
SELECT * FROM train_trip_geom t 
    LEFT JOIN traffic_geom r ON
    month(t.pickup_datetime) = month(r.event_time) AND 
    day(t.pickup_datetime) = day(r.event_time);
    
    
CREATE OR REPLACE VIEW train_trips_last_hour_traffic
AS SELECT * 
FROM trips_traffic
WHERE pickup_datetime - event_time < interval '1 hour';


CREATE OR REPLACE VIEW train_trips_nearby_last_hour_traffic
AS SELECT *
FROM trips_last_hour_traffic
WHERE st_distance_spheroid(trip_midpoint, traffic_location) <  st_distance_spheroid(pickup_location, dropoff_location) * (3 /2);



-- merge datasets for test set
 

CREATE OR REPLACE VIEW test_trips_collisions
AS
SELECT * FROM test_trip_geom t 
    LEFT JOIN collision_geom c ON
    month(t.pickup_datetime) = month(c.collision_datetime) AND 
    day(t.pickup_datetime) = day(c.collision_datetime);
    
CREATE OR REPLACE VIEW test_trips_last_hour_collisions
AS SELECT * 
FROM trips_collisions 
WHERE pickup_datetime - collision_datetime < interval '1 hour';

CREATE OR REPLACE VIEW test_trips_nearby_last_hour_collisions
AS SELECT *
FROM trips_last_hour_collisions
WHERE st_distance_spheroid(trip_midpoint, collision_location) <  st_distance_spheroid(pickup_location, dropoff_location) / 2;


CREATE OR REPLACE VIEW test_trips_traffic
AS
SELECT * FROM test_trip_geom t 
    LEFT JOIN traffic_geom r ON
    month(t.pickup_datetime) = month(r.event_time) AND 
    day(t.pickup_datetime) = day(r.event_time);
    
    
CREATE OR REPLACE VIEW test_trips_last_hour_traffic
AS SELECT * 
FROM trips_traffic
WHERE pickup_datetime - event_time < interval '1 hour';


CREATE OR REPLACE VIEW test_trips_nearby_last_hour_traffic
AS SELECT *
FROM trips_last_hour_traffic
WHERE st_distance_spheroid(trip_midpoint, traffic_location) <  st_distance_spheroid(pickup_location, dropoff_location) * (3 /2);

-- Save results to output CSV


copy (select id, avg(traffic_vol) as last_hour_avg_traffic_vol FROM train_trips_nearby_last_hour_traffic GROUP BY id) TO 'input/traffic_train.csv' (HEADER, DELIMITER ',');

copy (select id, count(collision_location) as last_hour_collisions FROM train_trips_nearby_last_hour_collisions GROUP BY id) TO 'input/collisions_train.csv' (HEADER, DELIMITER ',');


copy (select id, avg(traffic_vol) as last_hour_avg_traffic_vol FROM test_trips_nearby_last_hour_traffic GROUP BY id) TO 'input/traffic_test.csv' (HEADER, DELIMITER ',');

copy (select id, count(collision_location) as last_hour_collisions FROM test_trips_nearby_last_hour_collisions GROUP BY id) TO 'input/collisions_test.csv' (HEADER, DELIMITER ',');

