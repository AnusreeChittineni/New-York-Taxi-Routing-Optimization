-- Fuzzy spatial and temporal join of three datasets - taxi trips, collisons, traffic counts

INSTALL spatial;
LOAD spatial;

SELECT t1.pickup_location, t1.dropoff_location,  t2.location FROM taxi_data_std as t1  join collisions_2016_std as t2 ON ST_Distance(t1.pickup_location, t2.location) < 50;
