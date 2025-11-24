import duckdb

DB_PATH = "nyc_traffic_2016.duckdb"
conn = duckdb.connect(DB_PATH)
conn.execute("""INSTALL SPATIAL;""")
conn.execute("""LOAD SPATIAL;""")
conn.execute("""CREATE OR REPLACE VIEW taxi_data_std AS SELECT         
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
            ST_Point(pickup_longitude, pickup_latitude) AS pickup_location,
            ST_Point(dropoff_longitude, dropoff_latitude) as dropoff_location,
            store_and_fwd_flag,
            trip_distance,
            dropoff_datetime - pickup_datetime AS trip_duration
         FROM nyc_traffic_2016.taxi_data;""")


conn.execute("""CREATE OR REPLACE VIEW traffic_2016_std AS SELECT           
          make_timestamp(
                    CAST(Yr AS INTEGER),
                    CAST(M  AS INTEGER),
                    CAST(D  AS INTEGER),
                    CAST(HH AS INTEGER),
                    CAST(MM AS INTEGER),
                    0  -- seconds assumed 0
                ) AS event_time,
          Boro as boro,
          Vol as vol,
          SegmentID as segment_id,
          ST_GeomFromText(WktGeom) as location,
          street,
          fromSt as from_st,
          toSt as to_st,
          Direction as direction
          FROM nyc_traffic_2016.traffic_2016;""")


conn.execute("""CREATE OR REPLACE VIEW collisions_2016_std AS SELECT           
          borough as borough,
          ST_Point(longitude, latitude) AS location,
          FROM nyc_traffic_2016.collisions_2016;""")
