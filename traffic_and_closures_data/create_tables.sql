
-- NODEID is a DCP LION reference ID
CREATE TABLE IF NOT EXISTS closure_2023 AS
    SELECT 
        SEGMENTID,
        ONSTREETNAME,
        FROMSTREETNAME,
        BOROUGH_CODE,
        strptime(WORK_START_DATE, '%m/%d/%Y %-H:%-M:%-S') AS WORK_START_DATE,
        strptime(WORK_END_DATE, '%m/%d/%Y %-H:%-M:%-S') AS WORK_END_DATE,
        FROM read_csv_auto('data/Street_Closures_with_Segment_IDs.csv');


-- NODEID is distinct from SEGMENTID - needs to be reconciled
CREATE TABLE IF NOT EXISTS closure_2025 AS 
    SELECT 
        the_geom,
        CAST(REPLACE(SEGMENTID, ',', '') AS DOUBLE) AS SEGMENTID,
        OFT,
        ONSTREETNAME,
        FROMSTREETNAME,
        TOSTREETNAME,
        BOROUGH_CODE,
        CAST(WORK_START_DATE AS DATETIME) AS WORK_START_DATE,
        CAST(WORK_END_DATE AS DATETIME) AS WORK_END_DATE,
    FROM read_csv_auto('data/Street_Closures_due_to_construction_activities_by_Block.csv');

-- Convert Yr, D, HH, MM to a timestamp 
CREATE TABLE IF NOT EXISTS traffic AS 
    SELECT 
        make_timestamp(
            CAST(Yr AS INTEGER),
            CAST(M  AS INTEGER),
            CAST(D  AS INTEGER),
            CAST(HH AS INTEGER),
            CAST(MM AS INTEGER),
            0  -- seconds assumed 0
        ) AS event_time,
        Boro,
        Vol,
        SegmentID,
        WktGeom,
        street,
        fromSt, 
        toSt, 
        direction
    FROM read_csv_auto('data/Automated_Traffic_Volume_Counts_2025.csv');
