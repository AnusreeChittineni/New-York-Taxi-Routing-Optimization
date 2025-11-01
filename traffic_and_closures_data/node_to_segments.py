import requests
import time 
import duckdb
from functools import lru_cache

LION_URL = "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/ArcGIS/rest/services/LION/FeatureServer/0/query"

@lru_cache(maxsize=100000)
def get_segments_from_node_street(node_id: str, street_name: str) -> list[str]:
    """
    Return list of SegmentID(s) where Street == street_name AND
    (NodeIDFrom == node_id OR NodeIDTo == node_id).

    Note: this method returns multiple segment IDs when node_id identifies 
    an intersection.
    """

    # Defensive escaping for single quotes in street names
    node_id = node_id.rjust(7, '0')
    street_q = street_name.replace("'", "''")
    where = (
        f"Street = '{street_q}' AND "
        f"(NodeIDFrom = {node_id} OR NodeIDTo = {node_id}) AND SegmentID IS NOT NULL"
    )

    params = {
        "f": "json",
        "where": where,
        "outFields": "SegmentID",
        "returnGeometry": "false",
    }

    for attempt in range(3):
        try:
            r = requests.get(LION_URL, params=params, timeout=30)
            r.raise_for_status()
            response_json = r.json()
            if "error" in response_json:
                
                raise RuntimeError(response_json["error"])
            feats = response_json.get("features", [])
            segment_ids_str = map(lambda x: (x.get("attributes") or {}).get("SegmentID"), feats)
            segment_ids = list(set(map(lambda x: str(int(x)), segment_ids_str)))
            if len(segment_ids) == 0:
                return None
            print(f'API returned segment IDs {segment_ids} for node id {node_id}')
            return list(set(segment_ids))
        except Exception:
            if attempt == 2:
                return []
            time.sleep(0.5 * (attempt + 1))



def process_one(path_in: str, node_col: str, street_col: str, path_out: str):
    # Read with DuckDB (preserve types; we’ll treat both columns as text for safety)
    
    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE VIEW input_tbl AS
        SELECT * FROM read_csv_auto('{path_in}', header=true);
    """)

    # Pull distinct (node, street) pairs
    pairs = con.execute(f"""
        SELECT DISTINCT
            CAST({node_col} AS VARCHAR) AS node_raw,
            CAST({street_col} AS VARCHAR) AS street_raw
        FROM input_tbl
        WHERE {node_col} IS NOT NULL AND {street_col} IS NOT NULL
    """).df()


    # Rebuild with SQL-side cleaners (DuckDB UDFs via Python registration)
    def _clean_str_py(x) -> str:
        if x is None:
            return None
        x = str(x).strip()
        return x if x else None
    
    # Normalize
    pairs["node_norm"] = pairs["node_raw"].apply(_clean_str_py)
    pairs["street_norm"] = pairs["street_raw"].apply(_clean_str_py)
    pairs = pairs.dropna(subset=["node_norm", "street_norm"]).drop_duplicates()

    # Query LION once per unique pair
    def lookup(row):
        print(f"Row: {row.name} ", end='')
        return get_segments_from_node_street(row["node_norm"], row["street_norm"])

    pairs["SEGMENTID_LIST"] = pairs.apply(lookup, axis=1)

    # 2) explode to one row per segment id
    exploded = pairs.explode("SEGMENTID_LIST").rename(
        columns={"SEGMENTID_LIST": "SEGMENTID"}
    )

    # 3) drop empty + dedupe the triple (node, street, segment)
    exploded = (
        exploded.dropna(subset=["SEGMENTID"])
                .drop_duplicates(subset=["node_norm", "street_norm", "SEGMENTID"])
                .loc[:, ["node_norm", "street_norm", "SEGMENTID"]]
                .rename(columns={"node_norm": "__node__", "street_norm": "__street__"})
    )

    # 4) register mapping and join — this will create multiple output rows
    #    when a single (node, street) maps to multiple SegmentIDs.
    con.register("pair_map_df", exploded)

    con.create_function("_clean_str", _clean_str_py)

    con.execute(f"""
        CREATE OR REPLACE TABLE output_tbl AS
        WITH base AS (
            SELECT *,
                   CAST({node_col} AS VARCHAR)  AS __node__,
                   CAST({street_col} AS VARCHAR) AS __street__
            FROM input_tbl
        )
        SELECT
            b.*,
            m.SEGMENTID
        FROM base b
        LEFT JOIN pair_map_df m
          ON _clean_str(b.__node__) = m.__node__
         AND _clean_str(b.__street__) = m.__street__;
    """)

    # Write to CSV
    con.execute(f"COPY output_tbl TO '{path_out}' (HEADER, DELIMITER ',');")
    print(f"Wrote: {path_out}")
    con.close()


if __name__ == '__main__':
    process_one('data/Street_Closures_due_to_construction_activities_by_Intersection.csv', 'NODEID', 'ONSTREETNAME', 'data/Street_Closures_due_to_construction_activities_by_Intersection_Segment_ids.csv')