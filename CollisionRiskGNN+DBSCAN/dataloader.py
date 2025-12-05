import duckdb
import numpy as np
import torch
import pandas as pd

def load_temporal_data(db_path, history_steps=4):
    conn = duckdb.connect(db_path)
    print("Fetching and aggregating data from DuckDB...")
    
    # 1. SQL Query: Joins Features + Collisions + Clusters
    query = """
    SELECT
        -- A. Time Handling (Fixing the 24:00 rollover issue)
        CAST(
            make_timestamp(
                CAST(T1.Yr AS BIGINT), CAST(T1.M AS BIGINT), CAST(T1.D AS BIGINT), 
                0, 0, 0.0
            ) + (CAST(T1.TimeBin_Min AS BIGINT) * INTERVAL 1 MINUTE)
        AS VARCHAR) as global_time,
        
        T1.SegmentID,
        
        -- B. Dynamic Features
        AVG(T1.Avg_Volume) as Avg_Volume,
        MAX(T1.Severe_Collision_Count) as Severe_Collision_Count,
        
        -- C. Static Feature (DBSCAN Cluster)
        -- Use COALESCE(..., 0) because noise points won't be in the cluster table
        MAX(COALESCE(T2.Is_In_Cluster, 0)) as Is_In_Cluster, 
        
        -- D. Target Variable
        MAX(T1.Collision_Count) as Collision_Count
        
    FROM gnn_master_features T1
    -- LEFT JOIN ensures we keep all segments, even if not in a cluster
    LEFT JOIN feature_congestion_clusters T2 ON T1.SegmentID = T2.SegmentID
    
    GROUP BY 1, 2
    ORDER BY global_time, T1.SegmentID;
    """
    
    try:
        df = conn.execute(query).fetchdf()
    except Exception as e:
        print("\nCRITICAL SQL ERROR:")
        print("The database could not find the table 'feature_congestion_clusters'.")
        print("Please run 'python dbscan.py' FIRST to generate this table.")
        raise e
    
    # 2. Pivot Function (Long Format -> Wide Format)
    def safe_pivot(df, value_col):
        # Index=Time, Columns=Segments
        pivoted = df.pivot(index='global_time', columns='SegmentID', values=value_col)
        pivoted = pivoted.fillna(0).sort_index()
        return pivoted

    print("Pivoting data matrices...")
    df_vol = safe_pivot(df, 'Avg_Volume')
    df_hist = safe_pivot(df, 'Severe_Collision_Count')
    df_clust = safe_pivot(df, 'Is_In_Cluster')
    df_target = safe_pivot(df, 'Collision_Count')
    
    # Extract Metadata
    segments = df_vol.columns.values
    times = pd.to_datetime(df_vol.index.values)
    
    # 3. Stack Features into 3D Array
    # Shape: (Total_Time_Steps, Num_Nodes, 3_Features)
    feature_vol = df_vol.values
    feature_hist = df_hist.values
    feature_clust = df_clust.values
    
    X_full = np.stack([feature_vol, feature_hist, feature_clust], axis=-1)
    Y_full = (df_target.values > 0).astype(float)
    
    # 4. Create Sliding Windows (History -> Future)
    X_seq, Y_seq = [], []
    prediction_times = []
    T_total = len(times)
    
    if T_total <= history_steps:
         raise ValueError(f"Not enough time steps ({T_total}) for history {history_steps}")

    print("Creating sliding windows...")
    for t in range(T_total - history_steps):
        # Input: t to t+history (Sequence)
        X_seq.append(X_full[t : t + history_steps]) 
        # Target: t+history (The next immediate step)
        Y_seq.append(Y_full[t + history_steps])
        # Time: The timestamp of the TARGET
        prediction_times.append(times[t + history_steps])
        
    # Convert to Tensors
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32)
    
    conn.close()
    
    # Returns: Inputs, Targets, Legend(IDs), Clock(Times)
    return X_tensor, Y_tensor, segments, np.array(prediction_times)
