import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import duckdb
from scipy.spatial import cKDTree
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from dataloader import load_temporal_data
from normalize import get_normalized_adjacency
from gcn import CollisionRiskGCNLSTM 

DB_PATH = "nyc_traffic_2016.duckdb"
HISTORY_STEPS = 4
BATCH_SIZE = 64
LEARNING_RATE = 0.005 
EPOCHS = 30

def build_adjacency_matrix(segments, db_path):
    print("Building Adjacency Matrix...")
    conn = duckdb.connect(db_path)
    df_coords = conn.execute("""
        SELECT SegmentID, ANY_VALUE(X_coord) as X, ANY_VALUE(Y_coord) as Y
        FROM gnn_master_features
        GROUP BY SegmentID
    """).fetchdf()
    conn.close()
    df_coords = df_coords.set_index('SegmentID').reindex(segments).fillna(0)
    coords = df_coords[['X', 'Y']].values
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=6) 
    N = len(segments)
    A = np.zeros((N, N))
    for i in range(N):
        for j_neighbor in idx[i]:
            if i == j_neighbor: continue 
            A[i, j_neighbor] = 1
            A[j_neighbor, i] = 1 
    return A

if __name__ == "__main__":
    # 1. Load Data
    X, Y, segments, timestamps = load_temporal_data(DB_PATH, HISTORY_STEPS)
    
    # 2. Build Graph (Adjacency Matrix)
    A = build_adjacency_matrix(segments, DB_PATH)
    A_norm = get_normalized_adjacency(A) 
    
    # Convert timestamps to datetime objects
    times = pd.to_datetime(timestamps)
    
    # Create the Masks based on Day of Month - straitified split - doesn't work very well
    # Train on days 1-23
    # train_mask = times.day <= 23
    # Test on days 24-31
    # test_mask = times.day > 23
    # Apply Split
    # X_train, Y_train = X[train_mask], Y[train_mask]
    # X_test, Y_test = X[test_mask], Y[test_mask]

    X_train, Y_train = X, Y
    X_test, Y_test = X, Y
    
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples:  {len(X_test)}")
    
    # Continue with Class Weights and Loaders...
    num_pos = torch.sum(Y_train).item()
    num_neg = Y_train.numel() - num_pos
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)])
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # CRITICAL: Do NOT shuffle the test loader. We want to see the timeline unfold.
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model
    model = CollisionRiskGCNLSTM(
        num_nodes=len(segments),
        in_features=3,
        hidden_features=32,
        out_features=1,
        history_steps=HISTORY_STEPS
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Training Loop
    print("\n--- Starting Training ---")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X, A_norm)
            loss = criterion(logits, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Avg Loss: {total_loss / len(train_loader):.6f}")

    # 5. TIME-SEGMENTED ANALYSIS
    print("\n--- Generating Time-Segmented Analysis ---")
    model.eval()
    
    # Create a separate loader specifically for inference (No shuffle)
    inference_loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)
    
    all_probs_list = []
    
    with torch.no_grad():
        print(f"Processing {len(X)} time steps in batches...")
        for batch in inference_loader:
            batch_X = batch[0]
            # Run model on small batch
            batch_logits = model(batch_X, A_norm)
            batch_probs = torch.sigmoid(batch_logits)
            all_probs_list.append(batch_probs)
            
    # Combine all small batches back into one big array
    all_probs = torch.cat(all_probs_list, dim=0).numpy()
    
    # Extract Congestion
    all_volumes = X[:, -1, :, 0].numpy()
    
    # Extract Hours
    hours = pd.to_datetime(timestamps).hour

    # Define Periods
    time_periods = {
        "All_Day": (0, 24),
        "Morning_Rush": (7, 10),  
        "Evening_Rush": (16, 19), 
        "Late_Night": (22, 5)     
    }

    final_results = []

    for period_name, (start, end) in time_periods.items():
        print(f"Analyzing Period: {period_name}...")
        
        if start < end:
            mask = (hours >= start) & (hours < end)
        else: 
            mask = (hours >= start) | (hours < end)
            
        if not np.any(mask):
            print(f"Warning: No data found for {period_name}")
            continue

        period_probs = all_probs[mask]
        period_vols = all_volumes[mask]
        
        avg_risk = np.mean(period_probs, axis=0)
        avg_vol = np.mean(period_vols, axis=0)
        
        risk_thresh = np.quantile(avg_risk, 0.95) 
        vol_thresh = np.quantile(avg_vol, 0.80)   
        
        df_period = pd.DataFrame({
            'SegmentID': segments,
            'Time_Period': period_name,
            'Avg_Risk_Score': avg_risk,
            'Avg_Volume': avg_vol
        })
        
        def categorize(row):
            is_risky = row['Avg_Risk_Score'] >= risk_thresh
            is_congested = row['Avg_Volume'] >= vol_thresh
            
            if is_congested and is_risky: return "Critical Bottleneck"
            if is_congested and not is_risky: return "Safe Congestion"
            if not is_congested and is_risky: return "Hidden Danger"
            return "Safe Flow"

        df_period['Category'] = df_period.apply(categorize, axis=1)
        final_results.append(df_period)

    # Combine and Save
    full_df = pd.concat(final_results, ignore_index=True)
    
    output_filename = "nyc_road_safety_analysis.csv"
    full_df.to_csv(output_filename, index=False)
    
    print(f"\n Analysis Complete. Results saved to: {output_filename}")
    print("\n--- Summary of Categories (All Day) ---")
    print(full_df[full_df['Time_Period'] == 'All_Day']['Category'].value_counts())
    
    print("\n--- Sample: Top 5 Hidden Dangers (Morning Rush) ---")
    print(full_df[(full_df['Time_Period'] == 'Morning_Rush') & (full_df['Category'] == 'Hidden Danger')]
          .sort_values('Avg_Risk_Score', ascending=False)
          .head(5)[['SegmentID', 'Avg_Risk_Score', 'Avg_Volume']])
    
    # Evaluation (Metrics & Plots)
    print("\n--- Running Performance Metrics ---")

    model.eval()
    y_true = []
    y_scores = []

    # 1. Collect Predictions
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            logits = model(batch_X, A_norm)
            probs = torch.sigmoid(logits)
            y_true.extend(batch_Y.cpu().numpy().flatten())
            y_scores.extend(probs.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 2. Calculate Metrics
    y_pred = (y_scores > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Safe', 'Collision']))

    # Calculate AUC-ROC
    roc_auc = roc_auc_score(y_true, y_scores)
    print(f"AUC-ROC Score: {roc_auc:.4f} (1.0 is perfect, 0.5 is random)")

    # Calculate PR-AUC (Area Under Precision-Recall Curve)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC Score: {pr_auc:.4f}")

    # Visualization

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Safe', 'Pred Crash'], yticklabels=['Actual Safe', 'Actual Crash'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    # Plot 3: Precision-Recall Curve - best metric
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
