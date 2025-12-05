import torch
import torch.nn as nn

class CollisionRiskGCNLSTM(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_features, out_features, history_steps):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. Spatial Feature Extraction (GCN)
        self.gcn_linear = nn.Linear(in_features, hidden_features)
        
        # 2. Temporal Sequence Modeling (LSTM)
        self.lstm = nn.LSTM(hidden_features, hidden_features, num_layers=1, batch_first=True)
        
        # 3. Output Layer (Dense)
        self.output_linear = nn.Linear(hidden_features, out_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_seq, A_norm):        
        batch_size, T, N, F_in = X_seq.shape
        H_t_list = []
        
        # Step 1: Spatial Processing
        for t in range(T):
            X_t = X_seq[:, t, :, :]
            
            # GCN Operation (Conceptual): H_t = ReLU( A_norm @ X_t @ W_gcn )
            # 1. Linear Transformation: X_t @ W_gcn
            X_prime = self.gcn_linear(X_t) # (Batch, N, F_hidden)
            
            # 2. Graph Convolution (Spatial Aggregation): A_norm @ X_prime
            H_t = torch.bmm(A_norm.expand(batch_size, N, N), X_prime) # (Batch, N, F_hidden)
            H_t = self.relu(H_t)
            H_t_list.append(H_t)
        
        # Stacking the spatially processed features for the LSTM
        H_seq = torch.stack(H_t_list, dim=1) # (Batch, History Steps, N, F_hidden)
        
        # Reshape for LSTM: Combine N nodes into the batch dimension
        H_seq_reshaped = H_seq.view(batch_size * N, T, -1) # (Batch*N, T, F_hidden)
        
        # Step 2: Temporal Processing (LSTM)
        # output: (Batch*N, T, F_hidden), (hn, cn)
        lstm_out, _ = self.lstm(H_seq_reshaped) 
        
        H_final = lstm_out[:, -1, :] # (Batch*N, F_hidden)
        
        # Step 3: Output Prediction
        logits = self.output_linear(H_final) # (Batch*N, 1)
        logits = logits.view(batch_size, N) 
        
        return logits