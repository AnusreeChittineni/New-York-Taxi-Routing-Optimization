import numpy as np
import torch

def get_normalized_adjacency(A):
    """
    Input: A (N x N numpy array) - The adjacency matrix from your previous script
    Output: A_norm (N x N torch tensor) - Normalized for GCN
    """
    # 1. Add Self-Loops (Identity Matrix) - Nodes must influence themselves in the next time step
    N = A.shape[0]
    A_hat = A + np.eye(N)
    
    # 2. Calculate Degree Matrix (D) - Sum of rows gives the degree of each node
    D_hat = np.sum(A_hat, axis=1)
    
    # 3. Inverse Square Root of Degree - Handle division by zero for isolated nodes
    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.power(D_hat, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    
    # 4. Diagonalize
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    
    # 5. Symmetric Normalization: D^(-0.5) * A_hat * D^(-0.5)
    A_norm = D_inv_sqrt_mat @ A_hat @ D_inv_sqrt_mat
    
    return torch.tensor(A_norm, dtype=torch.float32)