#!/usr/bin/env python3
# train_imputation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset_stgnn import get_dataloaders
try:
    from torch_geometric.nn import SAGEConv, ChebConv
except ImportError:
    print("Warning: PyTorch Geometric not installed. The network will mock layers.")
    class SAGEConv(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.lin = nn.Linear(in_chan, out_chan)
        def forward(self, x, edge_index):
            return self.lin(x)

class TemporalConvNet(nn.Module):
    """ simple 1D Temporal Convolution """
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(TemporalConvNet, self).__init__()
        self.conv = nn.Conv1d(num_inputs, num_channels, kernel_size)
    def forward(self, x):
        # x: (Batch*Nodes, Features, Timesteps)
        out = self.conv(x)
        return F.relu(out)

class STGCN_Block(nn.Module):
    """
    Spatio-Temporal Graph Convolution Block.
    Contains one GNN layer for spatial and one Temporal layer.
    """
    def __init__(self, in_channels, spatial_channels, out_channels, seq_len):
        super(STGCN_Block, self).__init__()
        self.sage = SAGEConv(in_channels, spatial_channels)
        self.tcn = TemporalConvNet(spatial_channels, out_channels)
        
    def forward(self, x, edge_index):
        # x shape: (B, N, seq_len, F)
        B, N, T, F_in = x.shape
        
        # 1. Spatial Convolution (Apply GNN on each timestep independently)
        spatial_out = []
        for t in range(T):
            x_t = x[:, :, t, :] # (B, N, F_in)
            # Batch shape for sparse GNNs requires flattening B and N or passing individually
            x_t_flat = x_t.contiguous().view(B * N, F_in)
            
            # Adjust edge_index for batched graph
            batch_edge_index = edge_index.clone()
            
            out = self.sage(x_t_flat, batch_edge_index)
            spatial_out.append(out.view(B, N, -1))
            
        x_spa = torch.stack(spatial_out, dim=2) # (B, N, T, spatial_channels)
        
        # 2. Temporal Convolution
        S_chan = x_spa.shape[-1]
        x_spa_flat = x_spa.view(B * N, T, S_chan).permute(0, 2, 1) # (B*N, S_chan, T)
        
        t_out = self.tcn(x_spa_flat) # (B*N, out_channels, T_new)
        t_out = t_out.permute(0, 2, 1).contiguous().view(B, N, t_out.shape[-1], -1)
        
        return t_out

class ArterialImputationNetwork(nn.Module):
    def __init__(self, num_features, seq_len, target_len):
        super(ArterialImputationNetwork, self).__init__()
        
        self.block1 = STGCN_Block(in_channels=num_features, spatial_channels=64, out_channels=32, seq_len=seq_len)
        # Linear layer mapping back to imputation features over target length
        self.regressor = nn.Linear(32, target_len * num_features)
        self.target_len = target_len
        self.num_features = num_features
        
    def forward(self, x, edge_index):
        # x: (B, N, seq_len, F)
        out = self.block1(x, edge_index)
        # Pool temporal dimension
        out = out.mean(dim=2) # (B, N, 32)
        
        preds = self.regressor(out) # (B, N, target_len * F)
        return preds.view(-1, out.shape[1], self.target_len, self.num_features)

def train_imputation_model():
    """ Masked Training Loop for Imputation """
    base_dir = '/data2/NPTOI/arterial'
    train_loader, val_loader = get_dataloaders(base_dir, batch_size=8) # Test with 8
    
    # Init Model
    # Since dataset_stgnn mocked features as 40
    model = ArterialImputationNetwork(num_features=40, seq_len=12, target_len=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss() # MAE
    
    print("Starting Training Loop...")
    for epoch in range(1): # Test with 1 epoch
        model.train()
        total_loss = 0
        
        for batch_idx, (x_masked, y_true, mask_bool, edge_index) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Squeeze edge_index which gets batched by data loader
            e_idx = edge_index[0] 
            
            # Forward
            preds = model(x_masked, e_idx) # (B, N, target_len, F)
            
            # Masked Loss: Only calculate error on nodes that were dropped!
            # mask_bool: (B, N)
            
            p = preds[mask_bool] # (Masked_nodes, target_len, F)
            y = y_true[mask_bool] # (Masked_nodes, target_len, F)
            
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Masked MAE Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch} Validation -> Avg Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train_imputation_model()
