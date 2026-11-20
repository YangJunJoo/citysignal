#!/usr/bin/env python3
# dataset_stgnn.py

import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ArterialGraphDataset(Dataset):
    """
    Spatio-Temporal Graph Dataset for CV-ATSPM Imputation.
    Loads matching parquet files, applies a sliding temporal window of size H (history),
    and sets up targets for size F (future/imputation target).
    """
    def __init__(self, data_dir, node_map_path, edge_index_path, 
                 seq_len=12, target_len=1, mask_ratio=0.15, mode='train'):
        
        self.seq_len = seq_len
        self.target_len = target_len
        self.mask_ratio = mask_ratio
        
        # Load topology
        self.node_mapping = pd.read_csv(node_map_path)
        self.num_nodes = len(self.node_mapping)
        self.seg_to_id = dict(zip(self.node_mapping['Seg_Unique'].astype(str), self.node_mapping['node_id']))
        
        edges = pd.read_csv(edge_index_path)
        # Filter edges to only known nodes
        edges = edges[edges['source'].astype(str).isin(self.seg_to_id) & edges['target'].astype(str).isin(self.seg_to_id)]
        
        source_idx = edges['source'].astype(str).map(self.seg_to_id).values
        target_idx = edges['target'].astype(str).map(self.seg_to_id).values
        self.edge_index = torch.tensor([source_idx, target_idx], dtype=torch.long)
        
        # Load and align temporal features
        print(f"Loading Parquet files from {data_dir}...")
        self.X, self.feature_names = self._load_and_align_data(data_dir)
        
        # X shape: (num_nodes, num_time_steps, num_features)
        self.num_time_steps, self.num_features = self.X.shape[1], self.X.shape[2]
        
        # Generate sliding windows
        self.valid_indices = self._generate_windows()
        
        # Split Data (Train/Val/Test)
        train_thresh = int(len(self.valid_indices) * 0.7)
        val_thresh = int(len(self.valid_indices) * 0.85)
        
        if mode == 'train':
            self.indices = self.valid_indices[:train_thresh]
        elif mode == 'val':
            self.indices = self.valid_indices[train_thresh:val_thresh]
        else:
            self.indices = self.valid_indices[val_thresh:]
            
        print(f"Initialized {mode} dataset with {len(self.indices)} samples.")
        print(f"Graph Nodes: {self.num_nodes}, Features: {self.num_features}, Edges: {self.edge_index.shape[1]}")

    def _load_and_align_data(self, data_dir):
        # In a real implementation this should load all dates chronologically
        # and pivot into a vast 3D tensor (Nodes x Time x Features).
        # For memory safety in this skeleton, we will mock a subset tensor structure
        # based on existing features (since full parquet alignment takes time).
        
        print("Mocking data alignment across time steps for skeleton...")
        # Assume 14 days of 15-min bins = 1344 time steps
        mock_timesteps = 1344
        mock_features = 40 # Using top 40 metrics continuous
        
        X = np.random.randn(self.num_nodes, mock_timesteps, mock_features).astype(np.float32)
        feature_names = [f"feat_{i}" for i in range(mock_features)]
        return torch.tensor(X), feature_names

    def _generate_windows(self):
        total_len = self.num_time_steps
        window_size = self.seq_len + self.target_len
        indices = []
        for i in range(total_len - window_size + 1):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.seq_len
        target_end = end + self.target_len
        
        x = self.X[:, start:end, :] # (N, seq_len, F)
        y = self.X[:, end:target_end, :] # (N, target_len, F)
        
        # Generate Artificial Spatial Mask (Simulating missing ATSPM nodes)
        # We select random nodes to "drop" and require model to impute
        num_mask_nodes = int(self.num_nodes * self.mask_ratio)
        mask_nodes = torch.randperm(self.num_nodes)[:num_mask_nodes]
        
        masked_x = x.clone()
        masked_x[mask_nodes, :, :] = 0.0 # Set missing to zero
        
        # Mask boolean tensor to know which nodes are imputed vs true
        mask_bool = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask_bool[mask_nodes] = True
        
        return masked_x, y, mask_bool, self.edge_index

def get_dataloaders(base_dir, batch_size=32):
    edge_path = os.path.join(base_dir, 'imputation_network', 'stgnn_edge_index.csv')
    node_path = os.path.join(base_dir, 'imputation_network', 'stgnn_node_mapping.csv')
    data_path = os.path.join(base_dir, 'matched_metrics')
    
    train_dataset = ArterialGraphDataset(data_path, node_path, edge_path, mode='train')
    val_dataset = ArterialGraphDataset(data_path, node_path, edge_path, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
