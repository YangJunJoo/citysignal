#!/usr/bin/env python3
# build_adjacency_matrix.py

import os
import geopandas as gpd
import pandas as pd
import numpy as np

def build_adjacency_matrix(shapefile_path, output_dir):
    """
    Parses the OS_Primary_Basemap_segments shapefile and constructs a Line Graph
    adjacency list / matrix mapping Seg_Unique -> Adjoining Seg_Unique.
    """
    print(f"Loading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    if 'Seg_Unique' not in gdf.columns or 'start_id' not in gdf.columns or 'end_id' not in gdf.columns:
        raise ValueError("Required columns (Seg_Unique, start_id, end_id) not found in shapefile.")
        
    df = gdf[['Seg_Unique', 'start_id', 'end_id', 'Length_m']].copy()
    
    # We want a Line Graph where nodes are Seg_Unique.
    # An edge exists from Seg A to Seg B iff A.end_id == B.start_id
    
    # Let's map start_ids to Seg_Unique
    # B.start_id -> list of B.Seg_Unique
    start_map = df.groupby('start_id')['Seg_Unique'].apply(list).to_dict()
    
    edges = []
    
    # Create directed edges (upstream segment -> downstream segment)
    for _, row in df.iterrows():
        source_seg = str(row['Seg_Unique'])
        downstream_inter = str(row['end_id'])
        
        # Who starts at downstream_inter? The downstream segments!
        if downstream_inter in start_map:
            for target_seg in start_map[downstream_inter]:
                # We optionally capture the geographical distance logic here if needed
                # For GCN it's 1 if connected, or 1/d for distance
                edges.append((source_seg, str(target_seg), 1.0))
                
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
    
    num_edges = len(edges_df)
    unique_nodes = set(edges_df['source']).union(set(edges_df['target']))
    print(f"Extracted {num_edges} directed edges connecting {len(unique_nodes)} unique Seg_Unique segments.")
    
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'stgnn_edge_index.csv')
    edges_df.to_csv(out_csv, index=False)
    
    # Save a node mapping format to ensure numerical IDs (0 to N-1) for PyTorch Geometric
    node_mapping = pd.DataFrame({'Seg_Unique': list(unique_nodes)})
    node_mapping['node_id'] = range(len(node_mapping))
    node_mapping.to_csv(os.path.join(output_dir, 'stgnn_node_mapping.csv'), index=False)
    
    print(f"Saved Edge Index to {out_csv}")
    print(f"Saved Node ID Mapping to stgnn_node_mapping.csv")

if __name__ == "__main__":
    shape_path = '/data2/NPTOI/arterial/basemap/OS_Primary_Basemap_segments.shp'
    out_dir = '/data2/NPTOI/arterial/imputation_network'
    build_adjacency_matrix(shape_path, out_dir)
