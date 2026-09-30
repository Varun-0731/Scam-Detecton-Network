import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.utils import subgraph
except ImportError as e:
    print(f"Error importing torch_geometric: {e}")
    print("Please install it using: pip install torch_geometric")
    Data, NeighborLoader, subgraph = None, None, None

def build_pyg_graph(account_features: pd.DataFrame, edge_features: pd.DataFrame, feature_cols: List[str]) -> 'Data':
    """
    Constructs a PyTorch Geometric Data object from tabular account and edge features.
    """
    if Data is None:
        raise ImportError("torch_geometric is required to build the PyG graph.")
        
    print("Building PyG graph...")
    
    # 1. Create node ID mapping
    unique_accounts = account_features['account_id'].values
    num_nodes = len(unique_accounts)
    account_to_idx = {acc: idx for idx, acc in enumerate(unique_accounts)}
    
    # 2. Build edge_index
    # Only keep edges where both source and target exist in account_features
    # (Should be all of them based on feature_engineering logic, but good practice)
    edges_filtered = edge_features[
        edge_features['source'].isin(account_to_idx) & 
        edge_features['target'].isin(account_to_idx)
    ]
    
    source_idx = edges_filtered['source'].map(account_to_idx).values
    target_idx = edges_filtered['target'].map(account_to_idx).values
    
    edge_index = torch.tensor(np.vstack([source_idx, target_idx]), dtype=torch.long)
    
    # 3. Build node features tensor (x)
    x = torch.tensor(account_features[feature_cols].values, dtype=torch.float)
    
    # 4. Build edge attributes
    edge_attr_cols = ['txn_count', 'total_amount', 'avg_amount', 'max_amount', 'type_diversity']
    # Scale edge features manually or assume raw is fine. Let's scale for stability
    from sklearn.preprocessing import StandardScaler
    edge_attr_scaled = StandardScaler().fit_transform(edges_filtered[edge_attr_cols])
    edge_attr = torch.tensor(edge_attr_scaled, dtype=torch.float)
    
    # 5. Build labels (y)
    y = torch.tensor(account_features['is_fraud'].values, dtype=torch.long)
    
    # 6. Train/Val/Test split masks (70/15/15)
    indices = np.arange(num_nodes)
    labels = account_features['is_fraud'].values
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=42)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Assemble Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Map back original IDs for reference (optional but helpful)
    # Storing non-tensor objects in Data might complain in older PyG, but typically works in newer versions.
    # To be safe, we will just print summary.
    
    print("\n=== Graph Summary ===")
    print(f"Num nodes: {data.num_nodes}")
    print(f"Num edges: {data.num_edges}")
    print(f"Num node features: {data.num_node_features}")
    print(f"Num edge features: {data.num_edge_features}")
    print(f"Fraud nodes: {int(data.y.sum())} / {data.num_nodes}")
    print(f"Train nodes: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    
    return data

def create_neighbor_loader(data: 'Data', batch_size: int = 1024, num_neighbors: List[int] = [15, 10]) -> 'NeighborLoader':
    """
    Create a PyG NeighborLoader for batching.
    """
    if NeighborLoader is None:
        raise ImportError("torch_geometric is required.")
        
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True
    )
    return loader

def get_subgraph(data: 'Data', node_indices: List[int]) -> 'Data':
    """
    Extract a subgraph containing only specified nodes.
    """
    if subgraph is None:
        raise ImportError("torch_geometric is required.")
        
    subset = torch.tensor(node_indices, dtype=torch.long)
    sub_edge_index, sub_edge_attr = subgraph(
        subset, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes
    )
    
    sub_data = Data(
        x=data.x[subset],
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        y=data.y[subset]
    )
    return sub_data

def save_graph(data: 'Data', filepath: str):
    """Save PyG Data object."""
    torch.save(data, filepath)
    print(f"Graph saved to {filepath}")

def load_graph(filepath: str) -> 'Data':
    """Load PyG Data object."""
    data = torch.load(filepath, weights_only=False)
    print(f"Graph loaded from {filepath}")
    return data

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    account_features_path = os.path.join(project_root, 'data', 'account_features.pkl')
    edge_features_path = os.path.join(project_root, 'data', 'edge_features.pkl')
    graph_out_path = os.path.join(project_root, 'data', 'graph_data.pt')
    
    try:
        print(f"Loading features from {project_root}/data/ ...")
        accounts = pd.read_pickle(account_features_path)
        edges = pd.read_pickle(edge_features_path)
        
        feature_cols = [c for c in accounts.columns if c not in ['account_id', 'is_fraud']]
        
        data = build_pyg_graph(accounts, edges, feature_cols)
        
        save_graph(data, graph_out_path)
        
    except Exception as e:
        print(f"Graph construction failed: {e}")
