import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from collections import deque
import networkx.algorithms.community as nx_comm
from typing import Dict, List, Any

def build_networkx_graph(edge_features: pd.DataFrame, account_features: pd.DataFrame) -> nx.DiGraph:
    """Create a directed NetworkX graph from edge features."""
    G = nx.DiGraph()
    
    # Build a lookup dict keyed by account_id
    id_col = 'account_id' if 'account_id' in account_features.columns else account_features.index.name or 'index'
    
    # Add nodes with attributes
    for _, row in account_features.iterrows():
        node_id = row.get('account_id', row.name)
        node_attr = row.to_dict()
        node_attr.pop('account_id', None)  # Don't store redundant id in attrs
        G.add_node(node_id, **node_attr)
        
    # Add edges with attributes
    for _, row in edge_features.iterrows():
        edge_attr = row.to_dict()
        src = edge_attr.pop('source', None) or edge_attr.pop('nameOrig', None)
        dst = edge_attr.pop('target', None) or edge_attr.pop('nameDest', None)
        
        if src and dst:
            G.add_edge(src, dst, **edge_attr)
            
    return G

def detect_communities(G: nx.DiGraph, method: str = 'louvain') -> dict:
    """Convert to undirected for community detection and apply Louvain."""
    G_undirected = G.to_undirected()
    
    if method == 'louvain':
        communities = nx_comm.louvain_communities(G_undirected)
    else:
        # Fallback or other methods
        communities = nx_comm.louvain_communities(G_undirected)
        
    community_dict = {}
    for i, comm in enumerate(communities):
        community_dict[i] = list(comm)
        
    return community_dict

def score_communities(communities: dict, account_features: pd.DataFrame, risk_scores: dict = None) -> pd.DataFrame:
    """Score each community based on fraud rate and risk scores."""
    # Build a lookup set and dict for fast access
    if 'account_id' in account_features.columns:
        acct_set = set(account_features['account_id'].values)
        acct_lookup = account_features.set_index('account_id')
    else:
        acct_set = set(account_features.index)
        acct_lookup = account_features
    
    scores = []
    
    for comm_id, nodes in communities.items():
        size = len(nodes)
        
        # Filter account features for nodes in this community
        valid_nodes = [n for n in nodes if n in acct_set]
        if not valid_nodes:
            continue
            
        comm_features = acct_lookup.loc[valid_nodes]
        
        fraud_count = int(comm_features['is_fraud'].sum()) if 'is_fraud' in comm_features.columns else 0
        fraud_rate = fraud_count / size if size > 0 else 0
        
        avg_risk = 0
        if risk_scores:
            risk_vals = [risk_scores.get(n, 0) for n in valid_nodes]
            avg_risk = float(np.mean(risk_vals)) if risk_vals else 0
            
        # Try to find total transaction volume
        total_volume = 0
        if 'total_amount_sent' in comm_features.columns:
            total_volume = float(comm_features['total_amount_sent'].sum())
            
        scores.append({
            'community_id': comm_id,
            'size': size,
            'fraud_count': fraud_count,
            'fraud_rate': fraud_rate,
            'avg_risk_score': avg_risk,
            'total_transaction_volume': total_volume,
            'internal_edge_density': 0
        })
        
    df_scores = pd.DataFrame(scores)
    if not df_scores.empty:
        df_scores = df_scores.sort_values('fraud_rate', ascending=False)
        
    return df_scores

def find_suspicious_communities(community_scores: pd.DataFrame, min_size: int = 3, min_fraud_rate: float = 0.1) -> pd.DataFrame:
    """Filter communities that meet suspicion criteria."""
    if community_scores.empty:
        return community_scores
    suspicious = community_scores[(community_scores['size'] >= min_size) & (community_scores['fraud_rate'] >= min_fraud_rate)]
    return suspicious

def trace_money_flow(G: nx.DiGraph, source_account: str, max_depth: int = 5) -> list:
    """BFS from source account following directed edges."""
    if source_account not in G:
        return []
        
    paths = []
    queue = deque([([source_account], [], 0.0)]) # (path, amounts, total_flow)
    
    while queue:
        path, amounts, total = queue.popleft()
        curr_node = path[-1]
        
        if len(path) > max_depth:
            continue
            
        is_leaf = True
        for neighbor in G.successors(curr_node):
            if neighbor not in path: # Avoid cycles
                is_leaf = False
                edge_data = G.get_edge_data(curr_node, neighbor)
                amount = edge_data.get('total_amount', 0) if edge_data else 0
                queue.append((path + [neighbor], amounts + [amount], total + amount))
                
        if is_leaf and len(path) > 1:
            paths.append({
                'path': path,
                'amounts': amounts,
                'total_flow': total
            })
            
    return paths

def find_flow_patterns(G: nx.DiGraph, fraud_accounts: list, max_depth: int = 3) -> pd.DataFrame:
    """Identify common flow patterns from fraud accounts."""
    patterns = []
    
    for account in fraud_accounts:
        if account not in G:
            continue
            
        out_degree = G.out_degree(account)
        in_degree = G.in_degree(account)
        
        pattern_type = "unknown"
        if out_degree > 3 and in_degree <= 1:
            pattern_type = "fan-out"
        elif in_degree > 3 and out_degree <= 1:
            pattern_type = "fan-in"
        elif in_degree == 1 and out_degree == 1:
            pattern_type = "chain"
            
        flows = trace_money_flow(G, account, max_depth)
        
        patterns.append({
            'account': account,
            'out_degree': out_degree,
            'in_degree': in_degree,
            'pattern': pattern_type,
            'num_paths': len(flows)
        })
        
    return pd.DataFrame(patterns)

def compute_network_risk(G: nx.DiGraph, account_features: pd.DataFrame, risk_scores: dict = None) -> pd.DataFrame:
    """Compute network-level risk for each account."""
    # Build fraud lookup from account_features
    if 'account_id' in account_features.columns:
        fraud_set = set(account_features[account_features['is_fraud'] == 1]['account_id'].values)
    else:
        fraud_set = set(account_features[account_features['is_fraud'] == 1].index)
    
    pr = nx.pagerank(G, alpha=0.85, weight='total_amount')
    clustering = nx.clustering(G.to_undirected())
    
    risk_data = []
    for node in G.nodes():
        neighbors = list(set(list(G.successors(node)) + list(G.predecessors(node))))
        
        fraud_neighbors = sum(1 for n in neighbors if n in fraud_set)
        high_risk_neighbors = 0
        if risk_scores:
            high_risk_neighbors = sum(1 for n in neighbors if risk_scores.get(n, 0) > 0.5)
                
        neighbor_fraud_rate = fraud_neighbors / len(neighbors) if neighbors else 0
        base_risk = risk_scores.get(node, 0) if risk_scores else 0
        
        network_risk = base_risk * 0.5 + neighbor_fraud_rate * 0.3 + (high_risk_neighbors / max(1, len(neighbors))) * 0.2
        
        risk_data.append({
            'account': node,
            'neighbor_fraud_rate': neighbor_fraud_rate,
            'high_risk_neighbor_count': high_risk_neighbors,
            'pagerank': pr.get(node, 0),
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node),
            'clustering_coefficient': clustering.get(node, 0),
            'network_risk_score': network_risk
        })
        
    df = pd.DataFrame(risk_data)
    if not df.empty:
        df.set_index('account', inplace=True)
    return df

def run_community_detection(account_features_path='data/account_features.pkl', 
                            edge_features_path='data/edge_features.pkl', 
                            embeddings_path='data/node_embeddings.npy') -> dict:
    """Run full community detection pipeline."""
    os.makedirs('outputs', exist_ok=True)
    print("Loading data...")
    
    try:
        with open(account_features_path, 'rb') as f:
            account_features = pickle.load(f)
        with open(edge_features_path, 'rb') as f:
            edge_features = pickle.load(f)
    except FileNotFoundError:
        print("Data files not found. Creating dummy data for testing...")
        account_features = pd.DataFrame({'is_fraud': [0, 1, 0, 1], 'total_amount_sent': [100, 200, 0, 500]}, index=['A', 'B', 'C', 'D'])
        edge_features = pd.DataFrame({'source': ['A', 'B', 'C'], 'target': ['B', 'C', 'D'], 'total_amount': [100, 200, 300]})
        
    # Take a subgraph of top 50K accounts for performance
    if len(account_features) > 50000:
        sort_col = 'total_txn_count' if 'total_txn_count' in account_features.columns else account_features.select_dtypes(include='number').columns[0]
        account_features = account_features.nlargest(50000, sort_col)
        
        if 'account_id' in account_features.columns:
            top_ids = set(account_features['account_id'].values)
        else:
            top_ids = set(account_features.index)
        edge_features = edge_features[(edge_features['source'].isin(top_ids)) & (edge_features['target'].isin(top_ids))]

    print("Building NetworkX graph...")
    G = build_networkx_graph(edge_features, account_features)
    
    print("Detecting communities...")
    communities = detect_communities(G)
    
    print("Scoring communities...")
    scores = score_communities(communities, account_features)
    
    print("Finding suspicious communities...")
    suspicious = find_suspicious_communities(scores)
    
    print("Computing network risk...")
    risk_df = compute_network_risk(G, account_features)
    
    # Save outputs
    scores.to_csv('outputs/community_scores.csv', index=False)
    suspicious.to_csv('outputs/suspicious_communities.csv', index=False)
    risk_df.to_csv('outputs/network_risk.csv')
    
    summary = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_communities': len(communities),
        'num_suspicious_communities': len(suspicious)
    }
    
    print("Pipeline completed. Summary:", summary)
    return summary

if __name__ == '__main__':
    run_community_detection()
