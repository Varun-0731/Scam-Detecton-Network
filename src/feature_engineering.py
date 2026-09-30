import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute account-level features based on transaction history.
    Accounts can be senders, receivers, or both.
    """
    print("Computing sender features...")
    # Sender features
    sender_aggs = df.groupby('nameOrig').agg(
        out_txn_count=('step', 'count'),
        total_amount_sent=('amount', 'sum'),
        avg_amount_sent=('amount', 'mean'),
        max_amount_sent=('amount', 'max'),
        std_amount_sent=('amount', 'std'),
        unique_counterparties_out=('nameDest', 'nunique'),
        zero_balance_after_txn=('is_balance_zeroed', 'sum'),
        active_steps=('step', 'nunique')
    ).reset_index()
    
    # Compute type counts for senders
    type_counts = pd.crosstab(df['nameOrig'], df['type']).reset_index()
    # Ensure all required types exist
    for t in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
        if t not in type_counts.columns:
            type_counts[t] = 0
            
    type_counts = type_counts.rename(columns={
        'CASH_IN': 'cashin_count',
        'CASH_OUT': 'cashout_count',
        'DEBIT': 'debit_count',
        'PAYMENT': 'payment_count',
        'TRANSFER': 'transfer_count'
    })
    
    sender_aggs = sender_aggs.merge(type_counts, on='nameOrig', how='left')
    
    # Balance change ratio
    df['bal_change_ratio_tmp'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1.0)
    bal_change = df.groupby('nameOrig')['bal_change_ratio_tmp'].mean().reset_index(name='balance_change_ratio')
    sender_aggs = sender_aggs.merge(bal_change, on='nameOrig', how='left')
    df.drop(columns=['bal_change_ratio_tmp'], inplace=True)
    
    # Step gap (time between txns)
    def calc_avg_gap(steps):
        if len(steps) <= 1:
            return 0.0
        return (steps.max() - steps.min()) / (len(steps) - 1)
        
    step_gap = df.groupby('nameOrig')['step'].apply(calc_avg_gap).reset_index(name='avg_step_gap')
    sender_aggs = sender_aggs.merge(step_gap, on='nameOrig', how='left')
    
    print("Computing receiver features...")
    # Receiver features
    receiver_aggs = df.groupby('nameDest').agg(
        in_txn_count=('step', 'count'),
        total_amount_received=('amount', 'sum'),
        avg_amount_received=('amount', 'mean'),
        max_amount_received=('amount', 'max'),
        unique_counterparties_in=('nameOrig', 'nunique')
    ).reset_index()
    
    print("Merging account features...")
    # Rename IDs for merging
    sender_aggs = sender_aggs.rename(columns={'nameOrig': 'account_id'})
    receiver_aggs = receiver_aggs.rename(columns={'nameDest': 'account_id'})
    
    # Merge both
    account_features = pd.merge(sender_aggs, receiver_aggs, on='account_id', how='outer')
    
    # Fill NaN for accounts that are only senders or only receivers
    account_features = account_features.fillna(0)
    
    # Calculate derived metrics across both
    account_features['total_txn_count'] = account_features['out_txn_count'] + account_features['in_txn_count']
    account_features['in_out_ratio'] = account_features['in_txn_count'] / (account_features['out_txn_count'] + 1.0)
    
    return account_features

def label_accounts(df: pd.DataFrame, account_features: pd.DataFrame) -> pd.DataFrame:
    """
    Label an account as fraud (1) if it originated any fraudulent transaction.
    """
    fraud_origins = df[df['isFraud'] == 1]['nameOrig'].unique()
    
    account_features['is_fraud'] = account_features['account_id'].isin(fraud_origins).astype(int)
    
    fraud_count = account_features['is_fraud'].sum()
    total_count = len(account_features)
    print(f"Labeled {fraud_count} fraud accounts out of {total_count} total accounts "
          f"({fraud_count/total_count*100:.3f}%).")
          
    return account_features

def compute_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction data into edges between (nameOrig, nameDest).
    """
    print("Computing edge features...")
    
    edges = df.groupby(['nameOrig', 'nameDest']).agg(
        txn_count=('step', 'count'),
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        max_amount=('amount', 'max'),
        type_diversity=('type', 'nunique'),
        has_fraud=('isFraud', 'max')
    ).reset_index()
    
    edges = edges.rename(columns={
        'nameOrig': 'source',
        'nameDest': 'target'
    })
    
    print(f"Generated {len(edges)} unique edges.")
    return edges

def scale_features(account_features: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize continuous feature columns.
    """
    print(f"Scaling {len(feature_cols)} features...")
    scaler = StandardScaler()
    scaled_df = account_features.copy()
    scaled_df[feature_cols] = scaler.fit_transform(account_features[feature_cols])
    return scaled_df, scaler

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_data_path = os.path.join(project_root, 'data', 'processed_transactions.pkl')
    account_features_path = os.path.join(project_root, 'data', 'account_features.pkl')
    edge_features_path = os.path.join(project_root, 'data', 'edge_features.pkl')
    
    try:
        print(f"Loading processed data from {processed_data_path}...")
        df = pd.read_pickle(processed_data_path)
        
        # 1. Account Features
        accounts = compute_account_features(df)
        
        # 2. Labels
        accounts = label_accounts(df, accounts)
        
        # 3. Edge Features
        edges = compute_edge_features(df)
        
        # 4. Scale Node Features (excluding ID and label)
        cols_to_scale = [c for c in accounts.columns if c not in ['account_id', 'is_fraud']]
        accounts_scaled, scaler = scale_features(accounts, cols_to_scale)
        
        # Save features
        print("Saving account and edge features...")
        accounts_scaled.to_pickle(account_features_path)
        edges.to_pickle(edge_features_path)
        print("Done.")
        
    except Exception as e:
        print(f"Feature engineering failed: {e}")
