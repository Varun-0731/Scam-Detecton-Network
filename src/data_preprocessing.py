import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple, Dict, Any, Optional

def load_data(filepath: str, sample_frac: Optional[float] = None) -> pd.DataFrame:
    """
    Load CSV with optimized dtypes.
    If sample_frac is provided, take a stratified sample preserving all fraud cases.
    """
    print(f"Loading data from {filepath}...")
    dtypes = {
        'step': 'int16',
        'type': 'category',
        'amount': 'float32',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'nameDest': 'object',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8'
    }
    
    try:
        df = pd.read_csv(filepath, dtype=dtypes)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        raise
        
    print(f"Original dataset shape: {df.shape}")
    
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        fraud_df = df[df['isFraud'] == 1]
        legit_df = df[df['isFraud'] == 0]
        
        # Sample legitimate transactions
        legit_sampled = legit_df.sample(frac=sample_frac, random_state=42)
        
        # Combine all fraud with sampled legit
        df = pd.concat([fraud_df, legit_sampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Sampled dataset shape: {df.shape} (frac={sample_frac})")
        
    print("Memory usage:")
    print(df.memory_usage(deep=True).sum() / (1024 * 1024), "MB")
    print("Data types:")
    print(df.dtypes)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check missing values, remove duplicates, and add derived columns.
    """
    print("Cleaning data...")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing values found:")
        print(missing[missing > 0])
        df = df.dropna()
    else:
        print("No missing values found.")
        
    # Duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} duplicate rows.")
        
    # Derived columns
    print("Adding derived columns...")
    df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['amount_ratio_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1.0)
    df['is_balance_zeroed'] = ((df['newbalanceOrig'] == 0) & (df['amount'] > 0)).astype('int8')
    df['error_balance_orig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['error_balance_dest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    print("Data cleaning complete.")
    return df

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode and label encode the 'type' column.
    """
    print("Encoding data...")
    
    # Label encode
    type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    df['type_encoded'] = df['type'].map(type_map).astype('int8')
    
    # One-hot encode (keeping original)
    dummies = pd.get_dummies(df['type'], prefix='type', dtype='int8')
    df = pd.concat([df, dummies], axis=1)
    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by isFraud. Returns (train, val, test).
    """
    print("Splitting data...")
    # First split off test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['isFraud'], random_state=42
    )
    
    # Now split train_val into train and val
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, stratify=train_val_df['isFraud'], random_state=42
    )
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    return train_df, val_df, test_df

def get_fraud_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return summary statistics of the dataset.
    """
    total_tx = len(df)
    fraud_tx = int(df['isFraud'].sum())
    
    summary = {
        'total_transactions': total_tx,
        'fraud_count': fraud_tx,
        'fraud_rate': fraud_tx / total_tx if total_tx > 0 else 0,
        'fraud_by_type': df[df['isFraud'] == 1]['type'].value_counts().to_dict(),
        'flagged_count': int(df['isFlaggedFraud'].sum()),
        'unique_senders': df['nameOrig'].nunique(),
        'unique_receivers': df['nameDest'].nunique()
    }
    return summary

if __name__ == '__main__':
    # Define paths relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'PS_20174392719_1491204439457_log.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed_transactions.pkl')
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    try:
        # 1. Load data
        df = load_data(data_path)
        
        # 2. Clean data
        df = clean_data(df)
        
        # 3. Encode data
        df = encode_data(df)
        
        # 4. Print Summary
        summary = get_fraud_summary(df)
        print("\n=== Dataset Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
            
        # 5. Save processed data
        print(f"\nSaving processed data to {processed_data_path}...")
        df.to_pickle(processed_data_path)
        print("Done.")
        
    except Exception as e:
        print(f"Data preprocessing failed: {e}")
