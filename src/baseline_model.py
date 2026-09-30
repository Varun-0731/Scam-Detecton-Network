import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, 
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as e:
    print(f"⚠️  XGBoost not available. Will skip XGBoost baseline.")
    HAS_XGBOOST = False

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

def train_logistic_regression(X_train, y_train, X_val, y_val) -> tuple[LogisticRegression, dict]:
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_val, y_val, "Logistic Regression (Val)")
    return model, metrics

def train_random_forest(X_train, y_train, X_val, y_val) -> tuple[RandomForestClassifier, dict]:
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_val, y_val, "Random Forest (Val)")
    return model, metrics

def train_xgboost(X_train, y_train, X_val, y_val) -> tuple:
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        use_label_encoder=False, 
        eval_metric='aucpr',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    metrics = evaluate_model(model, X_val, y_val, "XGBoost (Val)")
    return model, metrics

def get_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    if hasattr(model, 'coef_'):
        importances = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))
        
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df['Abs_Importance'] = df['Importance'].abs()
    return df.sort_values(by='Abs_Importance', ascending=False).drop(columns=['Abs_Importance'])

def run_baselines(account_features_path: str = 'data/account_features.pkl') -> dict:
    if not os.path.exists(account_features_path):
        print(f"Warning: Data file {account_features_path} not found. Generating dummy data.")
        X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'f_{i}' for i in range(10)])
        y = np.random.randint(0, 2, 1000)
    else:
        df = pd.read_pickle(account_features_path)
        y = df['is_fraud'].values
        X = df.drop(columns=[col for col in ['is_fraud', 'account_id'] if col in df.columns])
    
    feature_names = X.columns.tolist()
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)
    
    models = {}
    
    print("Training Logistic Regression...")
    lr_model, _ = train_logistic_regression(X_train, y_train, X_val, y_val)
    models['LogisticRegression'] = lr_model
    
    print("Training Random Forest...")
    rf_model, _ = train_random_forest(X_train, y_train, X_val, y_val)
    models['RandomForest'] = rf_model
    
    if HAS_XGBOOST:
        print("Training XGBoost...")
        xgb_model, _ = train_xgboost(X_train, y_train, X_val, y_val)
        models['XGBoost'] = xgb_model
    else:
        print("Skipping XGBoost (libomp not installed).")
    
    os.makedirs('models', exist_ok=True)
    results = {}
    
    for name, model in models.items():
        joblib.dump(model, f'models/{name.lower()}.joblib')
        test_metrics = evaluate_model(model, X_test, y_test, f"{name} (Test)")
        results[name] = test_metrics
        
        fi_df = get_feature_importance(model, feature_names, name)
        fi_df.to_csv(f'models/{name.lower()}_feature_importance.csv', index=False)
        
    return results

if __name__ == '__main__':
    results = run_baselines()
    res_df = pd.DataFrame(results).T
    print("\nBaseline Models Comparison (Test Set):")
    print(res_df[['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']])
