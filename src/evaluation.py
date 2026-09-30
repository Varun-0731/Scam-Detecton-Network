import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, 
                             roc_auc_score, average_precision_score, confusion_matrix, 
                             roc_curve, precision_recall_curve, balanced_accuracy_score)

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'specificity': float(specificity),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred))
    }
    return metrics

def print_comparison_table(results: dict) -> pd.DataFrame:
    records = []
    for model_name, metrics in results.items():
        record = {'Model': model_name}
        for k, v in metrics.items():
            if k not in ['confusion_matrix', 'y_true', 'y_pred', 'y_prob']:
                record[k] = v
        records.append(record)
        
    df = pd.DataFrame(records).set_index('Model')
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(df.to_string())
    return df

def plot_roc_curves(results: dict, save_path: str = 'outputs/roc_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    for model_name, data in results.items():
        if 'y_true' in data and 'y_prob' in data:
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
            auc = roc_auc_score(data['y_true'], data['y_prob'])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_pr_curves(results: dict, save_path: str = 'outputs/pr_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    for model_name, data in results.items():
        if 'y_true' in data and 'y_prob' in data:
            precision, recall, _ = precision_recall_curve(data['y_true'], data['y_prob'])
            pr_auc = average_precision_score(data['y_true'], data['y_prob'])
            plt.plot(recall, precision, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(results: dict, save_path: str = 'outputs/confusion_matrices.png'):
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not installed. Skipping confusion matrix plot.")
        return
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_models = len(results)
    if n_models == 0:
        return
        
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for idx, (model_name, data) in enumerate(results.items()):
        if 'confusion_matrix' in data:
            cm = data['confusion_matrix']
            if isinstance(cm, dict):
                cm_arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
            else:
                cm_arr = np.array(cm)
                
            sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(model_name)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_evaluation_report(results: dict, save_path: str = 'outputs/evaluation_report.txt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 20 + "\n")
            for k, v in metrics.items():
                if k not in ['y_true', 'y_pred', 'y_prob']:
                    if isinstance(v, dict):
                        f.write(f"{k}:\n")
                        for sub_k, sub_v in v.items():
                            f.write(f"  {sub_k}: {sub_v}\n")
                    else:
                        f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")
            f.write("\n")

def run_full_evaluation(baseline_results: dict, gnn_results: dict) -> pd.DataFrame:
    all_results = {**baseline_results, **gnn_results}
    
    plot_roc_curves(all_results)
    plot_pr_curves(all_results)
    plot_confusion_matrices(all_results)
    
    generate_evaluation_report(all_results)
    df = print_comparison_table(all_results)
    
    return df

if __name__ == '__main__':
    # Example usage for testing
    dummy_res = {}
    for name in ['LogisticRegression', 'XGBoost', 'GraphSAGE']:
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.rand(1000)
        y_pred = (y_prob > 0.5).astype(int)
        dummy_res[name] = compute_metrics(y_true, y_pred, y_prob)
        dummy_res[name].update({'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred})
        
    run_full_evaluation(dummy_res, {})
