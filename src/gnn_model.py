import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from torch_geometric.nn import SAGEConv, GATConv
except ImportError:
    print("PyTorch Geometric not found. Please install it to use GNN models.")
    SAGEConv, GATConv = None, None

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=2, dropout=0.3):
        super(GraphSAGEModel, self).__init__()
        self.dropout = dropout
        if SAGEConv is not None:
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if SAGEConv is None: return torch.zeros((x.size(0), 2))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

    def get_embeddings(self, x, edge_index):
        if SAGEConv is None: return torch.zeros((x.size(0), self.lin.in_features))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.dropout = dropout
        if GATConv is not None:
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if GATConv is None: return torch.zeros((x.size(0), 2))
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

    def get_embeddings(self, x, edge_index):
        if GATConv is None: return torch.zeros((x.size(0), self.lin.in_features))
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return x

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def train_epoch(model, data, optimizer, criterion, mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_epoch(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[mask], dim=1)[:, 1]
        preds = out[mask].argmax(dim=1)
        
        loss = F.cross_entropy(out[mask], data.y[mask]).item()
        
    return {
        'loss': loss,
        'predictions': preds.cpu().numpy(),
        'probabilities': probs.cpu().numpy(),
        'true_labels': data.y[mask].cpu().numpy()
    }

def train_gnn(model, data, model_name: str, epochs=50, lr=0.001, patience=20) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_pr_auc = -1
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_pr_auc': []}
    
    os.makedirs('models', exist_ok=True)
    best_model_path = f'models/{model_name}_best.pt'
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, data, optimizer, criterion, data.train_mask)
        
        val_res = evaluate_epoch(model, data, data.val_mask)
        val_loss = val_res['loss']
        val_pr_auc = average_precision_score(val_res['true_labels'], val_res['probabilities'])
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pr_auc'].append(val_pr_auc)
        
        scheduler.step(val_loss)
        
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    model.load_state_dict(torch.load(best_model_path))
    return history

def extract_embeddings(model, data) -> np.ndarray:
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        emb = model.get_embeddings(data.x, data.edge_index)
    return emb.cpu().numpy()

def run_gnn_pipeline(graph_path: str = 'data/graph_data.pt', epochs: int = 50) -> dict:
    if SAGEConv is None:
        print("Skipping GNN pipeline since PyTorch Geometric is not installed.")
        return {}

    if not os.path.exists(graph_path):
        print(f"Warning: {graph_path} not found. Creating dummy data.")
        try:
            from torch_geometric.data import Data
            num_nodes = 500
            x = torch.randn(num_nodes, 10)
            edge_index = torch.randint(0, num_nodes, (2, 2000))
            y = torch.randint(0, 2, (num_nodes,))
            train_mask = torch.rand(num_nodes) < 0.6
            val_mask = (torch.rand(num_nodes) < 0.2) & ~train_mask
            test_mask = ~(train_mask | val_mask)
            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        except ImportError:
            return {}
    else:
        data = torch.load(graph_path, weights_only=False)

    in_channels = data.x.size(1)
    
    print(f"Training GraphSAGE ({epochs} epochs)...")
    sage_model = GraphSAGEModel(in_channels=in_channels)
    train_gnn(sage_model, data, "GraphSAGE", epochs=epochs)
    
    print(f"Training GAT ({epochs} epochs)...")
    gat_model = GATModel(in_channels=in_channels)
    train_gnn(gat_model, data, "GAT", epochs=epochs)
    
    results = {}
    
    for name, model in [("GraphSAGE", sage_model), ("GAT", gat_model)]:
        test_res = evaluate_epoch(model, data, data.test_mask)
        y_true = test_res['true_labels']
        y_pred = test_res['predictions']
        y_prob = test_res['probabilities']
        
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        results[name] = metrics
        
    os.makedirs('data', exist_ok=True)
    embeddings = extract_embeddings(sage_model, data)
    np.save('data/node_embeddings.npy', embeddings)
    
    return results

if __name__ == '__main__':
    results = run_gnn_pipeline()
    if results:
        import pandas as pd
        res_df = pd.DataFrame(results).T
        print("\nGNN Models Comparison (Test Set):")
        print(res_df[['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']])
