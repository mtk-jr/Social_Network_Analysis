import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from collections import Counter

# -----------------------------
# 1. Load node features
# -----------------------------
nodes_df = pd.read_csv("network_metrics_summary.csv", index_col=0)  # bank names as index
features = nodes_df[['degree','in_degree','out_degree','degree_centrality',
                     'betweenness','eigenvector','pagerank','capital']].values

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)
features = torch.tensor(features, dtype=torch.float)

# Map bank names to indices
bank2idx = {b: i for i, b in enumerate(nodes_df.index)}

# Load edges
edges_df = pd.read_csv("sample_interbank_network.csv")
edges_src = [bank2idx[b] for b in edges_df['source']]
edges_tgt = [bank2idx[b] for b in edges_df['target']]
edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)

# -----------------------------
# 2. Labels (example: binary)
# -----------------------------
num_nodes = len(nodes_df)
# Replace this with real labels if available
labels = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

# Create graph data object
data = Data(x=features, edge_index=edge_index, y=labels)
train_mask, test_mask = torch.rand(num_nodes) < 0.8, torch.rand(num_nodes) >= 0.8
data.train_mask = train_mask
data.test_mask = test_mask

# -----------------------------
# 3. Compute class weights
# -----------------------------
counts = Counter(labels[data.train_mask].tolist())
weight_0 = 1.0 / counts.get(0, 1)
weight_1 = 1.0 / counts.get(1, 1)
class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)

# -----------------------------
# 4. GCN model
# -----------------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=features.shape[1], hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# -----------------------------
# 5. Training and testing
# -----------------------------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data)
    preds = out.argmax(dim=1)
    acc = accuracy_score(data.y[data.test_mask], preds[data.test_mask])
    f1 = f1_score(data.y[data.test_mask], preds[data.test_mask])
    try:
        roc = roc_auc_score(data.y[data.test_mask], F.softmax(out[data.test_mask], dim=1)[:,1].detach().numpy())
    except:
        roc = 0.5
    return acc, f1, roc

# -----------------------------
# 6. Training loop
# -----------------------------
for epoch in range(1, 301):
    loss = train()
    if epoch % 25 == 0:
        acc, f1, roc = test()
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Test ACC {acc:.4f} | F1 {f1:.4f} | ROC-AUC {roc:.4f}")

acc, f1, roc = test()
print(f"\nFinal Test Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")

model.eval()
with torch.no_grad():
    out = model(data)
    probs = F.softmax(out, dim=1)[:, 1]  # Probability of being 'high risk'
    bank_names = nodes_df.index.tolist()
    risk_scores = {bank_names[i]: float(probs[i]) for i in range(len(bank_names))}
    high_risk_banks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
    print("High risk banks (sorted):")
    threshold = 0.7
    for bank, score in high_risk_banks:
        if score > threshold:
            print(f"{bank}: {score:.2f}")


