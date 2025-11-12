import torch
import pandas as pd
import networkx as nx
import random
from torch_geometric.data import Data
from contagion_model import balance_sheet_cascade

CSV_PATH = "sample_interbank_network.csv"
NUM_CRISIS_SAMPLES = 50  # Number of crisis scenarios

def load_and_prepare_data() -> Data:
    # --- Load network ---
    df_edges = pd.read_csv(CSV_PATH)
    G = nx.DiGraph()
    for _, row in df_edges.iterrows():
        s, t, w = str(row['source']), str(row['target']), float(row['exposure'])
        G.add_edge(s, t, exposure=w)
    for n in G.nodes():
        G.nodes[n].setdefault('capital', random.uniform(5.0, 20.0))
    
    all_nodes = list(G.nodes())
    
    # --- Load node metrics ---
    df_metrics = pd.read_csv("network_metrics_summary.csv", index_col=0)
    feature_cols = ['capital', 'eigenvector', 'in_degree', 'out_degree']
    x = torch.tensor(df_metrics[feature_cols].values, dtype=torch.float)
    
    # --- Edge tensors ---
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    source_nodes, target_nodes, exposure_values = [], [], []
    for u, v, data in G.edges(data=True):
        source_nodes.append(node_to_idx[u])
        target_nodes.append(node_to_idx[v])
        exposure_values.append(data.get('exposure', 1.0))
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(exposure_values, dtype=torch.float).unsqueeze(1)
    
    # --- Generate multiple crisis labels ---
    all_x_list, all_y_list = [], []
    initial_trigger_nodes = random.sample(all_nodes, k=NUM_CRISIS_SAMPLES)
    for node in initial_trigger_nodes:
        cascade = balance_sheet_cascade(G, [node])
        final_failed = cascade['failed_nodes']
        y = [1.0 if n in final_failed else 0.0 for n in all_nodes]
        all_x_list.append(x)
        all_y_list.append(torch.tensor(y, dtype=torch.float).unsqueeze(1))
    
    x_combined = torch.cat(all_x_list, dim=0)
    y_combined = torch.cat(all_y_list, dim=0)
    
    # --- Batch edge_index with offsets ---
    num_nodes = x.shape[0]
    edge_indices_list = []
    for i in range(NUM_CRISIS_SAMPLES):
        offset = i * num_nodes
        edge_indices_list.append(edge_index + offset)
    edge_index_batched = torch.cat(edge_indices_list, dim=1)
    edge_attr_batched = edge_attr.repeat(NUM_CRISIS_SAMPLES, 1)
    
    data = Data(x=x_combined, edge_index=edge_index_batched, edge_attr=edge_attr_batched, y=y_combined)
    return data
