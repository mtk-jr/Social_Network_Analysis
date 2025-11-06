import networkx as nx
import pandas as pd
import random
from typing import Tuple


def load_network_from_csv(path: str, weighted: bool = True) -> nx.DiGraph:
    
    df = pd.read_csv(path)
    G = nx.DiGraph()

    for _, row in df.iterrows():
        s = str(row['source'])
        t = str(row['target'])
        w = float(row['exposure']) if weighted and not pd.isna(row['exposure']) else 1.0

        
        if G.has_edge(s, t):
            G[s][t]['exposure'] += w
        else:
            G.add_edge(s, t, exposure=w)

    
    for n in G.nodes():
        G.nodes[n].setdefault('capital', round(random.uniform(5.0, 20.0), 2))

    return G


def generate_synthetic_network(n_banks: int = 30, prob: float = 0.08, seed: int = 42) -> nx.DiGraph:
    
    random.seed(seed)
    G_und = nx.erdos_renyi_graph(n=n_banks, p=prob, seed=seed)


    mapping = {i: f'Bank_{i+1}' for i in range(n_banks)}
    G_und = nx.relabel_nodes(G_und, mapping)

    G = nx.DiGraph()

    for u, v in G_und.edges():
       
        if random.random() < 0.6:
            w = round(random.uniform(0.5, 8.0), 2)
            G.add_edge(u, v, exposure=w)
        if random.random() < 0.4:
            w = round(random.uniform(0.2, 5.0), 2)
            G.add_edge(v, u, exposure=w)

    
    for n in range(1, n_banks + 1):
        name = f'Bank_{n}'
        cap = round(random.uniform(8.0, 30.0), 2)
        if not G.has_node(name):
            G.add_node(name)
        G.nodes[name]['capital'] = cap

    return G
