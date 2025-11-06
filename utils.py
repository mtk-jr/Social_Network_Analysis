import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple



def load_network_from_csv(path: str) -> nx.DiGraph:
    
    df = pd.read_csv(path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['lender'], row['borrower'], weight=row['exposure'])
    return G



def compute_network_metrics(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    
    metrics = {
        'degree': dict(G.degree()),
        'betweenness': nx.betweenness_centrality(G, normalized=True),
        'eigenvector': nx.eigenvector_centrality(G.to_undirected(), max_iter=1000),
        'clustering': nx.clustering(G.to_undirected())
    }

    return metrics


def print_top_influential_nodes(metrics: Dict[str, Dict[str, float]], top_n: int = 5):
    
    for key, vals in metrics.items():
        top_nodes = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\nTop {top_n} nodes by {key} centrality:")
        for node, val in top_nodes:
            print(f"  {node:<15}  {val:.4f}")



def simulate_contagion(
    G: nx.DiGraph,
    initial_failed: Set[str],
    threshold: float = 0.3,
    max_steps: int = 10
) -> List[Set[str]]:
    
    steps = []
    failed = set(initial_failed)
    steps.append(set(failed))

    for _ in range(max_steps):
        new_failures = set()
        for node in G.nodes():
            if node in failed:
                continue

            incoming_edges = G.in_edges(node, data=True)
            total_exposure = sum([d['weight'] for _, _, d in incoming_edges])
            failed_exposure = sum([d['weight'] for src, _, d in incoming_edges if src in failed])

            if total_exposure > 0 and (failed_exposure / total_exposure) >= threshold:
                new_failures.add(node)

        if not new_failures:
            break

        failed.update(new_failures)
        steps.append(set(failed))

    return steps



def generate_demo_network(n_banks: int = 10, seed: int = 42) -> nx.DiGraph:
    
    np.random.seed(seed)
    G = nx.gnp_random_graph(n_banks, p=0.3, directed=True)
    DG = nx.DiGraph()
    for u, v in G.edges():
        DG.add_edge(f"Bank_{u}", f"Bank_{v}", weight=np.random.uniform(0.1, 1.0))
    return DG



if __name__ == "__main__":
    
    G = generate_demo_network(8)
    metrics = compute_network_metrics(G)
    print_top_influential_nodes(metrics)

    
    initial_failed = {"Bank_2"}
    cascade = simulate_contagion(G, initial_failed, threshold=0.4)
    for step, failed_set in enumerate(cascade):
        print(f"Step {step}: {failed_set}")
