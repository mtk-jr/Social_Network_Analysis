import networkx as nx
from typing import Dict, Any


def compute_node_metrics(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    
    und = G.to_undirected()

    # Degree-related measures
    deg = dict(und.degree())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_c = nx.degree_centrality(und)
    bet = nx.betweenness_centrality(und)

   
    try:
        eig = nx.eigenvector_centrality(und, max_iter=500)
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    
    pr = nx.pagerank(G)

    metrics = {}
    for n in G.nodes():
        metrics[n] = {
            'degree': deg.get(n, 0),
            'in_degree': in_deg.get(n, 0),
            'out_degree': out_deg.get(n, 0),
            'degree_centrality': deg_c.get(n, 0.0),
            'betweenness': bet.get(n, 0.0),
            'eigenvector': eig.get(n, 0.0),
            'pagerank': pr.get(n, 0.0),
            'capital': G.nodes[n].get('capital', None),
        }

    return metrics


def compute_global_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    
    und = G.to_undirected()

    metrics = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(und),
        'avg_clustering': nx.average_clustering(und) if G.number_of_nodes() > 0 else 0.0,
        'avg_shortest_path_length': nx.average_shortest_path_length(und) if nx.is_connected(und) else None,
    }

    return metrics
