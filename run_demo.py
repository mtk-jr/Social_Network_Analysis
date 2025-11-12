# run_demo.py
import os
import sys
import argparse
import random
import networkx as nx
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from build_network import load_network_from_csv, generate_synthetic_network
from metrics_analysis import compute_node_metrics, compute_global_metrics
from contagion_model import simulate_contagion, balance_sheet_cascade
from visualization import plot_static_failure, plot_plotly_network


def parse_attack_flag(attack_str):
    """
    Parse attack string formats:
      - "top_degree:3"
      - "top_betweenness:5"
      - "random:4"
    Returns tuple (strategy_name, k) or (None, 0) on parse fail.
    """
    if not attack_str:
        return None, 0
    try:
        if ":" in attack_str:
            strategy, k = attack_str.split(":", 1)
            return strategy.strip().lower(), int(k)
        else:
            return attack_str.strip().lower(), None
    except Exception:
        return None, 0


def pick_top_k_by_metric(G, metric_name, k):
    """
    Compute metric and return top-k node list.
    Supported metric_name: 'degree', 'betweenness', 'eigenvector' (falls back safely)
    """
    nodelist = list(G.nodes())
    if k is None or k <= 0:
        return []

    try:
        if metric_name == "degree":
            metric = nx.degree_centrality(G)
        elif metric_name == "betweenness":
            metric = nx.betweenness_centrality(G)
        elif metric_name == "eigenvector":
            # eigenvector may fail on directed graphs or for large networks, fallback gracefully
            try:
                metric = nx.eigenvector_centrality_numpy(G.to_undirected())
            except Exception:
                metric = nx.eigenvector_centrality(G.to_undirected(), max_iter=200)
        else:
            # unknown metric
            return []
    except Exception as e:
        print(f"⚠️  Could not compute metric '{metric_name}': {e}")
        return []

    sorted_nodes = sorted(metric.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


def resolve_initial_failures(G, args):
    """
    Resolve initial failures given args.initial and args.attack.
    Returns a validated list of node names that exist in G.
    """
    nodes_set = set(G.nodes())

    # 1) If attack strategy provided, use it
    if args.attack:
        strategy, k = parse_attack_flag(args.attack)
        if strategy in ("top_degree", "top-degree", "degree", "top_degree"):
            chosen = pick_top_k_by_metric(G, "degree", k)
            print(f"🔎 Attack strategy '{strategy}' selected top-{k}: {chosen}")
            return chosen
        if strategy in ("top_betweenness", "top-betweenness", "betweenness"):
            chosen = pick_top_k_by_metric(G, "betweenness", k)
            print(f"🔎 Attack strategy '{strategy}' selected top-{k}: {chosen}")
            return chosen
        if strategy in ("top_eigen", "top-eigen", "eigenvector"):
            chosen = pick_top_k_by_metric(G, "eigenvector", k)
            print(f"🔎 Attack strategy '{strategy}' selected top-{k}: {chosen}")
            return chosen
        if strategy in ("random",):
            k = k or 1
            chosen = random.sample(list(G.nodes()), min(k, G.number_of_nodes()))
            print(f"🔀 Random attack selected: {chosen}")
            return chosen
        print(f"⚠️  Unknown attack strategy '{args.attack}'. Ignoring --attack flag.")

    # 2) If explicit initial list provided, validate names
    if args.initial:
        provided = args.initial
        valid = [b for b in provided if b in nodes_set]
        invalid = [b for b in provided if b not in nodes_set]

        if valid:
            if invalid:
                print(f"⚠️  The following specified banks were NOT found and will be ignored: {invalid}")
            print(f"✅ Using provided initial failures (valid): {valid}")
            return valid
        else:
            print(f"⚠️  None of the provided initial banks exist in the network: {provided}")

    # 3) If no valid initial provided, check args.csv vs default behavior:
    # fallback: pick a random node (make reproducible if seed is present)
    fallback = [random.choice(list(G.nodes()))]
    print(f"ℹ️  No valid initial banks provided. Falling back to random initial failure: {fallback}")
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Run Financial Contagion Demo")
    parser.add_argument("--csv", type=str, help="Path to CSV (source,target,exposure)")
    parser.add_argument("--n", type=int, default=20, help="Number of banks for synthetic network")
    parser.add_argument("--prob", type=float, default=0.08, help="Edge probability for synthetic network")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
    parser.add_argument("--initial", type=str, nargs='+', help="List of initially failed banks (exact names)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Failure threshold for contagion")
    parser.add_argument("--p_fail", type=float, default=None, help="Probabilistic contagion probability")
    parser.add_argument("--attack", type=str, default=None,
                        help="Optional attack strategy: top_degree:K, top_betweenness:K, top_eigen:K, random:K")
    args = parser.parse_args()

    # set seeds for reproducibility
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except Exception:
        pass

    # Load dataset (default to your sample_interbank_network.csv in data/)
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = "sample_interbank_network.csv"

    print(f"\n📂 Loading interbank network from {csv_path}")
    G = load_network_from_csv(csv_path)

    # Ensure graph has 'weight' attribute expected by downstream code (normalize from 'exposure' if needed)
    # If edges use 'exposure' rename to 'weight' for compatibility
    # (many functions expect 'weight'; adjust only if the edge attribute exists)
    sample_edge_data = None
    try:
        # peek at one edge attribute name
        for u, v, d in G.edges(data=True):
            if d:
                sample_edge_data = list(d.keys())[0]
            break
    except Exception:
        pass

    if sample_edge_data and sample_edge_data != "weight":
        # copy that attribute to 'weight' if weight doesn't already exist
        attr = sample_edge_data
        nx.set_edge_attributes(G, { (u,v): d.get(attr) for u,v,d in G.edges(data=True) }, name='weight')

    print(f"✅ Network built: {G.number_of_nodes()} banks, {G.number_of_edges()} exposures")

    # Compute metrics
    print("\n📊 Computing network metrics...")
    node_metrics = compute_node_metrics(G)
    global_metrics = compute_global_metrics(G)

    print("\n🌐 Global Network Metrics:")
    for k, v in global_metrics.items():
        print(f"  {k:<25}: {v}")

    df_metrics = pd.DataFrame(node_metrics).T
    print("\n🏦 Node-Level Metrics (Top 10 by Degree Centrality):")
    # safe sort if column exists
    if 'degree_centrality' in df_metrics.columns:
        print(df_metrics.sort_values('degree_centrality', ascending=False).head(10))
    else:
        print(df_metrics.head(10))

    # Resolve initial failures (validates names and/or uses attack strategies)
    initial_failures = resolve_initial_failures(G, args)

    print(f"\n⚠️  Simulating contagion (threshold={args.threshold}) starting from: {initial_failures}")
    failed_nodes = simulate_contagion(G, initial_failures, threshold=args.threshold, p_fail=args.p_fail)
    print(f"💥 Total failed banks (threshold contagion): {len(failed_nodes)}")
    print(f"🧾 Failed banks: {sorted(failed_nodes)}")

    # Run balance sheet cascade simulation with try/except to avoid networkx KeyError if bad node slipped in
    try:
        print("\n💰 Running balance sheet cascade simulation...")
        cascade_result = balance_sheet_cascade(G, initial_failures)
        print(f"💣 Total failed banks (balance sheet contagion): {len(cascade_result['failed_nodes'])}")
        print(f"🔁 Failure propagation steps: {len(cascade_result['history'])}")
        print(f"📉 Final failed banks: {sorted(cascade_result['failed_nodes'])}")
    except nx.NetworkXError as e:
        print(f"❗ Error during balance_sheet_cascade: {e}")
        # defensive fallback: filter initial_failures and try again
        valid_initials = [b for b in initial_failures if b in G.nodes()]
        if not valid_initials:
            valid_initials = [random.choice(list(G.nodes()))]
            print(f"ℹ️  After filtering invalid names, falling back to random: {valid_initials}")
        cascade_result = balance_sheet_cascade(G, valid_initials)
        print(f"📉 Final failed banks (after recovery): {sorted(cascade_result['failed_nodes'])}")

    # Visualization
    print("\n🖼️  Generating network visualizations...")
    try:
        plot_static_failure(G, failed_nodes, title="Threshold Contagion Results")
        fig = plot_plotly_network(G, failed=cascade_result['failed_nodes'], title="Balance Sheet Contagion")
        fig.show()
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")

    # Save metrics
    out_path = "network_metrics_summary.csv"
    df_metrics.to_csv(out_path)
    print(f"\n📁 Node-level metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
