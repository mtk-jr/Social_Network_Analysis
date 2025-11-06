import copy
import random
import networkx as nx
from typing import List, Dict, Any


def simulate_contagion(G, initial_failures, threshold=0.4, p_fail=None):
    
    G = copy.deepcopy(G)
    failed = set(initial_failures)
    new_failures = set(initial_failures)

    while new_failures:
        next_failures = set()
        for node in G.nodes():
            if node not in failed:
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue

                failed_neighbors = sum(1 for n in neighbors if n in failed)

                
                if p_fail is None:
                    frac_failed = failed_neighbors / len(neighbors)
                    if frac_failed >= threshold:
                        next_failures.add(node)

                
                else:
                    if any(n in failed and random.random() < p_fail for n in neighbors):
                        next_failures.add(node)

        new_failures = next_failures
        failed |= next_failures

    return failed


def balance_sheet_cascade(G: nx.DiGraph, initial_failures: List[str]) -> Dict[str, Any]:
    
    G = copy.deepcopy(G)
    failed = set(initial_failures)
    history = [set(failed)]
    losses = {node: 0.0 for node in G.nodes()}
    new_failed = set(failed)

    while new_failed:
        next_failed = set()
        for b in new_failed:
            for creditor in G.predecessors(b):
                exposure = G[creditor][b].get('exposure', 0.0)
                losses[creditor] += exposure

                capital = G.nodes[creditor].get('capital', 0.0)
                if losses[creditor] >= capital and creditor not in failed:
                    next_failed.add(creditor)

        new_failed = next_failed - failed
        if not new_failed:
            break

        failed.update(new_failed)
        history.append(set(new_failed))

    return {"failed_nodes": failed, "history": history, "losses": losses}
