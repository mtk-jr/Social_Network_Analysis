import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Set, Dict


def plot_static_failure(G: nx.DiGraph, failed: Set[str], title: str = 'Network Failure'):
    und = G.to_undirected()
    pos = nx.spring_layout(und, seed=42)
    colors = ['red' if n in failed else 'green' for n in und.nodes()]
    plt.figure(figsize=(10, 8))
    nx.draw(und, pos, with_labels=True, node_color=colors, node_size=400)
    plt.title(title)
    plt.show()


def plot_plotly_network(G: nx.DiGraph, failed: Set[str] = None, title: str = 'Interactive Network') -> go.Figure:
    und = G.to_undirected()
    pos = nx.spring_layout(und, seed=42)

    
    edge_x = []
    edge_y = []
    for edge in und.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

   
    node_x = []
    node_y = []
    for node in und.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_color = []
    for node in und.nodes():
        if failed and node in failed:
            node_color.append('red')
        else:
            node_color.append('green')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(n) for n in und.nodes()],
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=15,
            line=dict(width=2, color='black')
        )
    )

   
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        titlefont_size=20,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    return fig



if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edges_from([
        ('Bank A', 'Bank B'),
        ('Bank B', 'Bank C'),
        ('Bank C', 'Bank D'),
        ('Bank D', 'Bank A'),
        ('Bank E', 'Bank B')
    ])
    failed_banks = {'Bank C'}
    fig = plot_plotly_network(G, failed=failed_banks, title='Bank Contagion Network')
    fig.show()
