import pandas as pd

# Load the node metrics CSV
df = pd.read_csv("network_metrics_summary.csv", index_col=0)

print("\nTop 10 banks by Degree Centrality:")
print(df.sort_values('degree_centrality', ascending=False).head(10))

print("\nTop 10 banks by Betweenness Centrality:")
print(df.sort_values('betweenness', ascending=False).head(10))

print("\nTop 10 banks by Eigenvector Centrality:")
print(df.sort_values('eigenvector', ascending=False).head(10))

print("\nTop 10 banks by Pagerank:")
print(df.sort_values('pagerank', ascending=False).head(10))
