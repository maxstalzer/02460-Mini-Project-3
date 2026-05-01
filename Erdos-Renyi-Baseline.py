import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from collections import Counter
import pickle

device = 'cpu'

# Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

print("Loaded data :)")

# Extract the number of nodes (N) for every graph in the training dataset
train_node_counts = [graph.num_nodes for graph in train_dataset]

# Find the unique node counts and how many times each occurs
unique_n, counts = np.unique(train_node_counts, return_counts=True)

# Calculate the empirical probability of each node count
probabilities = counts / counts.sum()

# Create a dictionary to easily view the distribution (Optional, but helpful for debugging)
empirical_node_dist = dict(zip(unique_n, probabilities))

print("Empirical Node Distribution (N : Probability):")
for n, p in empirical_node_dist.items():
    print(f"{n} nodes : {p:.4f}")

# =====================================================================
# Compute Link Probabilities (r) per Graph Size
# =====================================================================

# Track total actual edges and total possible edges for each node count (N)
edges_for_N = {n: {'actual': 0, 'possible': 0} for n in unique_n}

for graph in train_dataset:
    n = graph.num_nodes
    
    # PyG double-counts edges for undirected graphs, so we divide by 2
    actual_edges = graph.num_edges // 2 
    
    # Max possible undirected edges for N nodes is N * (N - 1) / 2
    possible_edges = n * (n - 1) // 2   
    
    edges_for_N[n]['actual'] += actual_edges
    edges_for_N[n]['possible'] += possible_edges

# Calculate link probability (r) for each N
link_probabilities = {}
for n, counts in edges_for_N.items():
    if counts['possible'] > 0:
        # r = total edges divided by total possible edges for graphs of size N
        link_probabilities[n] = counts['actual'] / counts['possible']
    else:
        link_probabilities[n] = 0.0 # Edge case fallback

print("\nLink Probabilities (r) per Node Count (N):")
for n, r in link_probabilities.items():
    print(f"N={n}: r={r:.4f}")

# =====================================================================
# Generate 1000 Baseline Graphs
# =====================================================================

num_graphs_to_generate = 1000
generated_graphs = []

for _ in range(num_graphs_to_generate):
    # 1. Sample N from the empirical distribution calculated in Step 2
    sampled_n = np.random.choice(unique_n, p=probabilities)
    
    # 2. Retrieve the matching link probability r calculated in Step 3
    r = link_probabilities[sampled_n]
    
    # 3. Sample a random graph using the Erdös-Rényi model
    # nx.erdos_renyi_graph returns a NetworkX graph object
    gen_graph = nx.erdos_renyi_graph(n=sampled_n, p=r)
    
    generated_graphs.append(gen_graph)

print(f"\nSuccessfully generated {len(generated_graphs)} baseline graphs! :)")

# =====================================================================
# Save the generated graphs
# =====================================================================
output_file = 'data/baseline_graphs.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(generated_graphs, f)

print(f"Saved generated graphs to {output_file}")