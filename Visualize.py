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

# =====================================================================
# 1. Load Empirical Data (Training Set)
# =====================================================================
print("Loading empirical training data...")
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Convert PyG training graphs to NetworkX
train_graphs_nx = [to_networkx(data, to_undirected=True) for data in train_dataset]

# =====================================================================
# 2. Load Generated Data (Pickles)
# =====================================================================
print("Loading generated graphs from disk...")
with open('data/baseline_graphs.pkl', 'rb') as f:
    baseline_graphs = pickle.load(f)

with open('data/deep_graphs.pkl', 'rb') as f:
    deep_graphs = pickle.load(f)

print(f"Loaded {len(baseline_graphs)} Baseline graphs and {len(deep_graphs)} Deep Generative graphs.")

# =====================================================================
# 3. Evaluate Novelty and Uniqueness (Step 5)
# =====================================================================
print("\nEvaluating Novelty and Uniqueness...")

# Compute Weisfeiler-Lehman graph hashes for the training set[cite: 1]
train_hashes = set(nx.weisfeiler_lehman_graph_hash(g) for g in train_graphs_nx)

def evaluate_metrics(generated_graphs_list, train_hashes_set):
    """Calculates Novel, Unique, and Novel+Unique percentages."""
    num_generated = len(generated_graphs_list)
    gen_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in generated_graphs_list]
    gen_hash_counts = Counter(gen_hashes)

    # 1. Novel: Generated hashes that do NOT appear in the training set[cite: 1]
    novel_graphs = [h for h in gen_hashes if h not in train_hashes_set]

    # 2. Unique: Generated hashes that appear exactly once in the generated set[cite: 1]
    unique_graphs = [h for h in gen_hashes if gen_hash_counts[h] == 1]

    # 3. Novel and Unique: Intersection of both conditions[cite: 1]
    novel_and_unique = [h for h in novel_graphs if gen_hash_counts[h] == 1]

    pct_novel = (len(novel_graphs) / num_generated) * 100
    pct_unique = (len(unique_graphs) / num_generated) * 100
    pct_n_u = (len(novel_and_unique) / num_generated) * 100
    
    return pct_novel, pct_unique, pct_n_u

base_nov, base_uni, base_nu = evaluate_metrics(baseline_graphs, train_hashes)
deep_nov, deep_uni, deep_nu = evaluate_metrics(deep_graphs, train_hashes)

# Print as a formatted table to match the project description requirements[cite: 1]
print("\n" + "="*60)
print(f"{'Model':<25} | {'Novel':<8} | {'Unique':<8} | {'Novel+Unique':<12}")
print("-" * 60)
print(f"{'Baseline (Erdös-Rényi)':<25} | {base_nov:>6.2f}% | {base_uni:>6.2f}% | {base_nu:>11.2f}%")
print(f"{'Deep Generative Model':<25} | {deep_nov:>6.2f}% | {deep_uni:>6.2f}% | {deep_nu:>11.2f}%")
print("="*60 + "\n")


# =====================================================================
# 4. Compute Graph Statistics (Step 6)
# =====================================================================
print("Computing Graph Statistics...")

def compute_stats(graph_list):
    """Helper function to extract all node-level stats from a list of graphs."""
    degrees = []
    clusterings = []
    eigenvectors = []
    
    for g in graph_list:
        degrees.extend([d for n, d in g.degree()])
        clusterings.extend(list(nx.clustering(g).values()))
        try:
            evc = nx.eigenvector_centrality(g, max_iter=1000, tol=1e-03)
            eigenvectors.extend(list(evc.values()))
        except nx.PowerIterationFailedConvergence:
            pass # Skip disconnected graphs that fail to converge
            
    return degrees, clusterings, eigenvectors

# Compute stats for all three datasets
emp_deg, emp_clust, emp_eig = compute_stats(train_graphs_nx)
base_deg, base_clust, base_eig = compute_stats(baseline_graphs)
deep_deg, deep_clust, deep_eig = compute_stats(deep_graphs)

print(f"Baseline mean degree: {np.mean(base_deg):.3f}")
print(f"Deep mean degree: {np.mean(deep_deg):.3f}")


# =====================================================================
# 5. Plotting the 3x3 Grid
# =====================================================================
print("Generating 3x3 Grid Plot...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Graph Statistics Comparison', fontsize=16)

def plot_hist_column(ax_col, data_emp, data_base, data_deep, title):
    """Plots a single metric down a column for all three models using shared bins."""
    # Compute shared bins across ALL distributions to make visual comparison easy
    all_data = data_emp + data_base + data_deep
    bins = np.histogram_bin_edges(all_data, bins=30)
    
    # Plot Empirical (Row 0)
    axes[0, ax_col].hist(data_emp, bins=bins, color='blue', alpha=0.7, density=True)
    axes[0, ax_col].set_title(f'Empirical: {title}')
    
    # Plot Baseline (Row 1)
    axes[1, ax_col].hist(data_base, bins=bins, color='orange', alpha=0.7, density=True)
    axes[1, ax_col].set_title(f'Baseline: {title}')
    
    # Plot Deep Generative Model (Row 2)
    axes[2, ax_col].hist(data_deep, bins=bins, color='green', alpha=0.7, density=True)
    axes[2, ax_col].set_title(f'Deep Gen: {title}')

# Plot Columns: Degree, Clustering, Eigenvector[cite: 1]
plot_hist_column(0, emp_deg, base_deg, deep_deg, 'Node Degree')
plot_hist_column(1, emp_clust, base_clust, deep_clust, 'Clustering Coefficient')
plot_hist_column(2, emp_eig, base_eig, deep_eig, 'Eigenvector Centrality')

# h_pad adds vertical space between rows, w_pad adds horizontal space between columns
plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0, w_pad=1.5)
plt.savefig("Histograms.png")
