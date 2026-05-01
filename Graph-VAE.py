import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from collections import Counter
import pickle

print("Hello")

device = 'cuda'

# Load the MUTAG dataset
dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

# Extract the number of nodes (N) for every graph in the training dataset
train_node_counts = [graph.num_nodes for graph in train_dataset]

# Find the unique node counts and how many times each occurs
unique_n, counts = np.unique(train_node_counts, return_counts=True)

# Calculate the empirical probability of each node count
probabilities = counts / counts.sum()

print("Loaded data :)")

class GNNEncoder(torch.nn.Module):
    """GNN Encoder for node-level latents in a VAE"""

    def __init__(self, node_feature_dim, state_dim, latent_dim, num_message_passing_rounds, dropout_rate=0.0):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim  # NEW: Dimension of the VAE latent space
        self.num_message_passing_rounds = num_message_passing_rounds

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate) 
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate) 
            ) for _ in range(num_message_passing_rounds)])
        
        # Update network - GRU 
        self.update_net = torch.nn.ModuleList([
            torch.nn.GRUCell(self.state_dim, self.state_dim) 
            for _ in range(num_message_passing_rounds)
        ])

        # MODIFIED: Output networks for mu and logstd per node
        # Instead of outputting a single number for classification, 
        # it outputs the mean and log variance for the latent distribution.
        self.mu_net = torch.nn.Linear(self.state_dim, self.latent_dim)
        self.logstd_net = torch.nn.Linear(self.state_dim, self.latent_dim)

        
    def forward(self, x, edge_index):
        """
        Note: The 'batch' argument is removed because we no longer 
        pool nodes into a single graph representation.
        """
        num_nodes = x.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = state + self.update_net[r](aggregated, state)

        # MODIFIED: No more graph_state aggregation! 
        # We directly map the updated node states to latent parameters.
        mu = self.mu_net(state)
        logstd = self.logstd_net(state)
        
        return mu, logstd

class GraphVAE(torch.nn.Module):
    def __init__(self, encoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        
        # MLP Decoder to replace the simple dot product
        self.decoder_net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, 1)
        )

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(0.5 * logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        # z shape: (N, latent_dim)
        N = z.size(0)
        
        # Create all pairs (z_i, z_j)
        z_i = z.unsqueeze(1).expand(N, N, -1)
        z_j = z.unsqueeze(0).expand(N, N, -1)
        
        # Concatenate pairs
        z_cat = torch.cat([z_i, z_j], dim=-1) # Shape: (N, N, 2 * latent_dim)
        
        # Pass pairs through the MLP to get logits
        adj_logits = self.decoder_net(z_cat).squeeze(-1) # Shape: (N, N)
        
        # Symmetrize the logits since MUTAG graphs are undirected
        adj_logits = (adj_logits + adj_logits.t()) / 2.0
        
        # Squash to probabilities
        return adj_logits, torch.sigmoid(adj_logits)

    def forward(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)
        adj_logits, adj_prob = self.decode(z)
        return adj_logits, adj_prob, mu, logstd

def compute_elbo_loss(adj_logits, edge_index, mu, logstd, batch, kl_beta):
    num_nodes = batch.shape[0]
    true_adj = torch.zeros((num_nodes, num_nodes), device=adj_logits.device)
    true_adj[edge_index[0], edge_index[1]] = 1.0

    same_graph_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
    
    # ---------------------------------------------------------
    # 1. RECONSTRUCTION LOSS (Works fine flattened)
    # ---------------------------------------------------------
    pred_logits = adj_logits[same_graph_mask]
    true_labels = true_adj[same_graph_mask]

    num_possible = true_labels.numel()
    num_edges = true_labels.sum()
    pos_weight = (num_possible - num_edges) / (num_edges + 1e-6)

    recon_loss = F.binary_cross_entropy_with_logits(
        pred_logits, 
        true_labels,
        pos_weight=pos_weight,
        reduction='sum'
    )

    # ---------------------------------------------------------
    # 2. DEGREE LOSS (Must preserve 2D shape to sum across rows)
    # ---------------------------------------------------------
    # # Get full NxN probabilities
    # full_pred_probs = torch.sigmoid(adj_logits)
    
    # # Zero out edges that connect different graphs in the batch
    # valid_pred_probs = full_pred_probs * same_graph_mask.float()
    
    # # Sum along rows to get the expected degree per node
    # expected_degrees = valid_pred_probs.sum(dim=1)
    # true_degrees = (true_adj * same_graph_mask.float()).sum(dim=1)
    
    # # Calculate MSE penalty
    # degree_loss = F.mse_loss(expected_degrees, true_degrees, reduction='sum')

    # ---------------------------------------------------------
    # 3. KL DIVERGENCE
    # ---------------------------------------------------------
    kl_loss = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())

    # Combine everything
    # total_loss = (recon_loss + (0.2 * degree_loss) + kl_beta * kl_loss) 
    total_loss = (recon_loss + kl_beta * kl_loss) 
    
    return total_loss

# =====================================================================
# Training Setup and Loop
# =====================================================================

# Hyperparameters
latent_dim = 16
state_dim = 32
num_message_passing_rounds = 3
learning_rate = 1e-3
num_epochs = 2000

# Initialize Models
encoder = GNNEncoder(
    node_feature_dim=node_feature_dim, 
    state_dim=state_dim, 
    latent_dim=latent_dim, 
    num_message_passing_rounds=num_message_passing_rounds
).to(device)

model = GraphVAE(encoder, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("\nStarting VAE Training...")

model.train()
for epoch in range(1, num_epochs + 1):
    total_loss = 0

    # 1. KL Annealing: Slowly increase beta from 0.0 to 0.01 over the first 300 epochs
    # After epoch 300, it stays capped at 0.01
    current_beta = min(0.01, (epoch / 300) * 0.01)
    
    for batch_data in train_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        # Forward pass (catch 4 variables now)
        adj_logits, adj_prob, mu, logstd = model(batch_data.x, batch_data.edge_index)

        # Compute ELBO (pass adj_logits instead of adj_recon)
        loss = compute_elbo_loss(adj_logits, batch_data.edge_index, mu, logstd, batch_data.batch, kl_beta=current_beta)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Track total loss (multiply by num graphs in batch to undo the mean for accurate epoch averaging)
        num_graphs_in_batch = batch_data.batch.max().item() + 1
        total_loss += loss.item() * num_graphs_in_batch

    # Print average loss per graph for the epoch
    avg_loss = total_loss / len(train_dataset)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{num_epochs:03d} | Average ELBO Loss: {avg_loss:.4f}")

print("Training complete...")

# =====================================================================
# Generate 1000 Deep Graphs
# =====================================================================
print("\nGenerating 1000 graphs from the trained VAE...")

# Ensure model is in evaluation mode (disables dropout, etc.)
model.eval()

num_graphs_to_generate = 1000
deep_graphs_nx = []

with torch.no_grad(): # We don't need gradients for generation
    for _ in range(num_graphs_to_generate):
        # 1. Sample N from the empirical distribution
        n = np.random.choice(unique_n, p=probabilities)
        
        # 2. Sample Z directly from a standard normal distribution N(0, 1)
        # We bypass the encoder entirely because the VAE has learned to map this 
        # standard distribution to meaningful graph structures!
        z = torch.randn((n, latent_dim)).to(device)

        # 3. Decode to get the NxN matrix of edge probabilities
        # Use an underscore to ignore the raw logits returned by decode
        _, adj_prob = model.decode(z)
        
        # # Force the graph to have roughly the right number of edges.
        # # MUTAG graphs have an average edge density of about ~20-25%.
        # # Let's take the top 25% most probable edges, ensuring the strongest connections survive.
        
        # N = adj_prob.shape[0]
        # num_edges_to_keep = int((N * N) * 0.15) # Adjust this 0.25 up or down!
        
        # # Flatten, find the threshold value for the top K edges, and apply it
        # flat_probs = adj_prob.view(-1)
        # threshold = torch.topk(flat_probs, num_edges_to_keep).values[-1]
        
        # adj_discrete = (adj_prob >= threshold).float()

        # # 4. Convert probabilities to discrete edges
        adj_discrete = torch.bernoulli(adj_prob)
        
        # 5. Clean up the Adjacency Matrix
        # MUTAG graphs are undirected and don't have self-loops.
        # We extract the upper triangle (ignoring the diagonal) and mirror it.
        adj_discrete = torch.triu(adj_discrete, diagonal=1)
        adj_discrete = adj_discrete + adj_discrete.T
        
        # 6. Convert the PyTorch tensor to a NetworkX graph object
        G = nx.from_numpy_array(adj_discrete.cpu().numpy())
        
        # Only keep the Largest Connected Component (LCC)
        # if len(G.nodes) > 0:
        #     largest_cc = max(nx.connected_components(G), key=len)
        #     G = G.subgraph(largest_cc).copy()

        # Only append if the graph didn't completely collapse
        if G.number_of_nodes() > 2:
            deep_graphs_nx.append(G)

print(f"Successfully generated {len(deep_graphs_nx)} deep generative graphs! :)")

# =====================================================================
# Save the generated graphs
# =====================================================================
output_file = 'data/deep_graphs.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(deep_graphs_nx, f)

print(f"Saved generated graphs to {output_file}")