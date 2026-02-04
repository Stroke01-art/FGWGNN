import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from GW_utils import *
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Try to import PyTorch Geometric for baseline GNN
try:
    from torch_geometric.nn import GraphConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. Baseline GNN will not work.")

# MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,dropout_rate=0.5):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True # default is linear model
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = nn.Dropout(p=self.dropout_rate)(F.relu((self.linears[layer](h))))
            return self.linears[self.num_layers - 1](h)

class RW_layer(nn.Module):  
    def __init__(self, input_dim, out_dim, hidden_dim = None, max_step = 1, size_subgraph = 10, size_graph_filter = 10, dropout = 0.5):
        super(RW_layer, self).__init__()
        self.max_step = max_step
        self.size_graph_filter = size_graph_filter
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.fc_in = torch.nn.Linear(input_dim, hidden_dim)
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, hidden_dim, out_dim))
        else:
            self.features_hidden = Parameter(torch.FloatTensor(size_graph_filter, input_dim, out_dim))
        self.adj_hidden = Parameter(torch.FloatTensor( (size_graph_filter*(size_graph_filter-1))//2 , out_dim))
        self.bn = nn.BatchNorm1d(out_dim)
  
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        
    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
    
    def init_from_data(self, train_adj, train_features, train_labels, device):
        """
        Data-driven initialization: sample prototypes from training data.
        Each prototype is initialized from a real graph in the training set.
        
        Args:
            train_adj: List of adjacency matrices
            train_features: List of feature matrices
            train_labels: Tensor of labels
            device: torch device
        """
        n_prototypes = self.size_graph_filter
        # Convert to tensor if needed
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels)
        # Convert to tensor if needed
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels)
        n_classes = len(torch.unique(train_labels))
        
        # Sample prototypes uniformly from each class
        prototypes_per_class = n_prototypes // n_classes
        remaining = n_prototypes % n_classes
        
        prototype_idx = 0
        for class_id in range(n_classes):
            # Get indices of this class
            class_indices = (train_labels == class_id).nonzero(as_tuple=True)[0]
            
            # Number of prototypes for this class
            n_samples = prototypes_per_class + (1 if class_id < remaining else 0)
            
            # Randomly sample from this class
            if len(class_indices) < n_samples:
                # If not enough samples, sample with replacement
                sampled_idx = class_indices[torch.randint(0, len(class_indices), (n_samples,))]
            else:
                # Sample without replacement
                perm = torch.randperm(len(class_indices))[:n_samples]
                sampled_idx = class_indices[perm]
            
            # Initialize prototypes from sampled graphs
            for idx in sampled_idx:
                # Convert sparse matrix to dense tensor if needed
                if hasattr(train_adj[idx], 'toarray'):
                    graph_adj = torch.from_numpy(train_adj[idx].toarray()).float().to(device)
                elif isinstance(train_adj[idx], torch.Tensor):
                    graph_adj = train_adj[idx].to(device)
                else:
                    graph_adj = torch.tensor(train_adj[idx], dtype=torch.float32).to(device)
                
                if isinstance(train_features[idx], torch.Tensor):
                    graph_features = train_features[idx].to(device)
                else:
                    graph_features = torch.tensor(train_features[idx], dtype=torch.float32).to(device)
                
                # Extract upper triangle of adjacency matrix
                n_nodes = graph_adj.shape[0]
                if n_nodes >= self.size_graph_filter:
                    # If graph is larger, take first N nodes
                    sub_adj = graph_adj[:self.size_graph_filter, :self.size_graph_filter]
                else:
                    # If graph is smaller, pad with zeros
                    sub_adj = torch.zeros(self.size_graph_filter, self.size_graph_filter, device=device)
                    sub_adj[:n_nodes, :n_nodes] = graph_adj
                
                # Extract upper triangle indices
                idx_i, idx_j = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
                adj_values = sub_adj[idx_i, idx_j]
                
                # Initialize each output dimension with the same structure
                # Add small noise for diversity
                for out_dim in range(self.out_dim):
                    noise = torch.randn_like(adj_values) * 0.01
                    self.adj_hidden.data[:, out_dim] = adj_values + noise
                
                # Initialize features
                if n_nodes >= self.size_graph_filter:
                    sub_features = graph_features[:self.size_graph_filter]
                else:
                    feature_dim = graph_features.shape[1]
                    sub_features = torch.zeros(self.size_graph_filter, feature_dim, device=device)
                    sub_features[:n_nodes] = graph_features
                
                # Initialize feature parameters
                if self.hidden_dim:
                    # If using hidden layer, initialize with Xavier + small noise from data
                    nn.init.xavier_normal_(self.features_hidden.data[prototype_idx], gain=0.01)
                    # Add influence from real features (small)
                    if sub_features.shape[1] >= self.hidden_dim:
                        feature_influence = sub_features[:, :self.hidden_dim].mean(dim=0)  # Shape: (hidden_dim,)
                        # Broadcast across output dimensions
                        for out_d in range(self.out_dim):
                            self.features_hidden.data[prototype_idx, :, out_d] += 0.1 * feature_influence
                else:
                    # Direct feature initialization
                    if sub_features.shape[1] == self.input_dim:
                        for out_dim in range(self.out_dim):
                            noise = torch.randn_like(sub_features) * 0.01
                            self.features_hidden.data[prototype_idx, :, out_dim] = sub_features.mean(dim=0) + noise.mean(dim=0)
                    else:
                        # Fallback: Xavier initialization
                        nn.init.xavier_normal_(self.features_hidden.data[prototype_idx], gain=0.01)
                
                prototype_idx += 1
                if prototype_idx >= n_prototypes:
                    break
            
            if prototype_idx >= n_prototypes:
                break
        
        print(f"RW_layer: Initialized {prototype_idx} prototypes from training data (balanced across {n_classes} classes)")

    
    def init_from_data(self, train_adj, train_features, train_labels, device):
        """
        Data-driven initialization: sample prototypes from training data.
        Each prototype is initialized from a real graph in the training set.
        
        Args:
            train_adj: List of adjacency matrices
            train_features: List of feature matrices
            train_labels: Tensor of labels
            device: torch device
        """
        n_prototypes = self.size_graph_filter
        # Convert to tensor if needed
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels)
        # Convert to tensor if needed
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels)
        n_classes = len(torch.unique(train_labels))
        
        # Sample prototypes uniformly from each class
        prototypes_per_class = n_prototypes // n_classes
        remaining = n_prototypes % n_classes
        
        prototype_idx = 0
        for class_id in range(n_classes):
            # Get indices of this class
            class_indices = (train_labels == class_id).nonzero(as_tuple=True)[0]
            
            # Number of prototypes for this class
            n_samples = prototypes_per_class + (1 if class_id < remaining else 0)
            
            # Randomly sample from this class
            if len(class_indices) < n_samples:
                # If not enough samples, sample with replacement
                sampled_idx = class_indices[torch.randint(0, len(class_indices), (n_samples,))]
            else:
                # Sample without replacement
                perm = torch.randperm(len(class_indices))[:n_samples]
                sampled_idx = class_indices[perm]
            
            # Initialize prototypes from sampled graphs
            for idx in sampled_idx:
                # Convert sparse matrix to dense tensor if needed
                if hasattr(train_adj[idx], 'toarray'):
                    graph_adj = torch.from_numpy(train_adj[idx].toarray()).float().to(device)
                elif isinstance(train_adj[idx], torch.Tensor):
                    graph_adj = train_adj[idx].to(device)
                else:
                    graph_adj = torch.tensor(train_adj[idx], dtype=torch.float32).to(device)
                
                if isinstance(train_features[idx], torch.Tensor):
                    graph_features = train_features[idx].to(device)
                else:
                    graph_features = torch.tensor(train_features[idx], dtype=torch.float32).to(device)
                
                # Extract upper triangle of adjacency matrix
                n_nodes = graph_adj.shape[0]
                if n_nodes >= self.size_graph_filter:
                    # If graph is larger, take first N nodes
                    sub_adj = graph_adj[:self.size_graph_filter, :self.size_graph_filter]
                else:
                    # If graph is smaller, pad with zeros
                    sub_adj = torch.zeros(self.size_graph_filter, self.size_graph_filter, device=device)
                    sub_adj[:n_nodes, :n_nodes] = graph_adj
                
                # Extract upper triangle indices
                idx_i, idx_j = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
                adj_values = sub_adj[idx_i, idx_j]
                
                # Initialize each output dimension with the same structure
                # Add small noise for diversity
                for out_dim in range(self.out_dim):
                    noise = torch.randn_like(adj_values) * 0.01
                    self.adj_hidden.data[:, out_dim] = adj_values + noise
                
                # Initialize features
                if n_nodes >= self.size_graph_filter:
                    sub_features = graph_features[:self.size_graph_filter]
                else:
                    feature_dim = graph_features.shape[1]
                    sub_features = torch.zeros(self.size_graph_filter, feature_dim, device=device)
                    sub_features[:n_nodes] = graph_features
                
                # Initialize feature parameters
                if self.hidden_dim:
                    # If using hidden layer, initialize with Xavier + small noise from data
                    nn.init.xavier_normal_(self.features_hidden.data[prototype_idx], gain=0.01)
                    # Add influence from real features (small)
                    if sub_features.shape[1] >= self.hidden_dim:
                        feature_influence = sub_features[:, :self.hidden_dim].mean(dim=0)  # Shape: (hidden_dim,)
                        # Broadcast across output dimensions
                        for out_d in range(self.out_dim):
                            self.features_hidden.data[prototype_idx, :, out_d] += 0.1 * feature_influence
                else:
                    # Direct feature initialization
                    if sub_features.shape[1] == self.input_dim:
                        for out_dim in range(self.out_dim):
                            noise = torch.randn_like(sub_features) * 0.01
                            self.features_hidden.data[prototype_idx, :, out_dim] = sub_features.mean(dim=0) + noise.mean(dim=0)
                    else:
                        # Fallback: Xavier initialization
                        nn.init.xavier_normal_(self.features_hidden.data[prototype_idx], gain=0.01)
                
                prototype_idx += 1
                if prototype_idx >= n_prototypes:
                    break
            
            if prototype_idx >= n_prototypes:
                break
        
        print(f"RW_layer: Initialized {prototype_idx} prototypes from training data (balanced across {n_classes} classes)")


    def forward(self, adj, features, idxs, device):
        adj_hidden_norm = torch.zeros( self.size_graph_filter, self.size_graph_filter, self.out_dim).to(device)
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
        adj_hidden_norm[idx[0], idx[1], :] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        
        #construct feature array for each subgraph
        x = features
        if self.hidden_dim:
            x = nn.ReLU()(self.fc_in(x)) # (#G, D_hid)
        x = torch.cat([x,torch.zeros(1, x.shape[1]).to(device)])
        x = x[idxs] # (#G, #Nodes_sub, D_hid)
        
        #construct feature array for each graph filter
        z = self.features_hidden # (N_filter, Dhid, Dout)

        zx = torch.einsum("mcn,abc->ambn", (z, x)) # (#G, #Nodes_filter, #Nodes_sub, D_out)
        out = []
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_graph_filter, device=device)             
                o = torch.einsum("ab,bcd->acd", (eye, z))
                t = torch.einsum("mcn,abc->ambn", (o, x))
            else:
                x = torch.einsum("abc,acd->abd",(adj, x))
                z = torch.einsum("abd,bcd->acd", (adj_hidden_norm, z)) # adj_hidden_norm: (Nhid,Nhid,Dout)
                t = torch.einsum("mcn,abc->ambn", (z, x))
            t = self.dropout(t) 
            t = torch.mul(zx, t) # (#G, #Nodes_filter, #Nodes_sub, D_out)
            t = torch.mean(t, dim=[1,2])
            out.append(t)
        out = sum(out)/len(out)
        return out

# Assume 'device' is defined globally or passed in
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGW_Layer(nn.Module):
    def __init__(self, input_dim, out_dim, size_subgraph=10, size_graph_filter=10, hidden_dim=None, dropout=0.5, use_local_alpha=True, alpha_hidden_dim=32):
        super(FGW_Layer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.size_graph_filter = size_graph_filter
        self.hidden_dim = hidden_dim
        
        # Learnable adj and feature matrix of graph filters
        self.C_filter_uptriangle = Parameter(torch.FloatTensor((size_graph_filter * (size_graph_filter - 1)) // 2, out_dim))
        if hidden_dim:
            self.fc_in = nn.Linear(input_dim, hidden_dim)
            self.F_filter = Parameter(torch.FloatTensor(size_graph_filter, hidden_dim, out_dim))
        else:
            self.F_filter = Parameter(torch.FloatTensor(size_graph_filter, input_dim, out_dim))
        
        # Learnable global alpha (for backward compatibility and ablations)
        self.alpha_raw = Parameter(torch.ones(1))
        
        # Local alpha configuration
        self.use_local_alpha = use_local_alpha  # Set to False to use only global alpha
        self.alpha_hidden_dim = alpha_hidden_dim
        
        # Determine input dimension for alpha MLPs
        alpha_in_dim = hidden_dim if hidden_dim is not None else input_dim
        
        # Local alpha MLP: generate single alpha value per subgraph
        # This alpha is broadcast to all prototypes
        if self.use_local_alpha:
            self.alpha_sub_mlp = nn.Sequential(
                nn.Linear(alpha_in_dim, alpha_hidden_dim),
                nn.ReLU(),
                nn.Linear(alpha_hidden_dim, 1),  # Single alpha per subgraph
            )

        # --- CORRECTED APPROACH: Learnable MLP to generate distributions ---
        # This small network learns a rule to assign probabilities,
        self.h_filter_logits = Parameter(torch.FloatTensor(size_graph_filter))

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Initializing the generators is standard practice for nn.Linear
        self.h_filter_logits.data.fill_(0.0) # Initialize to all ones before softmax
        
        self.C_filter_uptriangle.data.uniform_(-1, 1)
        self.F_filter.data.uniform_(0, 1)
        self.alpha_raw.data.fill_(0.0)  # Initialize alpha to 0.5 after sigmoid
        
        # Initialize local alpha MLP with Xavier
        if self.use_local_alpha:
            for module in self.alpha_sub_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
    def init_from_data(self, train_adj, train_features, train_labels, device):
        """
        Data-driven initialization: sample prototypes from training data.
        Similar to FGW-GNN's supervised sampling initialization.
        
        Args:
            train_adj: List of adjacency matrices
            train_features: List of feature matrices  
            train_labels: Tensor of labels
            device: torch device
        """
        n_prototypes = self.size_graph_filter
        # Convert to tensor if needed
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels)
        # Convert to tensor if needed
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels)
        n_classes = len(torch.unique(train_labels))
        
        # Sample prototypes uniformly from each class
        prototypes_per_class = n_prototypes // n_classes
        remaining = n_prototypes % n_classes
        
        prototype_idx = 0
        for class_id in range(n_classes):
            # Get indices of this class
            class_indices = (train_labels == class_id).nonzero(as_tuple=True)[0]
            
            # Number of prototypes for this class
            n_samples = prototypes_per_class + (1 if class_id < remaining else 0)
            
            # Randomly sample from this class
            if len(class_indices) < n_samples:
                sampled_idx = class_indices[torch.randint(0, len(class_indices), (n_samples,))]
            else:
                perm = torch.randperm(len(class_indices))[:n_samples]
                sampled_idx = class_indices[perm]
            
            # Initialize prototypes from sampled graphs
            for idx in sampled_idx:
                # Convert sparse matrix to dense tensor if needed
                if hasattr(train_adj[idx], 'toarray'):
                    graph_adj = torch.from_numpy(train_adj[idx].toarray()).float().to(device)
                elif isinstance(train_adj[idx], torch.Tensor):
                    graph_adj = train_adj[idx].to(device)
                else:
                    graph_adj = torch.tensor(train_adj[idx], dtype=torch.float32).to(device)
                
                if isinstance(train_features[idx], torch.Tensor):
                    graph_features = train_features[idx].to(device)
                else:
                    graph_features = torch.tensor(train_features[idx], dtype=torch.float32).to(device)
                
                # Extract structure for prototype
                n_nodes = graph_adj.shape[0]
                if n_nodes >= self.size_graph_filter:
                    sub_adj = graph_adj[:self.size_graph_filter, :self.size_graph_filter]
                else:
                    sub_adj = torch.zeros(self.size_graph_filter, self.size_graph_filter, device=device)
                    sub_adj[:n_nodes, :n_nodes] = graph_adj
                
                # Extract upper triangle for C_filter_uptriangle
                idx_i, idx_j = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
                adj_values = sub_adj[idx_i, idx_j]
                
                # Initialize structure parameters (replicate for each output dim)
                for out_dim in range(self.out_dim):
                    noise = torch.randn_like(adj_values) * 0.01
                    self.C_filter_uptriangle.data[:, out_dim] = adj_values + noise
                
                # Extract and initialize features
                if n_nodes >= self.size_graph_filter:
                    sub_features = graph_features[:self.size_graph_filter]
                else:
                    feature_dim = graph_features.shape[1]
                    sub_features = torch.zeros(self.size_graph_filter, feature_dim, device=device)
                    sub_features[:n_nodes] = graph_features
                
                # Initialize F_filter
                if self.hidden_dim:
                    # Initialize with Xavier + influence from real features
                    nn.init.xavier_normal_(self.F_filter.data[prototype_idx], gain=0.01)
                    if sub_features.shape[1] >= self.hidden_dim:
                        feature_influence = sub_features[:, :self.hidden_dim].mean(dim=0)  # Shape: (hidden_dim,)
                        # Broadcast across output dimensions
                        for out_d in range(self.out_dim):
                            self.F_filter.data[prototype_idx, :, out_d] += 0.1 * feature_influence
                else:
                    # Direct feature initialization from graph
                    if sub_features.shape[1] == self.input_dim:
                        for out_dim in range(self.out_dim):
                            # Use actual node features with small noise for diversity
                            self.F_filter.data[prototype_idx, :, out_dim] = sub_features.mean(dim=0)
                            self.F_filter.data[prototype_idx, :, out_dim] += torch.randn(self.input_dim, device=device) * 0.01
                    else:
                        nn.init.xavier_normal_(self.F_filter.data[prototype_idx], gain=0.01)
                
                prototype_idx += 1
                if prototype_idx >= n_prototypes:
                    break
            
            if prototype_idx >= n_prototypes:
                break
        
        print(f"FGW_Layer: Initialized {prototype_idx} prototypes from training data (balanced across {n_classes} classes)")


    def forward(self, adj, features, idxs, device):
        # Construct adj array for each filter
        C_filter = torch.zeros(self.size_graph_filter, self.size_graph_filter, self.out_dim).to(device)
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1, device=device)
        C_filter[idx[0], idx[1], :] = self.relu(self.C_filter_uptriangle)
        C_filter = C_filter + torch.transpose(C_filter, 0, 1)
        
        # Construct features for each subgraph
        x = features
        if self.hidden_dim:
            x = self.relu(self.fc_in(x))
        x = torch.cat([x, torch.zeros(1, x.shape[1]).to(device)])
        x = x[idxs]  # (batch, N_sub, D_hid)

        # Get dynamic shapes for this batch
        batch_size = x.shape[0]
        N_sub = x.shape[1]
        N_filter = self.size_graph_filter

        # --- DYNAMICALLY GENERATE DISTRIBUTIONS ---
        
        # Create normalized indices [0, 1] as input to the generator
        # node_indices_sub = torch.linspace(0, 1, N_sub, device=device).unsqueeze(-1)
        # node_indices_filter = torch.linspace(0, 1, N_filter, device=device).unsqueeze(-1)
        
        # Generate logits (raw scores) from the MLPs
        # h_input_logits = self.h_input_generator(node_indices_sub).squeeze(-1)
        # h_input = F.softmax(h_input_logits, dim=0)
        # h_input_batch = h_input.unsqueeze(0).expand(batch_size, -1)

        h_input_batch = torch.ones(batch_size, N_sub, device=device) / N_sub
        # h_filter_batch = h_filter_batch.unsqueeze(0).expand(batch_size, -1)  # (batch, N_filter)

        # Compute h_filter by applying softmax to the learnable parameter
        h_filter_batch = F.softmax(self.h_filter_logits, dim=0)
        h_filter_batch = h_filter_batch.unsqueeze(0).expand(batch_size, -1) # (batch, N_filter)

        # --- Compute local or global alpha ---
        if self.use_local_alpha:
            # Compute patch-level embeddings for local alpha generation
            # Subgraph embedding: mean pooling over nodes
            x_pooled = x.mean(dim=1)  # (batch_size, D_feat)
            
            # Generate single alpha value per subgraph (broadcast to all prototypes)
            sub_alpha_logits = self.alpha_sub_mlp(x_pooled)  # (batch_size, 1)
            sub_alpha = torch.sigmoid(sub_alpha_logits)  # (batch_size, 1), range [0, 1]
            
            # Reshape for broadcasting: (batch, 1) -> (batch, 1, 1) 
            # Will broadcast to (D_out, batch, N_sub, N_filter) automatically
            alpha_tensor = sub_alpha.unsqueeze(-1)  # (batch_size, 1, 1)
        else:
            # Use global scalar alpha (backward compatibility)
            alpha_tensor = torch.sigmoid(self.alpha_raw)

        # --- Compute FGW distance with correctly formed inputs ---
        fgw_dists = entropic_fused_gromov_wasserstein(
            C_filter=C_filter, F_filter=self.F_filter, h_filter=h_filter_batch,
            C_sub=adj, F_sub=x, h_sub=h_input_batch,
            alpha=alpha_tensor, 
            epsilon=0.05,
            gw_iter=1,
            sinkhorn_iter=2,
        )
        return fgw_dists

class KerGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, kernel='rw', max_step = 1, size_graph_filter=10, size_subgraph=10,
                 num_mlp_layers=1, mlp_hidden_dim=None, dropout_rate=0.5, no_norm=False, use_node_norm=False,
                 use_local_alpha=False, alpha_hidden_dim=32):
        super(KerGNN, self).__init__()
        self.no_norm = no_norm
        self.use_node_norm = use_node_norm  # use BatchNorm
        self.dropout_rate = dropout_rate
        self.use_local_alpha = use_local_alpha  # FGW local alpha setting
        self.alpha_hidden_dim = alpha_hidden_dim  # FGW alpha MLP hidden dim
        self.num_layers = len(hidden_dims) - 1
        self.ker_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        if use_node_norm:
            self.node_batch_norms = nn.ModuleList()
        
        for layer in range(self.num_layers):
            if layer == 0:
                if kernel == 'rw':
                    self.ker_layers.append(RW_layer(input_dim, hidden_dims[1], hidden_dim = hidden_dims[0], max_step = max_step, size_subgraph = size_subgraph, size_graph_filter = size_graph_filter[0], dropout = dropout_rate))
                elif kernel == 'fgw':
                    self.ker_layers.append(
                        FGW_Layer(input_dim, hidden_dims[1], hidden_dim = hidden_dims[0], 
                                 size_subgraph = size_subgraph, size_graph_filter = size_graph_filter[0], 
                                 dropout = dropout_rate, use_local_alpha = use_local_alpha, 
                                 alpha_hidden_dim = alpha_hidden_dim)
                    )
                else:
                    exit('Error: unrecognized model')
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[1]))
                
                # node-level BatchNorm
                if use_node_norm:
                    if hidden_dims[0]:  # if hidden_dim
                        self.node_batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
                    else:
                        self.node_batch_norms.append(nn.BatchNorm1d(input_dim))
            else:
                if kernel == 'rw':
                    self.ker_layers.append(RW_layer(hidden_dims[layer], hidden_dims[layer+1], hidden_dim = None, max_step = max_step, size_subgraph = size_subgraph, size_graph_filter = size_graph_filter[layer], dropout = dropout_rate))
                elif kernel == 'fgw':
                    self.ker_layers.append(
                        FGW_Layer(hidden_dims[layer], hidden_dims[layer+1], hidden_dim = None, 
                                 size_subgraph = size_subgraph, size_graph_filter = size_graph_filter[layer], 
                                 dropout = dropout_rate, use_local_alpha = use_local_alpha, 
                                 alpha_hidden_dim = alpha_hidden_dim)
                    )
                else:
                    exit('Error: unrecognized model')            
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[layer+1]))
                
                # node-level BatchNorm
                if use_node_norm:
                    self.node_batch_norms.append(nn.BatchNorm1d(hidden_dims[layer]))

        self.linears_prediction = nn.ModuleList()
        for layer in range(self.num_layers + 1):
            if layer == 0:
                self.linears_prediction.append(MLP(num_mlp_layers, input_dim, mlp_hidden_dim, output_dim))
            else:
                self.linears_prediction.append(MLP(num_mlp_layers, hidden_dims[layer], mlp_hidden_dim, output_dim))

    def forward(self, adj, features, idxs, graph_indicator):
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)

        hidden_rep = [features]
        h = features

        for layer in range(self.num_layers):
            # use node-level BatchNorm if specified
            if self.use_node_norm and layer > 0:
                # transform h to (num_nodes, feature_dim)
                h = self.node_batch_norms[layer-1](h)
            
            h = self.ker_layers[layer](adj, h, idxs, device)
            h = self.batch_norms[layer](h)  
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.zeros(n_graphs, h.shape[1], device=device).index_add_(0, graph_indicator, h)               
            
            if not self.no_norm:
                # Original normalization: divide by node count
                norm = counts.unsqueeze(1).repeat(1, pooled_h.shape[1])
                pooled_h = pooled_h/norm
            # else: no normalization (keep sum as is)
            
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.dropout_rate, training = self.training)

        return score_over_layer
    
# ============================================================================
# Baseline 1-layer GNN from paper (for comparison)
# ============================================================================

class SumTaskGNN(nn.Module):
    """
    Simple 1-layer GNN baseline from the paper.
    Used to verify that graph structure shouldn't help on sum task.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1,
                 dropout=0.0, bias=False, init_std=1e-2):
        super().__init__()
        self.init_std = init_std
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for SumTaskGNN")
        
        layers = []
        # First layer
        layers.append(GraphConv(in_channels, hidden_channels, bias=bias))
        # Remaining layers
        for _ in range(1, num_layers):
            layers.append(GraphConv(hidden_channels, hidden_channels, bias=bias))
        
        self.convs = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters as in the paper
        self.init_params()
    
    def init_params(self):
        """Initialize with Xavier normal, gain=1e-2 as in paper."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, data):
        """
        Forward pass using PyTorch Geometric Data object.
        
        Args:
            data: PyG Data with x, edge_index, batch
        Returns:
            logits: (batch_size, num_classes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
            x = self.dropout(x)
        
        x = self.pool(x, batch)
        x = self.readout(x)
        
        return x
