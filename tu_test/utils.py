import os
import sys
import torch
import torch.utils.data as utils
import numpy as np
import logging
from math import ceil
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import OneHotEncoder,normalize
import networkx as nx

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

def neighborhood(G, node, n):
    paths = nx.single_source_shortest_path(G, node)
    return [node for node, traversed_nodes in paths.items()
            if len(traversed_nodes) == n+1]

def load_data(ds_name, use_node_labels=False,use_node_attri=False):    
    graph_indicator = np.loadtxt("datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    
    edges = np.loadtxt("datasets/%s/%s_A.txt"%(ds_name,ds_name), dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    
    if use_node_labels:
        x = np.loadtxt("datasets/%s/%s_node_labels.txt"%(ds_name,ds_name), dtype=np.int64).reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
    elif use_node_attri:
        x = np.loadtxt("datasets/%s/%s_node_attributes.txt"%(ds_name,ds_name), delimiter=',',dtype=np.float64)#.reshape(-1,1)
    else:
        x = A.sum(axis=1)
        
    adj = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        adj.append(A[idx:idx+graph_size[i],idx:idx+graph_size[i]])
        features.append(x[idx:idx+graph_size[i],:])
        idx += graph_size[i]

    class_labels = np.loadtxt("datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), dtype=np.int64)
    return adj, features, class_labels

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_to_shortest_path(adj_lst):
    """Convert adjacency matrices to normalized shortest path matrices.
    
    Args:
        adj_lst: List of sparse adjacency matrices
        
    Returns:
        List of normalized shortest path matrices (torch.Tensor)
    """
    path_matrices = []
    for adj in adj_lst:
        # Convert sparse adj to dense numpy array
        if isinstance(adj, torch.Tensor):
            adj_dense = adj.to_dense().cpu().numpy()
        else:
            adj_dense = adj.toarray()
            
        # Create networkx graph
        G = nx.from_numpy_array(adj_dense)
        
        # Get shortest path lengths between all pairs
        n = adj_dense.shape[0]
        path_matrix = np.zeros((n, n))
        
        # Handle disconnected components
        for source in range(n):
            # Get shortest paths from source to all nodes
            try:
                lengths = nx.single_source_shortest_path_length(G, source)
                for target, length in lengths.items():
                    path_matrix[source, target] = length
            except nx.NetworkXError:
                # Node is isolated
                continue
                
        # Set unreachable paths to large value
        mask = path_matrix == 0
        np.fill_diagonal(mask, False)  # Don't modify diagonal zeros
        if mask.any():
            path_matrix[mask] = 1e6
              
        path_matrices.append(torch.FloatTensor(path_matrix))
        
    return path_matrices

def generate_batches(adj, features, y, batch_size, device, shuffle=False, use_shortest_path=False):
    """Generate batches with option to use shortest path matrices instead of adjacency.
    
    Args:
        adj: List of adjacency matrices
        features: List of node feature matrices  
        y: List of graph labels
        batch_size: Batch size
        device: torch.device
        shuffle: Whether to shuffle the data
        use_shortest_path: Whether to convert adj to shortest path matrices
    """
    N = len(y)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    n_batches = ceil(N/batch_size)

    # Convert to shortest path matrices if requested
    if use_shortest_path:
        adj = adj_to_shortest_path(adj)

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()
    y_lst = list() 
    
    for i in range(0, N, batch_size):
        n_graphs = min(i+batch_size, N) - i
        n_nodes = sum([adj[index[j]].shape[0] for j in range(i, min(i+batch_size, N))])

        if use_shortest_path:
            # Initialize with large value to prevent cross-graph leakage in SPD
            adj_batch = torch.full((n_nodes, n_nodes), 1e6)
        else:
            adj_batch = lil_matrix((n_nodes, n_nodes))
            
        features_batch = np.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)

        idx = 0
        for j in range(i, min(i+batch_size, N)):
            n = adj[index[j]].shape[0]
            if use_shortest_path:
                adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]
            else:
                adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]
                
            features_batch[idx:idx+n,:] = features[index[j]]
            graph_indicator_batch[idx:idx+n] = j-i
            y_batch[j-i] = y[index[j]]
            idx += n
                  
        if use_shortest_path:
            adj_lst.append(adj_batch.to(device))
        else:
            adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
            
        features_lst.append(torch.FloatTensor(features_batch).to(device))
        graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
        y_lst.append(torch.LongTensor(y_batch).to(device))

    return adj_lst, features_lst, graph_indicator_lst, y_lst

def generate_sub_features_idx(adj_batch, features_batch, size_subgraph=10, k_neighbor=1):
    """
    Optimized generation of subgraph features and indices using tensor operations.
    Assumes adj_batch contains adjacency or distance matrices.
    """
    sub_features_idx_list = []
    sub_adj_list = []
    
    for i in range(len(adj_batch)):
        adj = adj_batch[i]
        # Ensure dense for processing
        if adj.is_sparse:
            adj = adj.to_dense()
            
        N = adj.shape[0]
        device = adj.device
        
        # Prepare for neighborhood selection
        dist_matrix = adj.clone()
        
        # Mask out values > k_neighbor
        too_far_mask = dist_matrix > k_neighbor
        
        # Set too far to Infinity for sorting
        sort_scores = dist_matrix.clone()
        sort_scores[too_far_mask] = float('inf')
        
        # Sort to find closest neighbors
        sorted_vals, sorted_indices = torch.sort(sort_scores, dim=1)
        
        # Select top size_subgraph
        effective_k = min(N, size_subgraph)
        final_indices = sorted_indices[:, :effective_k]
        final_vals = sorted_vals[:, :effective_k]
        
        # Check validity (not inf)
        valid_mask = final_vals != float('inf')
        
        # Prepare output padded with N
        out_indices = torch.full((N, size_subgraph), N, device=device, dtype=torch.long)
        
        # Fill valid indices
        final_indices_padded = final_indices.clone()
        final_indices_padded[~valid_mask] = N
        
        out_indices[:, :effective_k] = final_indices_padded
        sub_features_idx_list.append(out_indices)
        
        # Extract Sub-Adjacency
        # Pad with large value (1e6) so padded nodes appear distant
        adj_padded = torch.nn.functional.pad(adj, (0, 1, 0, 1), value=1e6)
        row_idx = out_indices.unsqueeze(2).expand(-1, -1, size_subgraph)
        col_idx = out_indices.unsqueeze(1).expand(-1, size_subgraph, -1)
        sub_adj = adj_padded[row_idx, col_idx]
        sub_adj_list.append(sub_adj)

    return sub_adj_list, sub_features_idx_list