"""
Utility functions for KerGNN Table 2 experiments.
Adapted from original FGW-KerGNN utils.py with Table 2 support.
"""
import os
import sys
import torch
import numpy as np
import logging
from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
import networkx as nx

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def accuracy(output, labels):
    """Compute classification accuracy."""
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
    """Create logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info('Log directory: %s', log_dir)
    
    return logger


def neighborhood(G, node, n):
    """Get n-hop neighborhood of a node."""
    paths = nx.single_source_shortest_path(G, node)
    return [node for node, traversed_nodes in paths.items()
            if len(traversed_nodes) == n+1]


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to pytorch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col))
    ).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def adj_to_shortest_path(adj_list, normalize=True):
    """
    Convert adjacency matrices to shortest path distance matrices.
    
    Args:
        adj_list: List of scipy sparse adjacency matrices
        normalize: If True, normalize to [0, 1] range
        
    Returns:
        List of shortest path distance matrices (scipy sparse)
    """
    from scipy.sparse.csgraph import shortest_path
    
    path_matrices = []
    for adj in adj_list:
        # Convert to dense if sparse
        if hasattr(adj, 'toarray'):
            adj_dense = adj.toarray()
        else:
            adj_dense = adj
            
        # Compute shortest path distances
        # directed=False for undirected graphs
        dist_matrix = shortest_path(adj_dense, directed=False, unweighted=True)
        
        # Handle infinite distances (disconnected components)
        # Replace inf with max_distance + 1
        max_finite = dist_matrix[np.isfinite(dist_matrix)].max() if np.any(np.isfinite(dist_matrix)) else 1.0
        dist_matrix[np.isinf(dist_matrix)] = max_finite + 1
        
        if normalize:
            # Normalize to [0, 1] range
            max_dist = dist_matrix.max()
            if max_dist > 0:
                dist_matrix = dist_matrix / max_dist
        
        # Convert back to sparse format
        path_matrices.append(csr_matrix(dist_matrix.astype(np.float32)))
    
    return path_matrices


def generate_batches_with_shortest_path(adj, features, y, batch_size, device, 
                                        shuffle=False, use_shortest_path=False, 
                                        normalize_sp=True):
    """
    Generate batches with optional shortest path conversion.
    
    Args:
        adj: List of adjacency matrices (scipy sparse)
        features: List of feature matrices (numpy arrays)
        y: List/array of labels
        batch_size: Batch size
        device: torch.device
        shuffle: Whether to shuffle data
        use_shortest_path: If True, convert adjacency to shortest path matrices
        normalize_sp: If True (and use_shortest_path=True), normalize to [0,1]
    
    Returns:
        adj_lst: List of batched adjacency/shortest-path tensors
        features_lst: List of batched feature tensors
        graph_indicator_lst: List of graph indicators
        y_lst: List of batched labels
    """
    # Convert to shortest path if requested
    if use_shortest_path:
        adj = adj_to_shortest_path(adj, normalize=normalize_sp)
    
    # Use standard batch generation
    return generate_batches(adj, features, y, batch_size, device, shuffle)



def adj_to_shortest_path(adj_list, normalize=True):
    """
    Convert adjacency matrices to shortest path distance matrices.
    
    Args:
        adj_list: List of scipy sparse adjacency matrices
        normalize: If True, normalize to [0, 1] range
        
    Returns:
        List of shortest path distance matrices (scipy sparse)
    """
    from scipy.sparse.csgraph import shortest_path
    
    path_matrices = []
    for adj in adj_list:
        # Convert to dense if sparse
        if hasattr(adj, 'toarray'):
            adj_dense = adj.toarray()
        else:
            adj_dense = adj
            
        # Compute shortest path distances
        # directed=False for undirected graphs
        dist_matrix = shortest_path(adj_dense, directed=False, unweighted=True)
        
        # Handle infinite distances (disconnected components)
        # Replace inf with max_distance + 1
        max_finite = dist_matrix[np.isfinite(dist_matrix)].max() if np.any(np.isfinite(dist_matrix)) else 1.0
        dist_matrix[np.isinf(dist_matrix)] = max_finite + 1
        
        if normalize:
            # Normalize to [0, 1] range
            max_dist = dist_matrix.max()
            if max_dist > 0:
                dist_matrix = dist_matrix / max_dist
        
        # Convert back to sparse format
        path_matrices.append(csr_matrix(dist_matrix.astype(np.float32)))
    
    return path_matrices


def generate_batches_with_shortest_path(adj, features, y, batch_size, device, 
                                        shuffle=False, use_shortest_path=False, 
                                        normalize_sp=True):
    """
    Generate batches with optional shortest path conversion.
    
    Args:
        adj: List of adjacency matrices (scipy sparse)
        features: List of feature matrices (numpy arrays)
        y: List/array of labels
        batch_size: Batch size
        device: torch.device
        shuffle: Whether to shuffle data
        use_shortest_path: If True, convert adjacency to shortest path matrices
        normalize_sp: If True (and use_shortest_path=True), normalize to [0,1]
    
    Returns:
        adj_lst: List of batched adjacency/shortest-path tensors
        features_lst: List of batched feature tensors
        graph_indicator_lst: List of graph indicators
        y_lst: List of batched labels
    """
    # Convert to shortest path if requested
    if use_shortest_path:
        adj = adj_to_shortest_path(adj, normalize=normalize_sp)
    
    # Use standard batch generation
    return generate_batches(adj, features, y, batch_size, device, shuffle)


def generate_batches(adj, features, y, batch_size, device, shuffle=False):
    """
    Generate batches for KerGNN training/testing.
    
    Args:
        adj: List of adjacency matrices (scipy sparse)
        features: List of feature matrices (numpy arrays)
        y: List/array of labels
        batch_size: Batch size
        device: torch.device
        shuffle: Whether to shuffle data
    
    Returns:
        adj_lst: List of batched adjacency tensors
        features_lst: List of batched feature tensors
        graph_indicator_lst: List of graph indicators
        y_lst: List of batched labels
    """
    N = len(y)
    
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    adj_lst = []
    features_lst = []
    graph_indicator_lst = []
    y_lst = []
    
    for i in range(0, N, batch_size):
        n_graphs = min(i + batch_size, N) - i
        n_nodes = sum([adj[index[j]].shape[0] 
                      for j in range(i, min(i + batch_size, N))])

        # Create batch adjacency matrix
        adj_batch = lil_matrix((n_nodes, n_nodes))
        features_batch = np.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)

        idx = 0
        for j in range(i, min(i + batch_size, N)):
            n = adj[index[j]].shape[0]
            adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]    
            features_batch[idx:idx+n, :] = features[index[j]]
            graph_indicator_batch[idx:idx+n] = j - i
            y_batch[j - i] = y[index[j]]
            idx += n
        
        # Convert to torch tensors
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
        features_lst.append(torch.FloatTensor(features_batch).to(device))
        graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
        y_lst.append(torch.LongTensor(y_batch).to(device))

    return adj_lst, features_lst, graph_indicator_lst, y_lst


def generate_sub_features_idx(adj_batch, features_batch, size_subgraph=10, k_neighbor=1):
    """
    Generate subgraph indices and adjacencies for each node.
    
    Args:
        adj_batch: List of adjacency tensors
        features_batch: List of feature tensors
        size_subgraph: Size of subgraph
        k_neighbor: k-hop neighborhood
    
    Returns:
        sub_adj_list: List of subgraph adjacency tensors
        sub_features_idx_list: List of subgraph node indices
    """
    sub_features_idx_list = []
    sub_adj_list = []
    
    for i in range(len(adj_batch)):
        adj = adj_batch[i]
        x = features_batch[i]
        num_B_nodes = x.shape[0]
        
        # Convert to networkx graph
        G = nx.from_numpy_array(adj.to_dense().cpu().numpy())
        
        x_sub_adj = torch.zeros(x.shape[0], size_subgraph, size_subgraph).to(device)
        x_sub_idx = torch.zeros(x.shape[0], size_subgraph).to(device)

        for node in range(x.shape[0]):
            # Determine k-hop neighbors
            tmp = []
            for k in range(k_neighbor + 1):
                tmp = tmp + neighborhood(G, node, k)
            
            if len(tmp) > size_subgraph:
                tmp = tmp[:size_subgraph]
            
            sub_idxs = tmp
            
            # Pad if necessary
            if len(tmp) < size_subgraph:
                padded_sub_idxs = tmp + [num_B_nodes] * (size_subgraph - len(tmp))
            else:
                padded_sub_idxs = tmp
            
            x_sub_idx[node] = torch.tensor(padded_sub_idxs)

            # Get subgraph adjacency
            G_sub = G.subgraph(sub_idxs)
            tmp = nx.to_numpy_array(G_sub)
            
            if tmp.shape[0] < size_subgraph:
                tmp_adj = np.zeros([size_subgraph, size_subgraph])
                tmp_adj[:tmp.shape[0], :tmp.shape[1]] = tmp
                tmp = tmp_adj
            
            x_sub_adj_ = torch.from_numpy(tmp).float().to(device)
            
            # Normalize if needed
            if 2 in x_sub_adj_:
                x_sub_adj_ = x_sub_adj_ / 2
            
            x_sub_adj[node] = x_sub_adj_

        sub_features_idx_list.append(x_sub_idx.long())
        sub_adj_list.append(x_sub_adj)

    return sub_adj_list, sub_features_idx_list




def generate_sub_features_idx_fast(adj_batch, features_batch, size_subgraph=10, k_neighbor=1):
    """
    优化版本：使用纯 PyTorch 操作，GPU 加速
    
    相比原版：
    - 不使用 networkx（避免 CPU-GPU 转换）
    - 批量计算 k-hop neighbors（使用邻接矩阵的幂）
    - 全部在 GPU 上计算
    
    预期加速：10-20x
    """
    sub_features_idx_list = []
    sub_adj_list = []
    
    for i in range(len(adj_batch)):
        adj = adj_batch[i]
        x = features_batch[i]
        num_nodes = x.shape[0]
        device = adj.device
        
        # 转换为 dense（如果是 sparse）
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        # 计算 k-hop 邻接矩阵
        # A^k 表示 k-hop 的连通性
        adj_k = adj_dense.clone()
        if k_neighbor > 0:
            adj_power = adj_dense.clone()
            for k in range(1, k_neighbor + 1):
                adj_power = torch.matmul(adj_power, adj_dense)
                adj_k = adj_k + adj_power
        
        # 二值化（有连接就是1）
        adj_k = (adj_k > 0).float()
        
        # 为每个节点选择 k-hop neighbors
        x_sub_idx = torch.zeros(num_nodes, size_subgraph, dtype=torch.long, device=device)
        x_sub_adj = torch.zeros(num_nodes, size_subgraph, size_subgraph, device=device)
        
        for node in range(num_nodes):
            # 获取该节点的 k-hop neighbors（包括自己）
            neighbors = torch.nonzero(adj_k[node], as_tuple=False).squeeze(-1)
            
            # 确保包含节点自己
            if node not in neighbors:
                neighbors = torch.cat([torch.tensor([node], device=device), neighbors])
            
            # 截断或填充到 size_subgraph
            n_neighbors = min(len(neighbors), size_subgraph)
            
            if n_neighbors > 0:
                # 取前 size_subgraph 个邻居
                sub_idxs = neighbors[:size_subgraph]
                
                # 填充索引
                x_sub_idx[node, :n_neighbors] = sub_idxs
                if n_neighbors < size_subgraph:
                    # 用 num_nodes 填充（padding index）
                    x_sub_idx[node, n_neighbors:] = num_nodes
                
                # 提取子图邻接矩阵
                # 使用 advanced indexing
                sub_adj = adj_dense[sub_idxs][:, sub_idxs]
                x_sub_adj[node, :n_neighbors, :n_neighbors] = sub_adj
            else:
                # 如果没有邻居，用 padding
                x_sub_idx[node, :] = num_nodes
        
        sub_features_idx_list.append(x_sub_idx)
        sub_adj_list.append(x_sub_adj)
    
    return sub_adj_list, sub_features_idx_list





def generate_sub_features_idx_vectorized(adj_batch, features_batch, size_subgraph=10, k_neighbor=1):
    """
    完全向量化版本：避免 Python 循环，使用纯 PyTorch 批量操作
    
    预期加速：20-50x
    """
    sub_features_idx_list = []
    sub_adj_list = []
    
    for i in range(len(adj_batch)):
        adj = adj_batch[i]
        x = features_batch[i]
        num_nodes = x.shape[0]
        device = adj.device
        
        # 转换为 dense
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        # 计算 k-hop 邻接矩阵（包括自己）
        adj_k = adj_dense.clone()
        # 添加自连接
        adj_k = adj_k + torch.eye(num_nodes, device=device)
        
        if k_neighbor > 0:
            adj_power = adj_dense.clone()
            for k in range(1, k_neighbor + 1):
                adj_power = torch.matmul(adj_power, adj_dense)
                adj_k = adj_k + adj_power
        
        # 二值化
        adj_k = (adj_k > 0).float()
        
        # 批量处理：为所有节点一次性找到邻居
        # 方法：对每一行排序，取前 size_subgraph 个
        
        # 创建邻居索引矩阵 (num_nodes, num_nodes)
        # 每行是该节点的邻居列表
        node_indices = torch.arange(num_nodes, device=device).unsqueeze(0).expand(num_nodes, -1)
        
        # 只保留有连接的节点索引，其他设为 num_nodes（padding）
        neighbor_indices = torch.where(adj_k > 0, node_indices, torch.full_like(node_indices, num_nodes))
        
        # 排序，使有效邻居在前面（因为 num_nodes 最大，会被排到后面）
        sorted_neighbors, _ = torch.sort(neighbor_indices, dim=1)
        
        # 取前 size_subgraph 个
        x_sub_idx = sorted_neighbors[:, :size_subgraph]
        
        # 批量提取子图邻接矩阵
        # 使用 advanced indexing: adj_dense[i, j] -> sub_adj[node, :, :]
        # 对每个节点 n，提取 adj_dense[sub_idx[n], :][:, sub_idx[n]]
        
        # 方法：创建一个 gather index
        batch_idx = torch.arange(num_nodes, device=device).unsqueeze(1).unsqueeze(2).expand(-1, size_subgraph, size_subgraph)
        row_idx = x_sub_idx.unsqueeze(2).expand(-1, -1, size_subgraph)  # (num_nodes, size_subgraph, size_subgraph)
        col_idx = x_sub_idx.unsqueeze(1).expand(-1, size_subgraph, -1)  # (num_nodes, size_subgraph, size_subgraph)
        
        # 处理 padding（num_nodes 索引）
        valid_row = (row_idx < num_nodes).float()
        valid_col = (col_idx < num_nodes).float()
        valid_mask = valid_row * valid_col
        
        # 将越界索引替换为 0（避免索引错误）
        row_idx_safe = torch.clamp(row_idx, 0, num_nodes - 1)
        col_idx_safe = torch.clamp(col_idx, 0, num_nodes - 1)
        
        # 提取子图邻接矩阵
        x_sub_adj = adj_dense[row_idx_safe, col_idx_safe]
        
        # 应用 mask（padding 位置设为 0）
        x_sub_adj = x_sub_adj * valid_mask
        
        sub_features_idx_list.append(x_sub_idx)
        sub_adj_list.append(x_sub_adj)
    
    return sub_adj_list, sub_features_idx_list



def prepare_table2_data(datasets, batch_size, device, use_shortest_path=False, normalize_sp=True):
    """
    Prepare Table 2 datasets for training/testing.
    
    Args:
        datasets: Dictionary from generate_table2_datasets()
        batch_size: Batch size
        device: torch.device
        use_shortest_path: If True, use shortest path distance matrices
        normalize_sp: If True, normalize shortest path to [0, 1]
    
    Returns:
        Dictionary with prepared batches for train and test sets
    """
    prepared = {}
    
    # Prepare training data
    print("Preparing training data...")
    train_adj, train_feat, train_indicator, train_y = generate_batches_with_shortest_path(
        datasets['train']['adj'],
        datasets['train']['features'],
        datasets['train']['labels'],
        batch_size,
        device,
        shuffle=True,
        use_shortest_path=use_shortest_path,
        normalize_sp=normalize_sp
    )
    
    prepared['train'] = {
        'adj': train_adj,
        'features': train_feat,
        'graph_indicator': train_indicator,
        'labels': train_y,
        'name': datasets['train']['name']
    }
    
    # Prepare test data
    prepared['test'] = {}
    for test_name, test_data in datasets['test'].items():
        print(f"Preparing test data: {test_name}...")
        test_adj, test_feat, test_indicator, test_y = generate_batches_with_shortest_path(
            test_data['adj'],
            test_data['features'],
            test_data['labels'],
            batch_size,
            device,
            shuffle=False,
            use_shortest_path=use_shortest_path,
            normalize_sp=normalize_sp
        )
        
        prepared['test'][test_name] = {
            'adj': test_adj,
            'features': test_feat,
            'graph_indicator': test_indicator,
            'labels': test_y,
            'name': test_name
        }
    
    return prepared


if __name__ == "__main__":
    # Quick test
    print("Testing utility functions...")
    
    # Test batch generation
    from data_generator import generate_sum_task_data
    
    adj_list, feat_list, labels, _ = generate_sum_task_data(
        num_samples=10,
        num_nodes=20,
        feature_dim=16,
        graph_type='regular',
        graph_param=5,
        seed=42
    )
    
    device = torch.device('cpu')
    adj_batch, feat_batch, indicator, y_batch = generate_batches(
        adj_list, feat_list, labels, batch_size=5, device=device
    )
    
    print(f"Generated {len(adj_batch)} batches")
    print(f"First batch: {adj_batch[0].shape}")
    print("Utility test successful!")


def convert_to_pyg_data(adj_list, features_list, labels):
    """
    Convert scipy sparse matrices to PyTorch Geometric Data objects.
    
    Args:
        adj_list: List of scipy sparse adjacency matrices
        features_list: List of scipy sparse feature matrices  
        labels: Tensor of labels
    
    Returns:
        List of PyG Data objects
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("PyTorch Geometric required for baseline GNN")
    
    import torch
    from scipy.sparse import coo_matrix
    
    data_list = []
    for i in range(len(adj_list)):
        # Convert adjacency matrix to edge_index
        adj_coo = coo_matrix(adj_list[i])
        edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
        
        # Convert features
        if hasattr(features_list[i], 'toarray'):
            x = torch.tensor(features_list[i].toarray(), dtype=torch.float)
        else:
            x = torch.tensor(features_list[i], dtype=torch.float)
        
        # Create Data object
        y = labels[i].unsqueeze(0) if labels[i].dim() == 0 else labels[i]
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list
