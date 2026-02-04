"""
Data generation for Table 2 experiments with KerGNN.
Generates sum task data with different graph distributions.
Matches the implementation from successful run_table_2 reproduction.
"""
import numpy as np
import networkx as nx
import torch
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional
import random as py_random


def sample_self_teacher(d, rng):
    """Sample self (feature-based) component of teacher."""
    return rng.standard_normal(size=(d, 1)).astype(np.float32)


def sample_top_teacher(d, rng):
    """Sample top (topology-based) component of teacher."""
    return rng.standard_normal(size=(d, 1)).astype(np.float32)


def sample_teacher(feature_dim, use_self=True, use_top=False, seed=None):
    """
    Generate teacher vector for sum task.
    
    Args:
        feature_dim: Feature dimension
        use_self: Use self component (feature sum)
        use_top: Use topology component (degree-weighted sum)
        seed: Random seed
    
    Returns:
        teacher: Teacher vector of shape (2*feature_dim,)
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    self_teacher = np.zeros((feature_dim, 1), dtype=np.float32)
    top_teacher = np.zeros_like(self_teacher)
    
    if use_self:
        self_teacher = sample_self_teacher(feature_dim, rng)
    if use_top:
        top_teacher = sample_top_teacher(feature_dim, rng)
    
    teacher = np.concatenate([self_teacher, top_teacher], axis=0).flatten()
    norm = np.linalg.norm(teacher)
    
    if norm == 0.0:
        raise ValueError("Teacher is zero; enable at least one component.")
    
    return teacher / norm


def _build_graph(dist, num_nodes, rng):
    """
    Build a graph based on distribution specification.
    Matches the implementation from successful run_table_2.
    """
    nx_seed = int(rng.integers(1 << 30))
    
    if dist["name"] == "gnp":
        return nx.gnp_random_graph(num_nodes, dist["param"], seed=nx_seed)
    
    if dist["name"] == "ba":
        m = int(dist["param"])
        return nx.barabasi_albert_graph(num_nodes, m, seed=nx_seed)
    
    if dist["name"] == "regular":
        d = int(dist["param"])
        return nx.random_regular_graph(d, num_nodes, seed=nx_seed)
    
    if dist["name"] == "star":
        # Create star graph: nx.star_graph(n) creates a star with n+1 nodes
        # We want num_nodes total, so create star_graph(num_nodes-1)
        # Then shuffle node labels to randomize which node is the center
        G = nx.star_graph(num_nodes - 1)
        # Relabel nodes randomly to avoid always having node 0 as center
        nodes = list(range(num_nodes))
        py_random.Random(nx_seed).shuffle(nodes)
        mapping = {i: nodes[i] for i in range(num_nodes)}
        G = nx.relabel_nodes(G, mapping)
        return G
    
    if dist["name"] == "empty":
        return nx.empty_graph(num_nodes)
    
    # === New challenging graph types ===
    if dist["name"] == "chain":
        # Linear chain: maximum diameter, minimal clustering
        G = nx.path_graph(num_nodes)
        return G
    
    if dist["name"] == "binary_tree":
        # Balanced binary tree: hierarchical structure
        # Calculate depth to get approximately num_nodes
        depth = int(np.log2(num_nodes + 1)) - 1
        G = nx.balanced_tree(2, depth)
        actual_nodes = G.number_of_nodes()
        
        # If not enough nodes, connect additional nodes to leaves
        if actual_nodes < num_nodes:
            leaves = [n for n in G.nodes() if G.degree(n) == 1]
            extra_nodes = num_nodes - actual_nodes
            for i in range(extra_nodes):
                new_node = actual_nodes + i
                # Connect to a random leaf
                leaf = leaves[i % len(leaves)]
                G.add_edge(leaf, new_node)
        # If too many nodes, remove leaves until we reach num_nodes
        elif actual_nodes > num_nodes:
            while G.number_of_nodes() > num_nodes:
                # Find a leaf (degree 1) and remove it
                leaves = [n for n in G.nodes() if G.degree(n) == 1]
                if leaves:
                    G.remove_node(leaves[0])
                else:
                    break
        return G
    
    if dist["name"] == "lollipop":
        # Lollipop: dense clique + sparse tail
        # Split nodes: 60% clique, 40% path
        clique_size = max(3, int(num_nodes * 0.6))
        path_size = num_nodes - clique_size
        if path_size < 1:
            path_size = 1
            clique_size = num_nodes - 1
        G = nx.lollipop_graph(clique_size, path_size)
        return G
    
    if dist["name"] == "grid":
        # 2D grid: regular spatial structure
        # Make it approximately square
        side = int(np.sqrt(num_nodes))
        G = nx.grid_2d_graph(side, side)
        # Convert node labels to integers
        G = nx.convert_node_labels_to_integers(G)
        actual_nodes = G.number_of_nodes()
        
        # If not enough nodes, connect additional nodes to boundary
        if actual_nodes < num_nodes:
            # Find boundary nodes (nodes with degree < 4)
            boundary = [n for n in G.nodes() if G.degree(n) < 4]
            extra_nodes = num_nodes - actual_nodes
            for i in range(extra_nodes):
                new_node = actual_nodes + i
                # Connect to a boundary node
                neighbor = boundary[i % len(boundary)]
                G.add_edge(neighbor, new_node)
        # If too many nodes, remove boundary nodes
        elif actual_nodes > num_nodes:
            while G.number_of_nodes() > num_nodes:
                # Find boundary nodes and remove
                boundary = [n for n in G.nodes() if G.degree(n) < 4]
                if boundary:
                    G.remove_node(boundary[0])
                else:
                    break
        return G
    
    raise ValueError(f"Unsupported graph distribution: {dist['name']}")


def generate_sum_task_data(
    num_samples,
    num_nodes,
    feature_dim,
    graph_type,
    graph_param,
    margin=0.1,
    teacher=None,
    use_self=True,
    use_top=False,
    seed=None,
    normalize_features=False,
):
    """
    Generate sum task dataset for KerGNN.
    Matches the logic from successful run_table_2 reproduction.
    
    Args:
        num_samples: Number of graphs to generate
        num_nodes: Number of nodes per graph
        feature_dim: Feature dimension
        graph_type: Type of graph distribution
        graph_param: Parameter for graph generation
        margin: Minimum margin for classification
        teacher: Pre-specified teacher (if None, will generate)
        use_self: Use self component in teacher
        use_top: Use topology component in teacher
        seed: Random seed
        normalize_features: Whether to normalize feature std by 1/num_nodes
    
    Returns:
        adj_list: List of adjacency matrices (scipy sparse)
        features_list: List of feature matrices (numpy arrays)
        labels: Array of labels (0 or 1)
        teacher: The teacher vector used
    """
    rng = np.random.default_rng(seed)
    
    # Generate teacher if not provided
    if teacher is None:
        teacher = sample_teacher(feature_dim, use_self, use_top, seed)
    
    features_std = (1.0 / num_nodes) if normalize_features else 1.0
    dist_cfg = {"name": graph_type, "param": graph_param}
    
    adj_list = []
    features_list = []
    labels = []
    
    attempts = 0
    max_attempts = num_samples * 20
    
    while len(adj_list) < num_samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not generate {num_samples} samples with margin={margin}. "
                f"Generated {len(adj_list)}/{num_samples} after {attempts} attempts."
            )
        attempts += 1
        
        # Generate graph structure
        graph = _build_graph(dist_cfg, num_nodes, rng)
        
        # Get adjacency matrix (as numpy dense for label computation)
        adj_dense = nx.to_numpy_array(graph, dtype=np.float32)
        
        # Generate node features
        x = features_std * rng.standard_normal(size=(num_nodes, feature_dim)).astype(np.float32)
        
        # Compute label using teacher
        degrees = adj_dense.sum(axis=1, keepdims=True)
        sum_features = x.sum(axis=0)
        weighted_sum_features = (degrees * x).sum(axis=0)
        svm_features = np.concatenate([sum_features, weighted_sum_features], axis=0)
        
        score = float(np.dot(svm_features, teacher))
        
        # Check margin
        if abs(score) < margin:
            continue
        
        # Assign label (0 or 1 for KerGNN)
        label = 1 if score >= 0 else 0
        
        # Store as scipy sparse for KerGNN
        adj_sparse = csr_matrix(adj_dense)
        adj_list.append(adj_sparse)
        features_list.append(x)
        labels.append(label)
    
    return adj_list, features_list, np.array(labels), teacher


def generate_table2_datasets(
    num_train=2000,
    num_test=500,
    num_nodes=20,
    feature_dim=128,
    margin=0.1,
    train_seed=42,
    test_seed=10000
):
    """
    Generate all datasets for Table 2 experiment.
    
    Training: 5-regular graphs
    Testing: Multiple distribution shifts
    
    Returns:
        Dictionary with train and test datasets for each distribution
    """
    datasets = {}
    
    # Generate training data (5-regular)
    print("Generating training data (5-regular)...")
    train_adj, train_feat, train_labels, teacher = generate_sum_task_data(
        num_samples=num_train,
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        graph_type='regular',
        graph_param=5,
        margin=margin,
        use_self=True,
        use_top=False,
        seed=train_seed,
        normalize_features=False  # CRITICAL: Must be False to match paper
    )
    
    datasets['train'] = {
        'adj': train_adj,
        'features': train_feat,
        'labels': train_labels,
        'name': 'Train (Regular r=5)'
    }
    
    # Test configurations
    test_configs = [
        ('Regular r=5', 'regular', 5),
        ('Regular r=10', 'regular', 10),
        ('Regular r=15', 'regular', 15),
        ('GNP p=0.2', 'gnp', 0.2),
        ('GNP p=0.5', 'gnp', 0.5),
        ('GNP p=0.8', 'gnp', 0.8),
        ('BA m=3', 'ba', 3),
        ('BA m=15', 'ba', 15),
        ('Star', 'star', None),
        # New challenging graph types
        ('Chain', 'chain', None),
        ('Binary Tree', 'binary_tree', None),
        ('Lollipop', 'lollipop', None),
        ('Grid', 'grid', None),
    ]
    
    datasets['test'] = {}
    
    for name, graph_type, param in test_configs:
        print(f"Generating test data: {name}...")
        test_adj, test_feat, test_labels, _ = generate_sum_task_data(
            num_samples=num_test,
            num_nodes=num_nodes,
            feature_dim=feature_dim,
            graph_type=graph_type,
            graph_param=param,
            margin=margin,
            teacher=teacher,  # Use same teacher as training!
            use_self=True,
            use_top=False,
            seed=test_seed,
            normalize_features=False  # CRITICAL: Must be False to match paper
        )
        
        datasets['test'][name] = {
            'adj': test_adj,
            'features': test_feat,
            'labels': test_labels,
            'name': name
        }
        
        test_seed += 1000  # Different seed for each test set
    
    datasets['teacher'] = teacher
    
    return datasets


if __name__ == "__main__":
    # Quick test
    print("Testing data generation...")
    
    # Test single dataset generation
    adj_list, feat_list, labels, teacher = generate_sum_task_data(
        num_samples=10,
        num_nodes=20,
        feature_dim=16,
        graph_type='regular',
        graph_param=5,
        margin=0.1,
        seed=42,
        normalize_features=False
    )
    
    print(f"Generated {len(adj_list)} graphs")
    print(f"Feature dim: {feat_list[0].shape[1]}")
    print(f"Labels: {labels}")
    print(f"Teacher shape: {teacher.shape}")
    print(f"Feature std: {feat_list[0].std():.4f}")
    print(f"Feature mean: {feat_list[0].mean():.4f}")
    
    # Test Table 2 generation
    print("\nGenerating Table 2 datasets (small test)...")
    datasets = generate_table2_datasets(
        num_train=100,
        num_test=50,
        num_nodes=20,
        feature_dim=16
    )
    
    print(f"Train samples: {len(datasets['train']['adj'])}")
    print(f"Test distributions: {list(datasets['test'].keys())}")
    print("Data generation successful!")
