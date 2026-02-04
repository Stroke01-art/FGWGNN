import torch
from torch_geometric.utils import erdos_renyi_graph, barabasi_albert_graph, to_networkx, to_dense_adj, from_networkx
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch_geometric.data as pyg_data
from typing import List, Dict, Optional
import random

def generate_synthetic_graphs(num_graphs=1000, num_nodes_range=(10, 20), p_edge=0.3, feat_dim=8):
    """generate synthetic graphs for binary classification
    Args:
        num_graphs: graph numbers
        num_nodes_range: range of node numbers for each graph
        p_edge: prob of edge, 1st is 20% higher than 2nd class
        feat_dim: node feature dimension
    Returns:
        adj_lst: adj matrix
        features_lst: feture matrix
        idx_lst : graph indicator
        class_labels: graph label 0/1
    """
    adj_lst = []
    features_lst = []
    class_labels = []
    
    for i in range(num_graphs):
        # randomly chose class (0/1)
        class_label = np.random.randint(0, 2)
        actual_p = p_edge + 0.2 if class_label == 0 else p_edge
        
        # generate random graph
        n_nodes = np.random.randint(*num_nodes_range)
        G = nx.erdos_renyi_graph(n_nodes, actual_p)
        
        # change to adjacency csr matrix
        adj = nx.adjacency_matrix(G).astype(np.float32)
        adj_lst.append(adj)
        
        # generate node features (Gauss distribution with different mean value)
        mean = 0.5 if class_label == 0 else -0.5
        features = np.random.normal(loc=mean, scale=1.0, size=(n_nodes, feat_dim))
        features_lst.append(features)
        class_labels.append(class_label)
    
    return adj_lst, features_lst, np.array(class_labels)


def generate_graph_from_structure(
    num_nodes: int,
    graph_type: str,
    param=None
) -> torch.Tensor:
    """
    Generates an edge_index for a specified graph type.
    Handles 'empty' graphs as a special case.
    """
    if graph_type == 'regular':
        r = int(param)
        if (r * num_nodes) % 2 != 0: r = r - 1 if r > 0 else 0
        G = nx.random_regular_graph(r, num_nodes)
        return from_networkx(G).edge_index
        
    elif graph_type == 'gnp':
        p = param
        return erdos_renyi_graph(num_nodes, edge_prob=p)

    elif graph_type == 'ba':
        m = int(param)
        return barabasi_albert_graph(num_nodes, num_edges=m)

    elif graph_type == 'star':
        center_node, other_nodes = 0, torch.arange(1, num_nodes)
        s = torch.cat([torch.full_like(other_nodes, center_node), other_nodes])
        d = torch.cat([other_nodes, torch.full_like(other_nodes, center_node)])
        return torch.stack([s, d], dim=0)

    elif graph_type == 'empty':
        return torch.empty((2, 0), dtype=torch.long)
        
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    

def generate_graphless_dataset(
    num_graphs: int,
    num_nodes: int,
    num_features: int,
    graph_type: str,
    param=None,
    base_data: list = None
) -> (list, list): # type: ignore
    """
    Generates a dataset based on a graph-less teacher.
    If base_data is provided, it uses those features/labels and assigns new structures.
    If not, it creates new unique features/labels for each graph.
    """
    data_list = []
    
    # Define a simple "graph-less" teacher model
    teacher_weights = torch.randn(num_features, 1)

    is_base_provided = base_data is not None

    for i in range(num_graphs):
        if is_base_provided:
            # Use existing features and label
            features = base_data[i].x
            label = base_data[i].y
        else:
            # Create new, unique features and a corresponding label
            features = torch.randn(num_nodes, num_features)
            feature_sum = torch.sum(features, dim=0, keepdim=True)
            # Binary classification label based on the teacher
            label = torch.sign(torch.mm(feature_sum, teacher_weights)).long().squeeze()

        # Generate the specified graph structure
        edge_index = generate_graph_from_structure(num_nodes, graph_type, param)
        
        graph_data = pyg_data.Data(x=features, edge_index=edge_index, y=label)
        data_list.append(graph_data)

    if is_base_provided:
        return data_list # Just the list of graphs
    else:
        return data_list # Return the generated base data


def generate_unstructured_noise_dataset(
    num_graphs: int,
    num_nodes: int,
    num_features: int,
    noise_graph_type: str = 'gnp',
    noise_param: float = 0.2
) -> list:
    """
    Generates a dataset where each unique feature-label pair is assigned
    a random, uninformative graph structure.

    Args:
        num_graphs: The total number of unique graph samples to create.
        num_nodes: The number of nodes in each graph.
        num_features: The dimensionality of node features.
        noise_graph_type: The type of random graph to use (e.g., 'gnp').
        noise_param: The parameter for the graph generator (e.g., p for gnp).

    Returns:
        A list of PyG Data objects.
    """
    print(f"--- Generating Unstructured Noise Dataset (Design 2) ---")
    print(f"Using '{noise_graph_type}' graphs with param={noise_param} as structural noise.")
    
    data_list = []
    teacher_weights = torch.randn(num_features, 1)

    for _ in range(num_graphs):
        # 1. Create a unique feature set (X) and its label (y)
        features = torch.randn(num_nodes, num_features)
        feature_sum = torch.sum(features, dim=0, keepdim=True)
        label = torch.sign(torch.mm(feature_sum, teacher_weights)).long().squeeze()
        
        # 2. Generate a new, random graph structure (A)
        edge_index = generate_graph_from_structure(
            num_nodes,
            graph_type=noise_graph_type,
            param=noise_param
        )
        
        # 3. Combine them into a single PyG Data object
        graph_data = pyg_data.Data(x=features, edge_index=edge_index, y=label)
        data_list.append(graph_data)
        
    print(f"Successfully generated {len(data_list)} graph samples.")
    return data_list


def generate_mixed_structure_dataset(
    num_graphs: int,
    num_nodes: int,
    num_features: int,
) -> list:
    """
    Generates a dataset where each unique feature-label pair is assigned a
    graph structure randomly chosen from a mixture of different distributions.

    Args:
        num_graphs: The total number of unique graph samples to create.
        num_nodes: The number of nodes in each graph.
        num_features: The dimensionality of node features.

    Returns:
        A list of PyG Data objects.
    """
    print(f"--- Generating Mixed Structure Dataset (Design 3) ---")
    
    # Define the mixture of graph distributions to use
    structure_mixture = [
        {'type': 'regular', 'param': 5},
        {'type': 'ba', 'param': 3},
        {'type': 'gnp', 'param': 0.3},
        {'type': 'empty', 'param': None}
    ]
    print(f"Using a mixture of: {[s['type'] for s in structure_mixture]}")

    data_list = []
    teacher_weights = torch.randn(num_features, 1)

    for i in range(num_graphs):
        # 1. Create a unique feature set (X) and its label (y)
        features = torch.randn(num_nodes, num_features)
        feature_sum = torch.sum(features, dim=0, keepdim=True)
        label = torch.sign(torch.mm(feature_sum, teacher_weights)).long().squeeze()

        # 2. Pick a random graph structure from the mixture for this sample
        # You can also use a round-robin approach: structure_mixture[i % len(structure_mixture)]
        graph_spec = random.choice(structure_mixture)
        
        edge_index = generate_graph_from_structure(
            num_nodes,
            graph_type=graph_spec['type'],
            param=graph_spec['param']
        )
        
        # 3. Combine them into a single PyG Data object
        graph_data = pyg_data.Data(x=features, edge_index=edge_index, y=label)
        data_list.append(graph_data)

    print(f"Successfully generated {len(data_list)} graph samples.")
    return data_list


def convert_pyg_to_lists(pyg_dataset):
    """Converts a list of PyG Data objects to adj, features, and label lists."""
    adj_list, features_list, labels_list = [], [], []
    for data in pyg_dataset:
        # Convert edge_index to a NetworkX graph, then to numpy adjacency
        g = nx.Graph()
        g.add_nodes_from(range(data.num_nodes))
        g.add_edges_from(data.edge_index.t().numpy())
        adj = nx.to_numpy_array(g)
        adj_list.append(adj)
        
        features_list.append(data.x.numpy())
        labels_list.append(data.y.item())
        
    return adj_list, features_list, np.array(labels_list)
