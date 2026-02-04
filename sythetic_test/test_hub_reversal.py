"""
Extreme Hub Signal with Moderate Leaves Signal Experiment

Experiment Design:
- Graph Structure: All Star Graphs (1 hub, 20 leaves, N=21 total)
- Training: Strong signal ±10.0 for both hub and leaves
- Testing: Extreme hub ±100.0 vs moderate leaves ±5.2

Training Set - Hub and Leaves AGREE (strong signal):
  • Class 0: Hub = -10, All Leaves = -10
  • Class 1: Hub = +10, All Leaves = +10

Testing Set - Hub EXTREME+REVERSED, Leaves moderate+correct:
  • Class 0: Hub = +100 (WRONG, 10x training), All Leaves = -5.2 (correct, 0.52x training)
  • Class 1: Hub = -100 (WRONG, 10x training), All Leaves = +5.2 (correct, 0.52x training)

Question: Will extreme hub signal (100 vs 5.2 ≈ 19:1) overwhelm the leaves?
Expected: Model should fail if it trusts magnitude blindly.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import networkx as nx
import logging
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    prepare_table2_data, generate_sub_features_idx_vectorized,
    AverageMeter, accuracy, get_logger
)
from model import KerGNN


def generate_star_graph_data(
    num_samples,
    num_nodes=21,
    feature_dim=128,
    is_training=True,
    class0_hub_value=None,
    class0_leaves_value=None,
    class1_hub_value=None,
    class1_leaves_value=None,
    noise_std=1.0,
    noise_type='gaussian',
    noise_ratio=0.1,
    seed=42
):
    """
    Generate star graphs with configurable feature values.
    
    Args:
        num_samples: Number of graphs to generate
        num_nodes: Nodes per graph (default 21: 1 hub + 20 leaves)
        feature_dim: Feature dimension (default 1 for simplicity)
        is_training: If True, generate training data; if False, generate test data
        class0_hub_value: Feature value for Class 0 hub (default: -10 for train, +100 for test)
        class0_leaves_value: Feature value for Class 0 leaves (default: -10 for train, -5.2 for test)
        class1_hub_value: Feature value for Class 1 hub (default: +10 for train, -100 for test)
        class1_leaves_value: Feature value for Class 1 leaves (default: +10 for train, +5.2 for test)
        seed: Random seed
    
    Returns:
        adj_list, features_list, labels
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    adj_list = []
    features_list = []
    labels = []
    
    samples_per_class = num_samples // 2
    
    # Set default values based on is_training
    if is_training:
        # Training defaults
        if class0_hub_value is None:
            class0_hub_value = -10.0
        if class0_leaves_value is None:
            class0_leaves_value = -10.0
        if class1_hub_value is None:
            class1_hub_value = 10.0
        if class1_leaves_value is None:
            class1_leaves_value = 10.0
    else:
        # Testing defaults
        if class0_hub_value is None:
            class0_hub_value = 100.0
        if class0_leaves_value is None:
            class0_leaves_value = -5.2
        if class1_hub_value is None:
            class1_hub_value = -100.0
        if class1_leaves_value is None:
            class1_leaves_value = 5.2
    
    for class_id in range(2):
        for _ in range(samples_per_class):
            # Create star graph (1 hub + 20 leaves)
            G = nx.star_graph(num_nodes - 1)
            
            # Randomize node labels to avoid bias
            mapping = {i: j for i, j in enumerate(np.random.permutation(num_nodes))}
            G = nx.relabel_nodes(G, mapping)
            
            # Find the center (node with highest degree)
            degrees = dict(G.degree())
            center_node = max(degrees, key=degrees.get)
            leaf_nodes = [n for n in G.nodes() if n != center_node]
            
            # Initialize features (all zeros)
            features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
            
            # Set feature values based on class
            if class_id == 0:
                hub_value = class0_hub_value
                leaves_value = class0_leaves_value
            else:
                hub_value = class1_hub_value
                leaves_value = class1_leaves_value
            
            features[center_node, :] = hub_value
            for leaf in leaf_nodes:
                features[leaf, :] = leaves_value
            
            # DEBUG
            # Add Noise
            if noise_type == 'gaussian':
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, features.shape)
                    features = features + noise
            elif noise_type == 'relative':
                if noise_ratio > 0:
                    scale = noise_ratio * np.abs(features)
                    noise = np.random.normal(0, scale, features.shape)
                    features = features + noise
            features = features.astype(np.float32)

            if len(adj_list) < 20:
                mode = "TRAIN" if is_training else "TEST"
                print(f"  [{mode}] Class {class_id}, sample {len(adj_list)}: "
                      f"hub={features[center_node, 0]:.1f}, leaves={leaves_value:.1f}")
            
            adj = nx.adjacency_matrix(G).astype(np.float32)
            
            adj_list.append(adj)
            features_list.append(features)
            labels.append(class_id)

    # Verify data BEFORE shuffle (for debugging)
    if len(adj_list) == num_samples:  # Only verify once, when all samples generated
        _verify_before_shuffle = False  # Set to True to see pre-shuffle verification
        if _verify_before_shuffle:
            # print(f"\nPre-shuffle verification (signal_at_center={signal_at_center}):")
            for cid in range(2):
                start_idx = cid * samples_per_class
                for idx in range(start_idx, min(start_idx + 3, len(features_list))):
                    feat = features_list[idx]
                    adj = adj_list[idx]
                    # adj is already scipy sparse matrix, convert to networkx
                    G = nx.from_scipy_sparse_array(adj)
                    degrees = dict(G.degree())
                    center = max(degrees, key=degrees.get)
                    leaves = [n for n in G.nodes() if n != center]
                    center_val = feat[center, 0]
                    leaf_signals = sum(1 for n in leaves if abs(feat[n, 0]) > 0.1)
                    # DEBUG: check all nonzero features
                    all_nonzero = [(n, feat[n, 0]) for n in range(len(feat)) if abs(feat[n, 0]) > 0.1]
                    print(f"  Class {cid} sample {idx-start_idx}: center={center_val:.1f}, signal_leaves={leaf_signals}, all_signals={len(all_nonzero)}, center_node_id={center}")

    
    # Shuffle
    indices = np.random.permutation(num_samples)
    adj_list = [adj_list[i] for i in indices]
    features_list = [features_list[i] for i in indices]
    labels = np.array([labels[i] for i in indices])
    
    return adj_list, features_list, labels


def generate_hub_reversal_datasets(
    num_train=2000,
    num_test=400,
    num_nodes=20,
    feature_dim=128,
    train_class0_hub=None,
    train_class0_leaves=None,
    train_class1_hub=None,
    train_class1_leaves=None,
    test_class0_hub=None,
    test_class0_leaves=None,
    test_class1_hub=None,
    test_class1_leaves=None,
    noise_std=1.0,
    noise_type='gaussian',
    noise_ratio=0.1,
    train_seed=42,
    test_seed=10000
):
    """
    Generate datasets for signal position shift experiment.
    
    Training: Signal at center (hub)
    Testing: Signal at leaves (periphery)
    """
    print("=" * 80)
    print("Extreme Hub Signal with Moderate Leaves Signal Experiment")
    print("=" * 80)
    print(f"Graph Structure: Star graphs ({num_nodes} nodes)")
    print(f"  - 1 Hub node (degree = {num_nodes-1})")
    print(f"  - {num_nodes-1} Leaf nodes (degree = 1)")
    print(f"\nFeature Design:")
    print(f"  - Training: Strong signal ±10.0 for both hub and leaves")
    print(f"  - Testing: Extreme hub ±100.0 vs moderate leaves ±5.2")
    print(f"\nTraining Set (Hub and Leaves AGREE):")
    print(f"  - Class 0: Hub = -10.0, All Leaves = -10.0")
    print(f"  - Class 1: Hub = +10.0, All Leaves = +10.0")
    print(f"\nTesting Set (Hub EXTREME+REVERSED, Leaves moderate+correct):")
    print(f"  - Class 0: Hub = +100.0 (10x, WRONG), All Leaves = -5.2 (0.52x, correct)")
    print(f"  - Class 1: Hub = -100.0 (10x, WRONG), All Leaves = +5.2 (0.52x, correct)")
    print(f"\nMagnitude Challenge: Hub=100 vs Leaves=5.2 (19:1 ratio)")
    print(f"Quantity: 1 hub vs 20 leaves")
    print(f"Total: Hub contributes ±100, Leaves contribute ±104 (20×5.2)")
    print(f"Noise: Type={noise_type}, Std={noise_std}, Ratio={noise_ratio}")
    print("=" * 80)
    
    # Training data
    print("\nGenerating training data...")
    train_adj, train_features, train_labels = generate_star_graph_data(
        num_samples=num_train,
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        is_training=True,
        class0_hub_value=train_class0_hub,
        class0_leaves_value=train_class0_leaves,
        class1_hub_value=train_class1_hub,
        class1_leaves_value=train_class1_leaves,
        noise_std=noise_std,
        noise_type=noise_type,
        noise_ratio=noise_ratio,
        seed=train_seed
    )
    
    # Testing data
    print("\nGenerating test data...")
    print("\n" + "="*80)
    print("TEST SET GENERATION DEBUG:")
    print("="*80)
    test_adj, test_features, test_labels = generate_star_graph_data(
        num_samples=num_test,
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        is_training=False,
        class0_hub_value=test_class0_hub,
        class0_leaves_value=test_class0_leaves,
        class1_hub_value=test_class1_hub,
        class1_leaves_value=test_class1_leaves,
        noise_std=noise_std,
        noise_type=noise_type,
        noise_ratio=noise_ratio,
        seed=test_seed
    )
    
    # Verify the data
    print("\n" + "=" * 80)
    print("Data Verification:")
    print("=" * 80)
    
    for split_name, adjs, features, labels in [
        ('Training', train_adj, train_features, train_labels),
        ('Testing', test_adj, test_features, test_labels)
    ]:
        print(f"\n{split_name} Set ({len(adjs)} samples):")
        
# Simple verification: count signals per class
        for class_id in range(2):
            class_features = [f for f, l in zip(features, labels) if l == class_id]
            class_adjs = [a for a, l in zip(adjs, labels) if l == class_id]
            
            # For each sample, find which nodes have signals
            sample_summaries = []
            for sample_feat, sample_adj in zip(class_features[:10], class_adjs[:10]):
                G = nx.from_scipy_sparse_array(sample_adj)
                degrees = dict(G.degree())
                
                # Get signal info
                high_degree_nodes = [n for n, d in degrees.items() if d > 10]  # Hub nodes
                low_degree_nodes = [n for n, d in degrees.items() if d <= 2]  # Leaf nodes
                
                hub_signals = sum(1 for n in high_degree_nodes if abs(sample_feat[n, 0]) > 1e-6)
                leaf_signals = sum(1 for n in low_degree_nodes if abs(sample_feat[n, 0]) > 1e-6)
                
                sample_summaries.append((hub_signals, leaf_signals, len(high_degree_nodes), len(low_degree_nodes)))
            
            # Average across samples
            avg_hub_signals = np.mean([s[0] for s in sample_summaries])
            avg_leaf_signals = np.mean([s[1] for s in sample_summaries])
            num_hubs = sample_summaries[0][2]
            num_leaves = sample_summaries[0][3]
            
            print(f"  Class {class_id}:")
            print(f"    Hub nodes (degree>10): {avg_hub_signals:.1f}/{num_hubs} have signals")
            print(f"    Leaf nodes (degree<=2): {avg_leaf_signals:.1f}/{num_leaves} have signals")
    
    print("=" * 80)
    
    datasets = {
        'train': {
            'adj': train_adj,
            'features': train_features,
            'labels': train_labels
        },
        'test': {
            'adj': test_adj,
            'features': test_features,
            'labels': test_labels
        }
    }
    
    return datasets


def train_epoch(model, adj_batch, features_batch, subadj_batch, subidx_batch,
                graph_indicator_batch, y_batch, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    n_batches = len(adj_batch)
    
    for i in range(n_batches):
        optimizer.zero_grad()
        output = model(subadj_batch[i], features_batch[i], 
                      subidx_batch[i], graph_indicator_batch[i])
        loss = F.cross_entropy(output, y_batch[i])
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), output.size(0))
        train_acc.update(accuracy(output.data, y_batch[i].data), output.size(0))
    
    return train_loss.avg, train_acc.avg


def test_epoch(model, adj_batch, features_batch, subadj_batch, subidx_batch,
               graph_indicator_batch, y_batch, device):
    """Test for one epoch."""
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    
    n_batches = len(adj_batch)
    
    with torch.no_grad():
        for i in range(n_batches):
            output = model(subadj_batch[i], features_batch[i],
                          subidx_batch[i], graph_indicator_batch[i])
            loss = F.cross_entropy(output, y_batch[i])
            
            test_loss.update(loss.item(), output.size(0))
            test_acc.update(accuracy(output.data, y_batch[i].data), output.size(0))
    
    return test_loss.avg, test_acc.avg


def run_experiment(args, datasets, logger):
    """Run single experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Preparing batched data...")
    
    datasets['train']['name'] = 'hub_reversal_train'
    datasets['test']['name'] = 'hub_reversal_test'
    
    formatted_datasets = {
        'train': datasets['train'],
        'test': {
            'Test (Hub Reversed)': datasets['test']
        }
    }
    
    prepared = prepare_table2_data(formatted_datasets, args.batch_size, device)
    
    # Generate subgraph indices
    logger.info("Generating subgraph indices...")
    train_subadj, train_subidx = generate_sub_features_idx_vectorized(
        prepared['train']['adj'],
        prepared['train']['features'],
        size_subgraph=args.size_subgraph,
        k_neighbor=args.k
    )
    
    test_subadj, test_subidx = generate_sub_features_idx_vectorized(
        prepared['test']['Test (Hub Reversed)']['adj'],
        prepared['test']['Test (Hub Reversed)']['features'],
        size_subgraph=args.size_subgraph,
        k_neighbor=args.k
    )
    
    # Create model
    logger.info(f"Creating {args.kernel.upper()} KerGNN model...")
    model = KerGNN(
        input_dim=args.feature_dim,
        output_dim=2,
        hidden_dims=args.hidden_dims,
        kernel=args.kernel,
        max_step=args.max_step,
        size_graph_filter=[args.size_graph_filter] * (len(args.hidden_dims) - 1),
        size_subgraph=args.size_subgraph,
        num_mlp_layers=args.num_mlp_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout_rate=args.dropout_rate,
        no_norm=args.no_norm,
        use_node_norm=args.use_node_norm,
        use_local_alpha=args.use_local_alpha,
        alpha_hidden_dim=args.alpha_hidden_dim
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Data-driven initialization
    if args.init_method == 'data':
        logger.info("Using data-driven initialization...")
        for layer_idx, ker_layer in enumerate(model.ker_layers):
            if hasattr(ker_layer, 'init_from_data'):
                ker_layer.init_from_data(
                    datasets['train']['adj'],
                    datasets['train']['features'],
                    datasets['train']['labels'],
                    device
                )
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Training loop
    best_train_acc = 0
    best_test_acc = 0
    
    logger.info("\nStarting training...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, prepared['train']['adj'], prepared['train']['features'],
            train_subadj, train_subidx,
            prepared['train']['graph_indicator'], prepared['train']['labels'],
            optimizer, device
        )
        
        best_train_acc = max(best_train_acc, train_acc)
        
        # Test
        if epoch % args.test_freq == 0 or epoch == args.epochs:
            test_loss, test_acc = test_epoch(
                model, 
                prepared['test']['Test (Hub Reversed)']['adj'],
                prepared['test']['Test (Hub Reversed)']['features'],
                test_subadj, test_subidx,
                prepared['test']['Test (Hub Reversed)']['graph_indicator'],
                prepared['test']['Test (Hub Reversed)']['labels'],
                device
            )
            best_test_acc = max(best_test_acc, test_acc)
            
            logger.info(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                       f"Test Loss {test_loss:.4f} Acc {test_acc:.4f}")
        else:
            logger.info(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f}")
        
        scheduler.step()
    
    # Analyze learned parameters
    logger.info("\n" + "=" * 80)
    logger.info("Analysis of Learned Parameters:")
    logger.info("=" * 80)
    
    for layer_idx, ker_layer in enumerate(model.ker_layers):
        logger.info(f"\nLayer {layer_idx} ({type(ker_layer).__name__}):")
        
        if hasattr(ker_layer, 'alpha'):
            # FGW kernel
            alpha = torch.sigmoid(ker_layer.alpha).item()
            logger.info(f"  Alpha (structure weight): {alpha:.4f}")
            logger.info(f"  1-Alpha (feature weight): {1-alpha:.4f}")
            
            if alpha > 0.6:
                logger.info("  ⚠️  High structure weight - may rely on hub position!")
            elif alpha < 0.4:
                logger.info("  ✓  High feature weight - robust to position shift")
            else:
                logger.info("  ～ Balanced weights")
        
        if hasattr(ker_layer, 'adj_hidden'):
            # RW kernel
            adj_norm = torch.norm(ker_layer.adj_hidden.data).item()
            feat_norm = torch.norm(ker_layer.features_hidden.data).item()
            
            logger.info(f"  Adj weight norm: {adj_norm:.2f}")
            logger.info(f"  Feature weight norm: {feat_norm:.2f}")
            feat_dominance = feat_norm / (adj_norm + feat_norm)
            logger.info(f"  Feature dominance: {feat_dominance:.2%}")
            
            if feat_dominance > 0.7:
                logger.info("  ✓  Strong feature dominance - should generalize well")
            elif feat_dominance > 0.5:
                logger.info("  ～ Moderate feature dominance")
            else:
                logger.info("  ⚠️  Structure dominance - may fail on position shift!")
    
    logger.info("\n" + "=" * 80)
    logger.info("Final Results:")
    logger.info("=" * 80)
    logger.info(f"Best Train Accuracy: {best_train_acc:.2%}")
    logger.info(f"Best Test Accuracy:  {best_test_acc:.2%}")
    
    # Interpretation
    logger.info("\n" + "=" * 80)
    logger.info("INTERPRETATION:")
    logger.info("=" * 80)
    
    if best_test_acc > 0.95:
        logger.info("✓ EXCELLENT! Model resists extreme magnitude misleading!")
        logger.info("  Despite 19:1 magnitude ratio, model follows leaves.")
        logger.info("  Shows robustness to out-of-distribution extreme values.")
    elif best_test_acc > 0.80:
        logger.info("～ GOOD: Model partially resists extreme hub signal")
        logger.info("  Some robustness to magnitude shifts.")
    elif best_test_acc > 0.60:
        logger.info("⚠️  POOR: Model confused by extreme hub values")
        logger.info("  Struggles between extreme hub (±100) and moderate leaves (±5.2).")
    else:
        logger.info("✗ FAILED! Model blindly follows magnitude!")
        logger.info("  Model ignores 20 leaves (total ±104) and follows 1 hub (±100).")
        logger.info("  This indicates severe magnitude bias and poor OOD generalization.")
    
    logger.info("=" * 80)
    
    return {
        'train_acc': float(best_train_acc.cpu()) if torch.is_tensor(best_train_acc) else float(best_train_acc),
        'test_acc': float(best_test_acc.cpu()) if torch.is_tensor(best_test_acc) else float(best_test_acc)
    }


def main():
    parser = argparse.ArgumentParser(description='Extreme Hub Signal Experiment')
    
    # Experiment settings
    parser.add_argument('--kernel', default='rw', choices=['rw', 'fgw'],
                        help='Kernel type')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of runs')
    
    # Data settings
    parser.add_argument('--num-train', type=int, default=2000,
                        help='Number of training samples')
    parser.add_argument('--num-test', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--num-nodes', type=int, default=21,
                        help='Number of nodes per graph')
    parser.add_argument('--feature-dim', type=int, default=128,
                        help='Node feature dimension')
    
    # Feature value settings
    parser.add_argument('--noise-type', default='gaussian', choices=['gaussian', 'relative'], help='Noise type: gaussian or relative')
    parser.add_argument('--noise-std', type=float, default=1.0, help='Standard deviation of Gaussian noise added to features')
    parser.add_argument('--noise-ratio', type=float, default=0.1, help='Ratio for relative noise (std = ratio * |value|)')
    parser.add_argument('--train-class0-hub', type=float, default=-10.0,
                        help='Training Class 0 hub feature value')
    parser.add_argument('--train-class0-leaves', type=float, default=-10.0,
                        help='Training Class 0 leaves feature value')
    parser.add_argument('--train-class1-hub', type=float, default=10.0,
                        help='Training Class 1 hub feature value')
    parser.add_argument('--train-class1-leaves', type=float, default=10.0,
                        help='Training Class 1 leaves feature value')
    parser.add_argument('--test-class0-hub', type=float, default=100.0,
                        help='Testing Class 0 hub feature value')
    parser.add_argument('--test-class0-leaves', type=float, default=-5.2,
                        help='Testing Class 0 leaves feature value')
    parser.add_argument('--test-class1-hub', type=float, default=-100.0,
                        help='Testing Class 1 hub feature value')
    parser.add_argument('--test-class1-leaves', type=float, default=5.2,
                        help='Testing Class 1 leaves feature value')
    
    # Model settings
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[32, 64],
                        help='Hidden dimensions')
    parser.add_argument('--size-subgraph', type=int, default=8,
                        help='Size of subgraph')
    parser.add_argument('--size-graph-filter', type=int, default=8,
                        help='Size of graph filters')
    parser.add_argument('--k', type=int, default=1,
                        help='k-hop neighborhood')
    parser.add_argument('--max-step', type=int, default=1,
                        help='Max steps for RW kernel')
    parser.add_argument('--num-mlp-layers', type=int, default=1,
                        help='Number of MLP layers')
    parser.add_argument('--mlp-hidden-dim', type=int, default=None,
                        help='MLP hidden dimension')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--no-norm', action='store_true', default=True,
                        help='Disable normalization')
    parser.add_argument('--use-node-norm', action='store_true', default=False,
                        help='Use BatchNorm at node level')
    
    # Initialization
    parser.add_argument('--init-method', default='random', choices=['random', 'data'],
                        help='Initialization method')
    
    # FGW Alpha settings (only for FGW kernel)
    parser.add_argument('--use-local-alpha', action='store_true', default=False,
                        help='Use local alpha (learnable per patch) for FGW kernel')
    parser.add_argument('--alpha-hidden-dim', type=int, default=32,
                        help='Hidden dimension for local alpha MLPs')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--test-freq', type=int, default=10,
                        help='Test frequency')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                        help='Logging level')
    parser.add_argument('--save-dir', default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, f'hub_reversal_{args.kernel}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logger
    logger = get_logger(save_dir, __name__, 
                       f'hub_reversal_{args.kernel}.log',
                       level=getattr(logging, args.log_level))
    
    logger.info("=" * 80)
    logger.info("Extreme Hub Signal with Moderate Leaves Signal Experiment")
    logger.info("=" * 80)
    logger.info(f"Kernel: {args.kernel.upper()}")
    logger.info(f"\nObjective: Test model robustness to extreme OOD magnitude shifts")
    logger.info(f"           Training: ±10 | Testing: Hub ±100 (wrong) vs Leaves ±5.2 (correct)")
    logger.info("=" * 80)
    
    all_results = []
    
    for run_idx in range(args.num_runs):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Run {run_idx + 1}/{args.num_runs}")
        logger.info("=" * 80)
        
        # Set seed
        seed = 42 + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate data
        datasets = generate_hub_reversal_datasets(
            num_train=args.num_train,
            num_test=args.num_test,
            num_nodes=args.num_nodes,
            feature_dim=args.feature_dim,
            train_class0_hub=args.train_class0_hub,
            train_class0_leaves=args.train_class0_leaves,
            train_class1_hub=args.train_class1_hub,
            train_class1_leaves=args.train_class1_leaves,
            test_class0_hub=args.test_class0_hub,
            test_class0_leaves=args.test_class0_leaves,
            test_class1_hub=args.test_class1_hub,
            test_class1_leaves=args.test_class1_leaves,
            noise_std=args.noise_std,
            noise_type=args.noise_type,
            noise_ratio=args.noise_ratio,
            train_seed=seed,
            test_seed=seed + 10000
        )
        
        # Run experiment
        results = run_experiment(args, datasets, logger)
        all_results.append(results)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    train_accs = [r['train_acc'] for r in all_results]
    test_accs = [r['test_acc'] for r in all_results]
    
    logger.info(f"Train Accuracy: {np.mean(train_accs):.2%} ± {np.std(train_accs):.2%}")
    logger.info(f"Test Accuracy:  {np.mean(test_accs):.2%} ± {np.std(test_accs):.2%}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
