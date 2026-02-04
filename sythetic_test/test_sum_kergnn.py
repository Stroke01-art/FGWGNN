"""
Test script for KerGNN on Table 2 data.
Tests both RW_layer and FGW_layer architectures.
"""
import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import logging

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_table2_datasets
from utils import (
    prepare_table2_data, generate_sub_features_idx_vectorized,
    AverageMeter, accuracy, get_logger
)

# Import models
from model import KerGNN


def args_parser():
    parser = argparse.ArgumentParser(description='KerGNN Table 2 Experiments')
    
    # Experiment settings
    parser.add_argument('--kernel', default='rw', choices=['rw', 'fgw'],
                        help='Kernel type: rw or fgw')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of independent runs')
    
    # Data settings
    parser.add_argument('--num-train', type=int, default=2000,
                        help='Number of training samples')
    parser.add_argument('--num-test', type=int, default=100,
                        help='Number of test samples per distribution')
    parser.add_argument('--num-nodes', type=int, default=20,
                        help='Number of nodes per graph')
    parser.add_argument('--feature-dim', type=int, default=64,
                        help='Node feature dimension')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Classification margin')
    
    # Graph representation
    parser.add_argument('--use-shortest-path', action='store_true', default=True,
                        help='Use shortest path distance matrix instead of adjacency')
    parser.add_argument('--normalize-sp', action='store_true', default=True,
                        help='Normalize shortest path distances to [0, 1]')
    
    # Model settings
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[64, 32],
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
                        help='Disable normalization (keep sum as is)')    
    parser.add_argument('--use-node-norm', action='store_true', default=False,
                        help='Use BatchNorm at node level')
    
    # Initialization settings
    parser.add_argument('--init-method', default='random', choices=['random', 'data'],
                        help='Initialization method: random or data-driven')
    
    # FGW Alpha settings (only for FGW kernel)
    parser.add_argument('--use-local-alpha', action='store_true', default=True,
                        help='Use local alpha (learnable per patch) for FGW kernel')
    parser.add_argument('--alpha-hidden-dim', type=int, default=32,
                        help='Hidden dimension for local alpha MLPs')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
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
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def train_epoch(model, adj_batch, features_batch, subadj_batch, subidx_batch,
                graph_indicator_batch, y_batch, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    n_batches = len(adj_batch)
    epoch_start = time.time()
    
    for i in range(n_batches):
        optimizer.zero_grad()
        output = model(subadj_batch[i], features_batch[i], 
                      subidx_batch[i], graph_indicator_batch[i])
        loss = F.cross_entropy(output, y_batch[i])
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), output.size(0))
        train_acc.update(accuracy(output.data, y_batch[i].data), output.size(0))
    
    epoch_time = time.time() - epoch_start
    return train_loss.avg, train_acc.avg, epoch_time


def test_epoch(model, adj_batch, features_batch, subadj_batch, subidx_batch,
               graph_indicator_batch, y_batch, device):
    """Test for one epoch."""
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    
    n_batches = len(adj_batch)
    test_start = time.time()
    
    with torch.no_grad():
        for i in range(n_batches):
            output = model(subadj_batch[i], features_batch[i],
                          subidx_batch[i], graph_indicator_batch[i])
            loss = F.cross_entropy(output, y_batch[i])
            
            test_loss.update(loss.item(), output.size(0))
            test_acc.update(accuracy(output.data, y_batch[i].data), output.size(0))
    
    test_time = time.time() - test_start
    return test_loss.avg, test_acc.avg, test_time


def run_single_experiment(args, datasets, run_idx, logger):
    """Run single experiment (one random seed)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Preparing batched data...")
    prepared = prepare_table2_data(datasets, args.batch_size, device, 
                                          use_shortest_path=args.use_shortest_path,
                                          normalize_sp=args.normalize_sp)
    
    # Generate subgraph indices
    logger.info("Generating subgraph indices...")
    train_subadj, train_subidx = generate_sub_features_idx_vectorized(
        prepared['train']['adj'],
        prepared['train']['features'],
        size_subgraph=args.size_subgraph,
        k_neighbor=args.k
    )
    
    test_data = {}
    for test_name, test_prep in prepared['test'].items():
        test_subadj, test_subidx = generate_sub_features_idx_vectorized(
            test_prep['adj'],
            test_prep['features'],
            size_subgraph=args.size_subgraph,
            k_neighbor=args.k
        )
        test_data[test_name] = {
            'adj': test_prep['adj'],
            'features': test_prep['features'],
            'subadj': test_subadj,
            'subidx': test_subidx,
            'graph_indicator': test_prep['graph_indicator'],
            'labels': test_prep['labels']
        }
    
    # Create model
    logger.info(f"Creating {args.kernel.upper()} KerGNN model...")
    model = KerGNN(
        input_dim=args.feature_dim,
        output_dim=2,  # Binary classification
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
    
    # Initialize prototypes with data-driven method if requested
    if args.init_method == 'data':
        logger.info("Using data-driven initialization for prototypes...")
        # Get raw training data (before batching)
        train_adj_list = datasets['train']['adj']
        train_features_list = datasets['train']['features']
        train_labels = datasets['train']['labels']
        
        # Initialize each kernel layer
        for layer_idx, ker_layer in enumerate(model.ker_layers):
            if hasattr(ker_layer, 'init_from_data'):
                logger.info(f"  Initializing layer {layer_idx} ({type(ker_layer).__name__})...")
                ker_layer.init_from_data(train_adj_list, train_features_list, train_labels, device)
    else:
        logger.info("Using random initialization (default)...")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Training loop
    best_train_acc = 0
    test_results = {}
    total_train_time = 0.0
    total_test_time = 0.0
    
    logger.info("\nStarting training...")
    training_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_time = train_epoch(
            model, prepared['train']['adj'], prepared['train']['features'],
            train_subadj, train_subidx,
            prepared['train']['graph_indicator'], prepared['train']['labels'],
            optimizer, device
        )
        
        total_train_time += train_time
        best_train_acc = max(best_train_acc, train_acc)
        
        # Log
        log_str = (f"Epoch {epoch:03d} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} "
                  f"Time {train_time:.2f}s")
        
        # Test periodically
        if epoch % args.test_freq == 0 or epoch == args.epochs:
            test_results_epoch = {}
            test_time_total = 0.0
            for test_name, test_prep in test_data.items():
                test_loss, test_acc, test_time = test_epoch(
                    model, test_prep['adj'], test_prep['features'],
                    test_prep['subadj'], test_prep['subidx'],
                    test_prep['graph_indicator'], test_prep['labels'],
                    device
                )
                test_results_epoch[test_name] = test_acc
                test_time_total += test_time
                
                if args.verbose:
                    log_str += f" | {test_name}: {test_acc:.4f}"
            
            total_test_time += test_time_total
            log_str += f" (Test {test_time_total:.2f}s)"
            
            # Store final results
            if epoch == args.epochs:
                test_results = test_results_epoch
        
        logger.info(log_str)
        scheduler.step()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - training_start
    
    # Log timing summary
    logger.info("\n" + "-" * 80)
    logger.info("Training Time Summary:")
    logger.info(f"  Total training time: {total_train_time:.2f}s ({total_train_time/60:.2f}m)")
    logger.info(f"  Total test time: {total_test_time:.2f}s ({total_test_time/60:.2f}m)")
    logger.info(f"  Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    logger.info(f"  Average time per epoch: {total_train_time/args.epochs:.2f}s")
    logger.info("-" * 80)
    
    return test_results


def main():
    args = args_parser()
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, f'{args.kernel}_table2')
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logger
    logger = get_logger(save_dir, __name__, 
                       f'table2_{args.kernel}.log',
                       level=getattr(logging, args.log_level))
    
    logger.info("=" * 80)
    logger.info(f"Table 2 Experiment with {args.kernel.upper()} KerGNN")
    logger.info("=" * 80)
    logger.info(f"Settings: {vars(args)}")
    
    # Collect results across runs
    all_results = {}
    
    for run_idx in range(args.num_runs):
        logger.info("\n" + "=" * 80)
        logger.info(f"Run {run_idx + 1}/{args.num_runs}")
        logger.info("=" * 80)
        
        # Set seeds
        seed = 42 + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate data
        logger.info("Generating Table 2 datasets...")
        datasets = generate_table2_datasets(
            num_train=args.num_train,
            num_test=args.num_test,
            num_nodes=args.num_nodes,
            feature_dim=args.feature_dim,
            margin=args.margin,
            train_seed=seed,
            test_seed=seed + 10000
        )
        
        logger.info(f"Train samples: {len(datasets['train']['adj'])}")
        logger.info(f"Test distributions: {list(datasets['test'].keys())}")
        
        # Run experiment
        results = run_single_experiment(args, datasets, run_idx, logger)
        
        # Accumulate results
        for test_name, acc in results.items():
            if test_name not in all_results:
                all_results[test_name] = []
            all_results[test_name].append(float(acc))
        
        logger.info(f"\nRun {run_idx + 1} Results:")
        for test_name, acc in results.items():
            logger.info(f"  {test_name}: {acc:.4f}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS (Table 2)")
    logger.info("=" * 80)
    logger.info(f"{'Test Distribution':<25} {'Mean Accuracy':<15} {'Std Dev':<10}")
    logger.info("-" * 80)
    
    summary = []
    for test_name in sorted(all_results.keys()):
        accs = all_results[test_name]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        logger.info(f"{test_name:<25} {mean_acc:.4f} ({mean_acc*100:.2f}%)  ± {std_acc:.4f}")
        summary.append({
            'distribution': test_name,
            'mean': mean_acc,
            'std': std_acc
        })
    
    logger.info("=" * 80)
    
    # Save results
    import json
    results_file = os.path.join(save_dir, f'results_{args.kernel}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': summary,
            'raw_results': {k: [float(v) for v in vals] 
                           for k, vals in all_results.items()}
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
