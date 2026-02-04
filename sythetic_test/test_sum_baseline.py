"""
Test script for Baseline 1-layer GNN on Table 2 data.
This is the simple GNN from the paper that should perform poorly on star graphs.
"""
import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_table2_datasets
from utils import get_logger, AverageMeter, accuracy

# Import baseline GNN model

from model import SumTaskGNN, TORCH_GEOMETRIC_AVAILABLE

if not TORCH_GEOMETRIC_AVAILABLE:
    raise ImportError("PyTorch Geometric required for baseline GNN")

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool


class BaselineMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, bias=False, init_std=1e-2):
        super(BaselineMLP, self).__init__()
        self.dropout = dropout
        
        self.lins = torch.nn.ModuleList()
        # Input layer
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels, bias=bias))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=bias))
            
        # Output layer
        self.out_lin = torch.nn.Linear(hidden_channels, out_channels, bias=bias)
        
        # Init
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=init_std)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, batch = data.x, data.batch
        
        # MLP on each node
        for lin in self.lins:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Sum pooling
        x = global_add_pool(x, batch)
        
        # Output project
        x = self.out_lin(x)
        return x


def args_parser():
    parser = argparse.ArgumentParser(description='Baseline GNN Table 2 Experiments')
    
    # Experiment settings
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
    
    # Model settings
    parser.add_argument('--model', type=str, default='mlp', choices=['gnn', 'mlp'],
                        help='Model type (gnn or mlp)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Use bias in layers')
    parser.add_argument('--init-std', type=float, default=1e-2,
                        help='Initialization std (Xavier gain)')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (paper uses 0.1)')
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
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def convert_to_pyg_data(datasets):
    """Convert datasets to PyTorch Geometric format."""
    from scipy.sparse import coo_matrix
    
    def process_split(split_data):
        data_list = []
        for i in range(len(split_data['adj'])):
            # Convert adjacency to edge_index
            adj_coo = coo_matrix(split_data['adj'][i])
            edge_index = torch.tensor(
                np.vstack([adj_coo.row, adj_coo.col]), 
                dtype=torch.long
            )
            
            # Convert features
            if hasattr(split_data['features'][i], 'toarray'):
                x = torch.tensor(
                    split_data['features'][i].toarray(), 
                    dtype=torch.float
                )
            else:
                x = torch.tensor(
                    split_data['features'][i], 
                    dtype=torch.float
                )
            
            # Get label
            y = torch.tensor([split_data['labels'][i]], dtype=torch.long)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        return data_list
    
    # Convert training data
    train_data = process_split(datasets['train'])
    
    # Convert test data
    test_data = {}
    for test_name, test_split in datasets['test'].items():
        test_data[test_name] = process_split(test_split)
    
    return train_data, test_data


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        output = model(batch)
        loss = F.cross_entropy(output, batch.y)
        
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), batch.y.size(0))
        train_acc.update(accuracy(output.data, batch.y.data), batch.y.size(0))
    
    return train_loss.avg, train_acc.avg


def test_epoch(model, loader, device):
    """Test for one epoch."""
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = F.cross_entropy(output, batch.y)
            
            test_loss.update(loss.item(), batch.y.size(0))
            test_acc.update(accuracy(output.data, batch.y.data), batch.y.size(0))
    
    return test_loss.avg, test_acc.avg


def run_single_experiment(args, datasets, run_idx, logger):
    """Run single experiment (one random seed)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Convert to PyG format
    logger.info("Converting data to PyTorch Geometric format...")
    train_data, test_data = convert_to_pyg_data(datasets)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loaders = {
        name: DataLoader(data, batch_size=args.batch_size, shuffle=False)
        for name, data in test_data.items()
    }
    
    # Create model
    logger.info(f"Creating {args.num_layers}-layer {args.model.upper()} model...")
    
    if args.model == 'gnn':
        model = SumTaskGNN(
            in_channels=args.feature_dim,
            hidden_channels=args.hidden_dim,
            out_channels=2,  # Binary classification
            num_layers=args.num_layers,
            dropout=args.dropout,
            bias=args.bias,
            init_std=args.init_std
        ).to(device)
    else:
        model = BaselineMLP(
            in_channels=args.feature_dim,
            hidden_channels=args.hidden_dim,
            out_channels=2,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bias=args.bias,
            init_std=args.init_std
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params}")
    
    # Setup optimizer (SGD with lr=0.1 as in paper)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    # Training loop
    best_train_acc = 0
    test_results = {}
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        best_train_acc = max(best_train_acc, train_acc)
        
        # Log
        log_str = (f"Epoch {epoch:03d} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f}")
        
        # Test periodically
        if epoch % args.test_freq == 0 or epoch == args.epochs:
            test_results_epoch = {}
            for test_name, test_loader in test_loaders.items():
                test_loss, test_acc = test_epoch(model, test_loader, device)
                test_results_epoch[test_name] = test_acc
                
                if args.verbose:
                    log_str += f" | {test_name}: {test_acc:.4f}"
            
            # Store final results
            if epoch == args.epochs:
                test_results = test_results_epoch
        
        logger.info(log_str)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return test_results


def main():
    args = args_parser()
    
    # Create save directory
    model_name = f'baseline_gnn_{args.num_layers}layer' if args.model == 'gnn' else f'baseline_mlp_{args.num_layers}layer'
    save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logger
    import logging
    logger = get_logger(save_dir, __name__, 
                       f'{model_name}.log',
                       level=getattr(logging, args.log_level))
    
    logger.info("=" * 80)
    logger.info(f"Baseline {args.num_layers}-layer {args.model.upper()} - Table 2 Experiment")
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
    results_file = os.path.join(save_dir, f'results_{model_name}.json')
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
