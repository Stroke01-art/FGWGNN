import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from math import ceil
from sklearn.model_selection import StratifiedKFold
import requests
import zipfile
import io
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from model import KerGNN
from utils import generate_batches, generate_sub_features_idx, accuracy, AverageMeter, get_logger

def args_parser():
    parser = argparse.ArgumentParser(description='KerGNNs TU Dataset Test')
    
    parser.add_argument('--dataset', default='PROTEINS', help='dataset name')
    parser.add_argument('--kernel', default='fgw', choices=['rw', 'fgw', 'asfgw'], help='the kernel type')
    parser.add_argument('--num_slices', type=int, default=16, help='number of slices for ASFGW')
    parser.add_argument('--alpha_init', type=str, default='uniform', help='initialization for alpha')
    parser.add_argument('--w_init', type=str, default='uniform', help='initialization for weights')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[4, 8], help='hidden dimensions')
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of MLP layers')
    parser.add_argument('--mlp_hidden_dim', type=int, default=None, help='hidden dim of MLP')
    parser.add_argument('--size_graph_filter', nargs='+', type=int, default=[4], help='graph filter size')
    parser.add_argument('--max_step', type=int, default=1, help='random walk steps')
    parser.add_argument('--size_subgraph', type=int, default=4, help='subgraph size')
    parser.add_argument('--k', type=int, default=4, help='k-hop neighborhood')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout rate')
    # Use standard store_true convention: default is False (normalization enabled)
    parser.add_argument('--no_norm', action='store_true', default=True, help='disable normalization')
    parser.add_argument('--use_local_alpha', action='store_true', default=True, help='use local alpha for FGW')
    parser.add_argument('--alpha_hidden_dim', type=int, default=16, help='alpha MLP hidden dim')
    parser.add_argument('--log_level', default='INFO')
    parser.add_argument('--prefix', default='tu_test')
    parser.add_argument('--folds', type=int, default=10, help='number of folds for CV')
    
    return parser.parse_args()

def download_dataset(root, name):
    url = f"https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{name}.zip"
    print(f"Downloading {name} from {url}...")
    try:
        r = requests.get(url)
        if r.status_code != 200:
             raise Exception(f"HTTP {r.status_code}")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(root)
        print(f"Downloaded and extracted {name} to {root}")
    except Exception as e:
        print(f"Failed to download/extract {name}: {e}")
        # Try chrsmrrs mirror if original fails
        url2 = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
        print(f"Trying mirror {url2}...")
        try:
             r = requests.get(url2)
             if r.status_code != 200:
                  print(f"Mirror failed with HTTP {r.status_code}")
                  return
             z = zipfile.ZipFile(io.BytesIO(r.content))
             z.extractall(root)
             print(f"Downloaded and extracted {name} to {root}")
        except Exception as e2:
             print(f"Failed to download from mirror: {e2}")

def load_tu_dataset(name):
    # Fix for common typo NCII -> NCI1
    if name == "NCII":
        print("Warning: Dataset name 'NCII' detected. Assuming 'NCI1'.")
        name = "NCI1"

    root = "data"
    if not os.path.exists(root):
        os.makedirs(root)
        
    ds_dir = os.path.join(root, name)
    # Check for critical files
    if not os.path.exists(os.path.join(ds_dir, f"{name}_A.txt")):
        download_dataset(root, name)
    
    # Check again if file exists after download
    if not os.path.exists(os.path.join(ds_dir, f"{name}_A.txt")):
         raise FileNotFoundError(f"Dataset {name} could not be found or downloaded. Please check the dataset name.")

    # Manual parsing matching utils.load_data for consistency
    print(f"Loading {name} manually from {ds_dir}...")
    
    # Graph indicator
    graph_indicator = np.loadtxt(os.path.join(ds_dir, f"{name}_graph_indicator.txt"), dtype=np.int64)
    graph_labels = np.loadtxt(os.path.join(ds_dir, f"{name}_graph_labels.txt"), dtype=np.int64)
    
    # Adjacency
    edges = np.loadtxt(os.path.join(ds_dir, f"{name}_A.txt"), dtype=np.int64, delimiter=",")
    edges -= 1 # 1-based to 0-based
    
    num_nodes = graph_indicator.size
    # Create CSR matrix
    # Note: data needs to be coerced to format compatible with operations
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(num_nodes, num_nodes))
    
    # Features
    # Check for node labels
    node_labels_path = os.path.join(ds_dir, f"{name}_node_labels.txt")
    node_attrs_path = os.path.join(ds_dir, f"{name}_node_attributes.txt")
    
    if os.path.exists(node_labels_path):
        print("Using node labels.")
        x = np.loadtxt(node_labels_path, dtype=np.int64)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        enc = OneHotEncoder(sparse_output=False) # sparse_output for newer sklearn
        try:
             x = enc.fit_transform(x)
        except TypeError:
             enc = OneHotEncoder(sparse=False)
             x = enc.fit_transform(x)
    elif os.path.exists(node_attrs_path):
        print("Using node attributes.")
        x = np.loadtxt(node_attrs_path, delimiter=',', dtype=np.float64)
    else:
        print("Using node degrees as features.")
        x = A.sum(axis=1)
        
    # Split into list of graphs
    _, graph_size = np.unique(graph_indicator, return_counts=True)
    
    adj_lst = []
    features_lst = []
    
    idx = 0
    # Use A.todense() or keep sparse? model.py expects dense usually?
    # KerGNN generates subgraphs. If utils.py uses dense for subgraphs, we might need dense.
    # Looking at train(): model(subadj...) 
    # Let's ensure features are float32
    x = x.astype(np.float32)

    for i in range(graph_size.size):
        n_n = graph_size[i]
        # Extract submatrix
        adj = A[idx:idx+n_n, idx:idx+n_n]
        feat = x[idx:idx+n_n, :]
        
        # KerGNN expects scipy sparse matrix in adj_lst?
        # utils.generate_sub_features_idx takes adj (scipy sparse)
        adj_lst.append(adj)
        features_lst.append(feat)
        idx += n_n
        
    # Process Labels
    # TU dataset labels can be 1-based or 0-based, or -1/1
    # We should map them to 0..NumClasses-1
    unique_labels = np.unique(graph_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    class_labels = np.array([label_map[l] for l in graph_labels])
    n_classes = len(unique_labels)
    
    return adj_lst, features_lst, class_labels, n_classes

def train(model, optimizer, subadj, features, subidx, graph_indicator, y):
    optimizer.zero_grad()
    output = model(subadj, features, subidx, graph_indicator)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    # DEBUG: Check gradients
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    # print(f"Gradient norm: {total_norm}") 
    
    optimizer.step()
    return output, loss, total_norm

def evaluate(model, subadj, features, subidx, graph_indicator, y):
    output = model(subadj, features, subidx, graph_indicator)
    loss = F.cross_entropy(output, y)
    return output, loss

def run_fold(args, fold_idx, train_idx, test_idx, adj_lst, features_lst, y, n_classes, device, logger):
    logger.info(f"--- Fold {fold_idx+1}/{args.folds} ---")
    
    # Prepare data for this fold
    adj_train = [adj_lst[i] for i in train_idx]
    features_train = [features_lst[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    
    adj_test = [adj_lst[i] for i in test_idx]
    features_test = [features_lst[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    
    # In TU standard, we often use a portion of train for validation, or just report test on best epoch
    # Here we'll take 10% of train for validation (model selection)
    train_idx_inner, val_idx_inner = next(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(adj_train, y_train))
    
    adj_val = [adj_train[i] for i in val_idx_inner]
    features_val = [features_train[i] for i in val_idx_inner]
    y_val = [y_train[i] for i in val_idx_inner]
    
    adj_train_final = [adj_train[i] for i in train_idx_inner]
    features_train_final = [features_train[i] for i in train_idx_inner]
    y_train_final = [y_train[i] for i in train_idx_inner]
    
    # Generate batches
    use_shortest_path = True
    
    train_data = generate_batches(adj_train_final, features_train_final, y_train_final, args.batch_size, device, use_shortest_path=use_shortest_path)
    val_data = generate_batches(adj_val, features_val, y_val, args.batch_size, device, use_shortest_path=use_shortest_path)
    test_data = generate_batches(adj_test, features_test, y_test, args.batch_size, device, use_shortest_path=use_shortest_path)
    
    # Extract tuples
    adj_tr, feat_tr, gi_tr, y_tr = train_data
    adj_va, feat_va, gi_va, y_va = val_data
    adj_te, feat_te, gi_te, y_te = test_data
    
    # Subgraph generation
    subadj_tr, subidx_tr = generate_sub_features_idx(adj_tr, feat_tr, size_subgraph=args.size_subgraph, k_neighbor=args.k)
    subadj_va, subidx_va = generate_sub_features_idx(adj_va, feat_va, size_subgraph=args.size_subgraph, k_neighbor=args.k)
    subadj_te, subidx_te = generate_sub_features_idx(adj_te, feat_te, size_subgraph=args.size_subgraph, k_neighbor=args.k)
    
    # Model
    features_dim = features_lst[0].shape[1]
    model = KerGNN(
        input_dim=features_dim, output_dim=n_classes, hidden_dims=args.hidden_dims,
        kernel=args.kernel, size_subgraph=args.size_subgraph,
        size_graph_filter=args.size_graph_filter,
        num_mlp_layers=args.num_mlp_layers, mlp_hidden_dim=args.mlp_hidden_dim,
        dropout_rate=args.dropout_rate, no_norm=args.no_norm,
        max_step=args.max_step, use_local_alpha=args.use_local_alpha,
        alpha_hidden_dim=args.alpha_hidden_dim,
        num_slices=args.num_slices, 
        alpha_init=args.alpha_init, 
        w_init=args.w_init
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_val_acc = 0.0
    test_acc_at_best_val = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_avg = 0
        total_grad_norm = 0
        for i in range(len(adj_tr)):
            _, loss, grad_norm = train(model, optimizer, subadj_tr[i], feat_tr[i], subidx_tr[i], gi_tr[i], y_tr[i])
            train_loss_avg += loss.item()
            total_grad_norm += grad_norm
        train_loss_avg /= len(adj_tr)
        total_grad_norm /= len(adj_tr)

        # Validation
        model.eval()
        val_acc = AverageMeter()
        for i in range(len(adj_va)):
            out, _ = evaluate(model, subadj_va[i], feat_va[i], subidx_va[i], gi_va[i], y_va[i])
            val_acc.update(accuracy(out.data, y_va[i].data), out.size(0))
            
        # Test
        test_acc = AverageMeter()
        for i in range(len(adj_te)):
            out, _ = evaluate(model, subadj_te[i], feat_te[i], subidx_te[i], gi_te[i], y_te[i])
            test_acc.update(accuracy(out.data, y_te[i].data), out.size(0))
            
        scheduler.step()
        
        # Convert tensor metrics to float for comparison
        val_acc_val = val_acc.avg
        if isinstance(val_acc_val, torch.Tensor):
            val_acc_val = val_acc_val.item()
            
        test_acc_val = test_acc.avg
        if isinstance(test_acc_val, torch.Tensor):
            test_acc_val = test_acc_val.item()

        if val_acc_val >= best_val_acc:
            best_val_acc = val_acc_val
            test_acc_at_best_val = test_acc_val
            
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Loss {train_loss_avg:.4f} Grad {total_grad_norm:.4f} Val {val_acc_val:.4f} Test {test_acc_val:.4f}")
            
    logger.info(f"Fold {fold_idx+1} Result: Best Val {best_val_acc:.4f}, Test Acc {test_acc_at_best_val:.4f}")
    
    if isinstance(test_acc_at_best_val, torch.Tensor):
        return test_acc_at_best_val.item()
    return test_acc_at_best_val

def main():
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Save folder
    save_folder = os.path.join("save", args.dataset + "_tu_test")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    logger = get_logger(save_folder, __name__, "test_log.txt", level=args.log_level)
    
    # Load Data
    adj_lst, features_lst, class_labels, n_classes = load_tu_dataset(args.dataset)
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Num Graphs: {len(adj_lst)}")
    logger.info(f"Num Classes: {n_classes}")
    logger.info(f"Feature Dim: {features_lst[0].shape[1]}")
    logger.info(f"Kernel: {args.kernel}")
    logger.info(f"Normalization: {'Disabled' if args.no_norm else 'Enabled'}")
    
    # 10-Fold CV
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=123)
    
    acc_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(adj_lst, class_labels)):
        acc = run_fold(args, fold_idx, train_idx, test_idx, adj_lst, features_lst, class_labels, n_classes, device, logger)
        acc_results.append(acc)
        
    logger.info(f"Final Results ({args.folds}-fold CV):")
    # acc_results might contain floats or tensors, ensure conversion
    acc_results_float = []
    for a in acc_results:
        # Check if scalar tensor
        if isinstance(a, torch.Tensor):
            acc_results_float.append(a.item())
        else:
            acc_results_float.append(a)
    
    logger.info(f"Mean Accuracy: {np.mean(acc_results_float):.4f}")
    logger.info(f"Std Dev: {np.std(acc_results_float):.4f}")
    logger.info(f"Individual Scores: {acc_results_float}")

if __name__ == "__main__":
    main()
