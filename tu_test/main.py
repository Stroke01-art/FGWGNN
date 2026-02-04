import os 
import json
import time
import argparse

import numpy as np
from math import ceil
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from synthetic_graph_utils import *
from utils import *
from model import KerGNN

# Argument parser
def args_parser():
    parser = argparse.ArgumentParser(description='KerGNNs')

    parser.add_argument('--kernel', default='rw', choices=['rw', 'fgw', 'asfgw'], help='the kernel type')
    parser.add_argument('--iter', type=int, default=0, help='the index of fold in 10-fold validation,0-9')
    parser.add_argument('--lr', type=float, default=1e-2,  help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=64,  help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,  help='number of epochs to train')
    parser.add_argument('--test_freq', type=int, default=5,  help='frequency of evaluation on test dataset')
    parser.add_argument('--log_level', default='INFO', help='the level of log file')
    parser.add_argument('--dataset', default='IMDB-BINARY', help='dataset name')
    parser.add_argument('--use_node_labels', action='store_true', default=False, help='whether to use node labels')
    parser.add_argument('--use_node_attri', action='store_true', default=False, help='whether to use node attributes')
    parser.add_argument('--prefix', default='prefix', help='prefix of the file name for record')
    
    parser.add_argument('--k', type=int, default=1, help='use k-hop neighborhood to construct the subgraph ')
    parser.add_argument('--size_subgraph', type=int, default=8, help='size of the subgraph')
    parser.add_argument('--hidden_dims',  nargs='+', type=int, default=[64,32], help='size of hidden layer of NN')
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of MLP layers')
    parser.add_argument('--mlp_hidden_dim', type=int, default=16, help='hiddem dimension of MLP layers')
    parser.add_argument('--size_graph_filter', nargs='+', type=int, default=[8], help='number of hidden graph nodes at each layer')
    parser.add_argument('--max_step', type=int, default=1, help='max length of random walks')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout rate (1 - keep probability).')
    parser.add_argument('--no_norm', action='store_true', default=False, help='whether to apply normalization')
    parser.add_argument('--use_local_alpha', action='store_true', default=False, help='use per-patch alpha (FGW)')
    parser.add_argument('--alpha_hidden_dim', type=int, default=16, help='hidden dim for alpha MLP in FGW')
    
    # ASFGW parameters
    parser.add_argument('--num_slices', type=int, default=32, help='number of slices for ASFGW')
    parser.add_argument('--alpha_init', type=float, default=0.0, help='initial alpha logit (0.0 means alpha=0.5)')
    parser.add_argument('--w_init', type=float, default=0.0, help='initial w logit (0.0 means w=0.5)')
    
    args = parser.parse_args()
    return args

def load_data_robust(args):
    """
    Robust data loading:
    1. Try local utils.load_data
    2. If fails (FileNotFound), try PyTorch Geometric TUDataset (downloading if needed)
    """
    try:
        print(f"Attempting local load for {args.dataset}...")
        return load_data(args.dataset, args.use_node_labels, use_node_attri=args.use_node_attri)
    except Exception as e:
        print(f"Local load failed: {e}. Attempting download via torch_geometric...")
        try:
            from torch_geometric.datasets import TUDataset
            from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
            
            path = os.path.join(".", "datasets", "TUDataset")
            dataset = TUDataset(path, name=args.dataset, use_node_attr=args.use_node_attri)
            
            adj_lst = []
            features_lst = []
            class_labels = []
            
            print(f"Loaded {len(dataset)} graphs from TUDataset.")
            
            for data in dataset:
                # Edge Index to Sparse
                num_nodes = data.num_nodes
                # Convert to scipy sparse matrix
                adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
                adj_lst.append(adj)
                
                # Features
                if data.x is not None:
                    features_lst.append(data.x.numpy())
                else:
                    # Fallback features (e.g. constant or degree) - utils.load_data does this locally too
                    # If utils.load_data uses one-hot degrees, we might need to mimic.
                    # Simple fallback: Degree or constant
                    features_lst.append(np.ones((num_nodes, 1)))

                class_labels.append(data.y.item())
                
            return adj_lst, features_lst, np.array(class_labels)
            
        except ImportError:
            print("torch_geometric not installed. Cannot download dataset.")
            raise e
        except Exception as e2:
            print(f"torch_geometric load failed: {e2}")
            raise e

def main():
    args = args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # create catalog
    save_folder = os.path.join("save", args.dataset, "cv{}".format(args.iter))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    log_name =args.prefix + '_'+ args.dataset + '_{}'.format(args.iter) + '.log'
    logger = get_logger(save_folder, __name__, log_name, level=args.log_level)
    
    if args.dataset == 'synthetic':
        adj_lst, features_lst, class_labels = generate_synthetic_graphs(
        num_graphs=500,
        num_nodes_range=(10, 25),
        p_edge=0.25,
        feat_dim=1
    )
    else:
        # Robust loading
        adj_lst, features_lst, class_labels = load_data_robust(args)
                                                                      
    N = len(adj_lst)
    features_dim = features_lst[0].shape[1]
    print("graph number", len(features_lst))
    logger.info("in dim: {}".format(features_dim))

    enc = LabelEncoder()
    class_labels = enc.fit_transform(class_labels)
    n_classes = np.unique(class_labels).size
    y = [np.array(class_labels[i]) for i in range(class_labels.size)]

    if args.dataset == 'synthetic':
        # split train/val/test data 60/20/20 
        idx = np.arange(len(class_labels))
        train_index, test_index = train_test_split(idx, test_size=0.2, stratify=class_labels, random_state=42)
        train_index, val_index = train_test_split(train_index, test_size=0.25, stratify=class_labels[train_index], random_state=42)
    else: 
        try:
            splits = json.load(open('datasets/data_splits/{}_splits.json'.format(args.dataset), "r"))
            test_index = splits[args.iter]['test']
            train_index = splits[args.iter]['model_selection'][0]['train']
            val_index = splits[args.iter]['model_selection'][0]['validation']
        except Exception as e:
            print(f"Warning: Could not load split file ({e}). Using random split.")
            idx = np.arange(len(class_labels))
            train_index, test_index = train_test_split(idx, test_size=0.1, stratify=class_labels, random_state=42)
            train_index, val_index = train_test_split(train_index, test_size=0.1, stratify=class_labels[train_index], random_state=42)


    n_test = len(test_index)
    n_train = len(train_index)
    n_val = len(val_index)

    # Sampling
    adj_train = [adj_lst[i] for i in train_index]
    features_train = [features_lst[i] for i in train_index]
    y_train = [y[i] for i in train_index]

    adj_test = [adj_lst[i] for i in test_index]
    features_test = [features_lst[i] for i in test_index]
    y_test = [y[i] for i in test_index]

    adj_val = [adj_lst[i] for i in val_index]
    features_val = [features_lst[i] for i in val_index]
    y_val = [y[i] for i in val_index]

    # Create batches
    adj_test, features_test, graph_indicator_test, y_test = generate_batches(adj_test, features_test, y_test, args.batch_size, device, use_shortest_path=True)
    adj_train, features_train, graph_indicator_train, y_train = generate_batches(adj_train, features_train, y_train, args.batch_size, device, use_shortest_path=True)
    adj_val, features_val, graph_indicator_val, y_val = generate_batches(adj_val, features_val, y_val, args.batch_size, device, use_shortest_path=True)
    
    subadj_test, subidx_test = generate_sub_features_idx(adj_test, features_test,size_subgraph = args.size_subgraph, k_neighbor=args.k)
    subadj_train, subidx_train = generate_sub_features_idx(adj_train, features_train,size_subgraph = args.size_subgraph, k_neighbor=args.k)
    subadj_val, subidx_val = generate_sub_features_idx(adj_val, features_val,size_subgraph = args.size_subgraph, k_neighbor=args.k)
    
    n_test_batches = ceil(n_test/args.batch_size)
    n_train_batches = ceil(n_train/args.batch_size)
    n_val_batches = ceil(n_val/args.batch_size)

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
    
    # Record model
    logger.info(model)

    # set up training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    def train(epoch, adj, features, idxs, graph_indicator, y):
        optimizer.zero_grad()
        output = model(adj, features, idxs, graph_indicator)
        loss_train = F.cross_entropy(output, y)
        loss_train.backward()
        optimizer.step()
        return output, loss_train

    def test(adj, features, idxs, graph_indicator, y):
        output = model(adj, features, idxs, graph_indicator)
        loss_test = F.cross_entropy(output, y)
        return output, loss_test

    best_val_acc = 0
    best_test_acc = 0
    if os.path.exists((os.path.join(save_folder,'{}_val_model.pth.tar'.format(args.prefix)))):
        os.remove((os.path.join(save_folder,'{}_val_model.pth.tar'.format(args.prefix))))

    for epoch in range(args.epochs):
        start = time.time()
        # Train for one epoch
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        for i in range(n_train_batches):
            output, loss = train(epoch, subadj_train[i], features_train[i],subidx_train[i], graph_indicator_train[i], y_train[i])
            train_loss.update(loss.item(), output.size(0))
            train_acc.update(accuracy(output.data, y_train[i].data), output.size(0))

        if torch.cuda.is_available():        
            torch.cuda.empty_cache()

        # Evaluate on validation set
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        for i in range(n_val_batches):
            output, loss = test(subadj_val[i], features_val[i], subidx_val[i], graph_indicator_val[i], y_val[i])
            val_loss.update(loss.item(), output.size(0))
            val_acc.update(accuracy(output.data, y_val[i].data), output.size(0))
        torch.cuda.empty_cache()
        log_str = "epoch:" + '%03d ' % (epoch + 1) + "train_loss="+ "{:.5f} ".format(train_loss.avg)+ "train_acc="+ "{:.5f} ".format(train_acc.avg) + "val_acc="+ "{:.5f} ".format(val_acc.avg) + "time="+ "{:.5f} ".format(time.time() - start)
        if torch.cuda.is_available():       
            torch.cuda.empty_cache()

        # Evaluate on test set
        if epoch % args.test_freq == 0: 
            test_loss = AverageMeter()
            test_acc = AverageMeter()
            for i in range(n_test_batches):
                output, loss = test(subadj_test[i], features_test[i], subidx_test[i], graph_indicator_test[i], y_test[i])
                test_loss.update(loss.item(), output.size(0))
                test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
            log_str += "test_acc="+ "{:.5f} ".format(test_acc.avg)
        
        torch.cuda.empty_cache()

        logger.info(log_str)
        scheduler.step()
        
        # Remember best accuracy and save checkpoint
        is_val_best = val_acc.avg >= best_val_acc
        is_test_best = test_acc.avg >= best_test_acc
        best_val_acc = max(val_acc.avg, best_val_acc)
        best_test_acc = max(test_acc.avg, best_test_acc)

        if is_val_best:
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, os.path.join(save_folder,'{}_val_model.pth.tar'.format(args.prefix)) )

    logger.info("best_val_acc="+ "{:.5f}".format(best_val_acc))
    logger.info("best_test_acc="+ "{:.5f}".format(best_test_acc))
    # Print results
    print("Loading best val model and evaluating on test set...")

    # load val checkpoint to infer on test dataset
    val_checkpoint = torch.load(os.path.join(save_folder,'{}_val_model.pth.tar'.format(args.prefix)) )
    epoch = val_checkpoint['epoch']
    model.load_state_dict(val_checkpoint['state_dict'])
    optimizer.load_state_dict(val_checkpoint['optimizer'])

    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    for i in range(n_test_batches):
        output, loss = test(subadj_test[i], features_test[i], subidx_test[i], graph_indicator_test[i], y_test[i])
        test_loss.update(loss.item(), output.size(0))
        test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
    logger.info("best val model on test dataset acc="+ "{:.5f}".format(test_acc.avg))


if __name__ == "__main__":
    main()
