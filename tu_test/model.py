import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from GW_utils import *
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

try:
    from torch_geometric.nn import GraphConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. Baseline GNN will not work.")

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

    def forward(self, adj, features, idxs, device):
        if adj.max() > 1:
            adj = (adj == 1).float()
            
        adj_hidden_norm = torch.zeros( self.size_graph_filter, self.size_graph_filter, self.out_dim).to(device)
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1)
        adj_hidden_norm[idx[0], idx[1], :] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        
        x = features
        if self.hidden_dim:
            x = nn.ReLU()(self.fc_in(x))
        x = torch.cat([x,torch.zeros(1, x.shape[1]).to(device)])
        x = x[idxs]
        
        z = self.features_hidden

        zx = torch.einsum("mcn,abc->ambn", (z, x))
        out = []
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_graph_filter, device=device)             
                o = torch.einsum("ab,bcd->acd", (eye, z))
                t = torch.einsum("mcn,abc->ambn", (o, x))
            else:
                x = torch.einsum("abc,acd->abd",(adj, x))
                z = torch.einsum("abd,bcd->acd", (adj_hidden_norm, z))
                t = torch.einsum("mcn,abc->ambn", (z, x))
            t = self.dropout(t) 
            t = torch.mul(zx, t)
            t = torch.mean(t, dim=[1,2])
            out.append(t)
        out = sum(out)/len(out)
        return out

class FGW_Layer(nn.Module):
    def __init__(self, input_dim, out_dim, size_subgraph=10, size_graph_filter=10, hidden_dim=None, dropout=0.5, use_local_alpha=True, alpha_hidden_dim=32, gamma = 0.1):
        super(FGW_Layer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.size_graph_filter = size_graph_filter
        self.hidden_dim = hidden_dim
        
        self.C_filter_uptriangle = Parameter(torch.FloatTensor((size_graph_filter * (size_graph_filter - 1)) // 2, out_dim))
        if hidden_dim:
            self.fc_in = nn.Linear(input_dim, hidden_dim)
            self.F_filter = Parameter(torch.FloatTensor(size_graph_filter, hidden_dim, out_dim))
            self.ln_x = nn.LayerNorm(hidden_dim)
        else:
            self.F_filter = Parameter(torch.FloatTensor(size_graph_filter, input_dim, out_dim))
            self.ln_x = nn.LayerNorm(input_dim)
        
        self.alpha_raw = Parameter(torch.ones(1))
        
        self.use_local_alpha = use_local_alpha
        self.alpha_hidden_dim = alpha_hidden_dim
        
        alpha_in_dim = hidden_dim if hidden_dim is not None else input_dim
        
        if self.use_local_alpha:
            self.alpha_node = nn.Sequential(
                nn.Linear(alpha_in_dim, alpha_hidden_dim),
                nn.ReLU(),
                nn.Linear(alpha_hidden_dim, alpha_hidden_dim),
            )
            self.alpha_proto = nn.Sequential(
                nn.Linear(alpha_in_dim, alpha_hidden_dim),
                nn.ReLU(),
                nn.Linear(alpha_hidden_dim, alpha_hidden_dim),
            )
        
        self.gamma_param = Parameter(torch.tensor(gamma))
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.C_filter_uptriangle.data.uniform_(-1, 1)
        self.F_filter.data.uniform_(0, 1)
        self.alpha_raw.data.fill_(0.0)
        
        if self.use_local_alpha:
            for module in self.alpha_node:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            for module in self.alpha_proto:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
    def forward(self, adj, features, idxs, device):
        C_filter = torch.zeros(self.size_graph_filter, self.size_graph_filter, self.out_dim).to(device)
        idx = torch.triu_indices(self.size_graph_filter, self.size_graph_filter, 1, device=device)
        C_filter[idx[0], idx[1], :] = self.relu(self.C_filter_uptriangle)
        C_filter = C_filter + torch.transpose(C_filter, 0, 1)
        
        x = features
        if self.hidden_dim:
            x = self.relu(self.fc_in(x))
        x = self.ln_x(x)
        x = torch.cat([x, torch.zeros(1, x.shape[1]).to(device)])
        x = x[idxs]

        batch_size = x.shape[0]
        N_sub = x.shape[1]
        N_filter = self.size_graph_filter
        N_all = features.size(0)
        pad_mask = (idxs == N_all).float()
        vaild_mask = 1.0 - pad_mask

        h_input_batch = torch.ones(batch_size, N_sub, device=device) / N_sub
        h_input_batch = h_input_batch * vaild_mask
        h_input_batch = h_input_batch / (h_input_batch.sum(dim=1, keepdim=True) + 1e-9)
        
        h_filter_batch = torch.ones(batch_size, N_filter, device=device) / N_filter

        if self.use_local_alpha:
            z_node = self.alpha_node(x)
            z_proto = self.alpha_proto(self.F_filter.mean(dim=2))
            local_logit = torch.einsum("bnd,pd->bn p", (z_node, z_proto))
            global_logit = self.alpha_raw.view(1,1,1)
            alpha_tensor = torch.sigmoid(local_logit + global_logit)
        else:
            alpha_tensor = torch.sigmoid(self.alpha_raw)

        fgw_dists = entropic_fused_gromov_wasserstein(
            C_filter=C_filter, F_filter=self.F_filter, h_filter=h_filter_batch,
            C_sub=adj, F_sub=x, h_sub=h_input_batch,
            alpha=alpha_tensor, 
            epsilon=0.05,
            gw_iter=2,
            sinkhorn_iter=3,
        )

        similarity_matrix = torch.exp(-torch.abs(self.gamma_param)*fgw_dists)
        return similarity_matrix

class ASFGW_Layer(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        size_subgraph=10,
        hidden_dim=None,
        num_slices=32,
        dropout=0.5,
        gamma=0.01,
        alpha_init=0.0,
        use_local_alpha=True,    
        alpha_hidden_dim=32,
        w_init=0.0,
        w_hidden_dim=32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.m = size_subgraph   
        self.n_neigh = self.m - 1
        self.L = num_slices
        
        self.dx = hidden_dim if hidden_dim is not None else input_dim
        self.ds = self.n_neigh
        
        self.x_lin = nn.Linear(input_dim, self.dx)
        self.x_ln = nn.LayerNorm(self.dx)
        self.s_ln = nn.LayerNorm(self.ds)

        self.theta_x = Parameter(torch.randn(self.L, self.dx))
        self.theta_s = Parameter(torch.randn(self.L, self.ds))
        
        if str(alpha_init) == "uniform":
            self.alpha_raw = Parameter(torch.rand(1))
        elif str(alpha_init) == "fixed":
            self.alpha_raw = Parameter(torch.tensor(0.5))
        else:
            self.alpha_raw = Parameter(torch.tensor(float(alpha_init)))
        if use_local_alpha:
            self.alpha_net = nn.Sequential(
                nn.Linear(self.dx, alpha_hidden_dim),
                nn.ReLU(),
                nn.Linear(alpha_hidden_dim, 1)
            )
            
        self.w_net = nn.Sequential(
            nn.Linear(self.dx * 2, w_hidden_dim),
            nn.ReLU(),
            nn.Linear(w_hidden_dim, 1)
        )
        if str(w_init) == "uniform":
            self.w_raw_init = Parameter(torch.rand(1))
        elif str(w_init) == "zero":
             self.w_raw_init = Parameter(torch.tensor(0.0))
        else:
            self.w_raw_init = Parameter(torch.tensor(float(w_init)))

        self.proto_root_raw = Parameter(torch.zeros(out_dim, input_dim))
        self.proto_neigh_raw = Parameter(torch.zeros(out_dim, self.n_neigh, input_dim))
        self.proto_dist_radial = Parameter(torch.zeros(out_dim, self.n_neigh))
        
        num_params_neigh = (self.n_neigh * (self.n_neigh - 1)) // 2
        self.proto_dist_neigh = Parameter(torch.zeros(num_params_neigh, out_dim))

        self.log_gamma = Parameter(torch.log(torch.tensor(float(gamma))))
        
        self.dropout = nn.Dropout(dropout)
        self.is_initialized = False
        self._reset_parameters()

    def _init_slicers(self, device):
        dim_x = self.dx
        slices_x = []
        for _ in range((self.L // dim_x) + 1):
            mat = torch.randn(dim_x, dim_x, device=device)
            q, _ = torch.linalg.qr(mat)
            slices_x.append(q.t())
        self.theta_x.data.copy_(torch.cat(slices_x, dim=0)[:self.L])

        dim_s = self.ds
        slices_s = []
        for _ in range((self.L // dim_s) + 1):
            mat = torch.randn(dim_s, dim_s, device=device)
            q, _ = torch.linalg.qr(mat)
            slices_s.append(q.t())
        self.theta_s.data.copy_(torch.cat(slices_s, dim=0)[:self.L])

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.x_lin.weight)
        nn.init.zeros_(self.x_lin.bias)
        self._init_slicers(self.theta_x.device)
        nn.init.uniform_(self.proto_dist_neigh, -0.5, 0.5)
        nn.init.uniform_(self.proto_dist_radial, 0.0, 1.0) 
        if hasattr(self, 'alpha_net'):
            for m in self.alpha_net:
                if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
        for m in self.w_net:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def init_from_data(self, train_adj_batch, train_features_batch, device):
        if self.is_initialized: return
        K_samples = train_adj_batch.shape[0]
        if K_samples < self.out_dim: return
        print(f"PointedASFGW: K-Means Init for {self.out_dim} prototypes...")
        with torch.no_grad():
            try:
                from sklearn.cluster import KMeans
                X_root_np = train_features_batch[:, 0, :].cpu().numpy()
                kmeans = KMeans(n_clusters=self.out_dim, n_init=10).fit(X_root_np)
                closest_indices = []
                for center in kmeans.cluster_centers_:
                    dists = ((X_root_np - center)**2).sum(axis=1)
                    closest_indices.append(np.argmin(dists))
                indices = torch.tensor(closest_indices, device=device)
            except Exception:
                indices = torch.randperm(K_samples, device=device)[:self.out_dim]

            self.proto_root_raw.data.copy_(train_features_batch[indices, 0, :])
            self.proto_neigh_raw.data.copy_(train_features_batch[indices, 1:, :])
            
            real_adj = train_adj_batch[indices].to(device)
            valid_mask = (real_adj.sum(dim=-1) > 0).float() 
            valid_mask[:, 0] = 1.0
            
            dists = self._compute_normalized_distance_matrix(real_adj, valid_mask)
            radial_dists = dists[:, 0, 1:]
            radial_dists, _ = torch.sort(radial_dists, dim=1)
            self.proto_dist_radial.data.copy_(radial_dists)
            
            neigh_dists = dists[:, 1:, 1:]
            idx_i, idx_j = torch.triu_indices(self.n_neigh, self.n_neigh, 1, device=device)
            triu_vals = neigh_dists[:, idx_i, idx_j].t()
            triu_vals = torch.clamp(triu_vals, 0.01, 0.99)
            self.proto_dist_neigh.data.copy_(torch.logit(triu_vals))
            self.is_initialized = True

    def _compute_normalized_distance_matrix(self, adj, mask):
        B, m, _ = adj.shape
        device = adj.device
        adj_bin = (adj > 1e-5).float()
        inf_val = float(m)
        dists = torch.ones(B, m, m, device=device) * inf_val
        eye = torch.eye(m, device=device).unsqueeze(0).expand(B, -1, -1)
        dists = dists.masked_fill(eye.bool(), 0.0)
        dists = dists.masked_fill(adj_bin.bool() & (dists == inf_val), 1.0)
        curr_adj = adj_bin
        for k in range(2, m):
            curr_adj = torch.bmm(curr_adj, adj_bin)
            reached = (curr_adj > 0)
            update_mask = reached & (dists == inf_val)
            dists = dists.masked_fill(update_mask, float(k))
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        dists = dists.masked_fill(mask_2d == 0, inf_val)
        return dists / float(m)

    def _get_proto_neigh_structure(self, device):
        K = self.out_dim
        M = self.n_neigh
        C = torch.zeros(K, M, M, device=device)
        idx_i, idx_j = torch.triu_indices(M, M, 1, device=device)
        vals = torch.sigmoid(self.proto_dist_neigh).t()
        C[:, idx_i, idx_j] = vals
        C = C + C.transpose(1, 2)
        return C

    def _compute_sw(self, z_batch, z_proto, theta, valid_mask):
        theta_norm = F.normalize(theta, dim=1)
        proj_b = torch.matmul(z_batch, theta_norm.t())
        proj_p = torch.matmul(z_proto, theta_norm.t())
        proj_b_sorted, sort_idx = torch.sort(proj_b, dim=1)
        proj_p_sorted, _ = torch.sort(proj_p, dim=1)
        
        valid_mask_L = valid_mask.unsqueeze(-1).expand(-1, -1, self.L)
        weights = torch.gather(valid_mask_L, 1, sort_idx)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        
        diff_sq = (proj_b_sorted.unsqueeze(1) - proj_p_sorted.unsqueeze(0)) ** 2
        dist = (diff_sq * weights.unsqueeze(1)).sum(dim=2).mean(dim=-1)
        return dist

    def _compute_radial_dist(self, rad_batch, rad_proto, valid_mask):
        rad_b_sorted, sort_idx = torch.sort(rad_batch, dim=1)
        rad_p_sorted, _ = torch.sort(rad_proto, dim=1)
        weights = torch.gather(valid_mask, 1, sort_idx)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        diff_sq = (rad_b_sorted.unsqueeze(1) - rad_p_sorted.unsqueeze(0)) ** 2
        dist = (diff_sq * weights.unsqueeze(1)).sum(dim=2)
        return dist

    def forward(self, adj, features, idxs, device):
        B, m, _ = adj.shape
        N_all = features.size(0)
        x_all = torch.cat([features, torch.zeros(1, features.size(1), device=device)], dim=0)
        x_patch = x_all[idxs]
        
        if self.training and not self.is_initialized and B >= self.out_dim:
            self.init_from_data(adj, x_patch, device)
            
        x_root = x_patch[:, 0, :]
        x_neigh = x_patch[:, 1:, :]
        neigh_idx = idxs[:, 1:]
        pad_mask_neigh = (neigh_idx == N_all)
        valid_mask_neigh = (~pad_mask_neigh).float()

        h_root = self.x_ln(self.x_lin(x_root))
        h_proto_root = self.x_ln(self.x_lin(self.proto_root_raw))
        d_root_feat = torch.cdist(h_root, h_proto_root, p=2) ** 2

        full_mask = torch.ones(B, m, device=device)
        full_mask[:, 1:] = valid_mask_neigh
        dists_full = self._compute_normalized_distance_matrix(adj, full_mask)
        rad_graph = dists_full[:, 0, 1:]
        rad_proto = self.proto_dist_radial
        d_radial_str = self._compute_radial_dist(rad_graph, rad_proto, valid_mask_neigh)

        h_neigh = self.x_ln(self.x_lin(x_neigh))
        h_neigh = self.dropout(h_neigh)
        h_proto_neigh = self.x_ln(self.x_lin(self.proto_neigh_raw))
        sw_neigh_feat = self._compute_sw(h_neigh, h_proto_neigh, self.theta_x, valid_mask_neigh)

        adj_neigh = dists_full[:, 1:, 1:] 
        hs_neigh, _ = torch.sort(adj_neigh, dim=1)
        hs_neigh = self.s_ln(hs_neigh)
        adj_proto_neigh = self._get_proto_neigh_structure(device)
        hs_proto_neigh, _ = torch.sort(adj_proto_neigh, dim=1)
        hs_proto_neigh = self.s_ln(hs_proto_neigh)
        sw_neigh_str = self._compute_sw(hs_neigh, hs_proto_neigh, self.theta_s, valid_mask_neigh)

        if hasattr(self, 'alpha_net'):
            h_pooled = (h_neigh * valid_mask_neigh.unsqueeze(-1)).sum(1) / (valid_mask_neigh.sum(1, keepdim=True) + 1e-9)
            alpha_logit = self.alpha_net(h_pooled)
            alpha = torch.sigmoid(self.alpha_raw + alpha_logit)
        else:
            alpha = torch.sigmoid(self.alpha_raw)
            
        h_root_exp = h_root.unsqueeze(1).expand(-1, self.out_dim, -1)
        h_proto_exp = h_proto_root.unsqueeze(0).expand(B, -1, -1)
        w_input = torch.cat([h_root_exp, h_proto_exp], dim=-1)
        w = torch.sigmoid(self.w_net(w_input) + self.w_raw_init).squeeze(-1)

        d_neigh_context = (1 - alpha) * sw_neigh_feat + alpha * sw_neigh_str
        d_radial_term = alpha * d_radial_str
        d_root_term = (1 - alpha) * d_root_feat
        
        final_dist = (w ** 2) * d_root_term + 2 * w * (1 - w) * d_radial_term + ((1 - w) ** 2) * d_neigh_context
        gamma = torch.exp(self.log_gamma)
        similarity_matrix = torch.exp(-gamma * final_dist)
        return similarity_matrix

class KerGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, num_layers=1, kernel='rw', max_step=1,
                 size_graph_filter=10, size_subgraph=10,
                 num_mlp_layers=1, mlp_hidden_dim=None, dropout_rate=0.5, no_norm=False, use_node_norm=True,
                 use_local_alpha=False, alpha_hidden_dim=32,
                 num_slices=32, rwse_steps=8, struct_dim=None,
                 alpha_init=0.0, w_init=0.0):
        super(KerGNN, self).__init__()

        self.num_layers = num_layers
        self.kernel = kernel
        self.dropout_rate = dropout_rate
        self.no_norm = no_norm
        self.use_node_norm = use_node_norm

        if hidden_dims is None:
            hidden_dims = [input_dim, 64, 64]

        self.ker_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.node_batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                if kernel == 'rw':
                    self.ker_layers.append(
                        RW_layer(input_dim, hidden_dims[1], hidden_dim=hidden_dims[0],
                                 max_step=max_step, size_subgraph=size_subgraph,
                                 size_graph_filter=size_graph_filter[0], dropout=dropout_rate)
                    )
                elif kernel == 'fgw':
                    self.ker_layers.append(
                        FGW_Layer(input_dim, hidden_dims[1], hidden_dim=hidden_dims[0],
                                  size_subgraph=size_subgraph, size_graph_filter=size_graph_filter[0],
                                  dropout=dropout_rate, use_local_alpha=use_local_alpha,
                                  alpha_hidden_dim=alpha_hidden_dim)
                    )
                elif kernel == 'asfgw':
                    self.ker_layers.append(
                        ASFGW_Layer(input_dim, hidden_dims[1], hidden_dim=hidden_dims[0],
                                   size_subgraph=size_subgraph,
                                   dropout=dropout_rate,
                                   use_local_alpha=use_local_alpha,
                                   alpha_hidden_dim=alpha_hidden_dim,
                                   num_slices=num_slices,
                                   alpha_init=alpha_init,
                                   w_init=w_init)
                    )
                else:
                    exit('Error: unrecognized model')

                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[1]))

                if use_node_norm:
                    self.node_batch_norms.append(nn.BatchNorm1d(input_dim))
            else:
                if kernel == 'rw':
                    self.ker_layers.append(
                        RW_layer(hidden_dims[layer], hidden_dims[layer+1], hidden_dim=None,
                                 max_step=max_step, size_subgraph=size_subgraph,
                                 size_graph_filter=size_graph_filter[layer], dropout=dropout_rate)
                    )
                elif kernel == 'fgw':
                    self.ker_layers.append(
                        FGW_Layer(hidden_dims[layer], hidden_dims[layer+1], hidden_dim=None,
                                  size_subgraph=size_subgraph, size_graph_filter=size_graph_filter[layer],
                                  dropout=dropout_rate, use_local_alpha=use_local_alpha,
                                  alpha_hidden_dim=alpha_hidden_dim)
                    )
                elif kernel == 'asfgw':
                    self.ker_layers.append(
                        ASFGW_Layer(hidden_dims[layer], hidden_dims[layer+1], hidden_dim=None,
                                   size_subgraph=size_subgraph,
                                   dropout=dropout_rate,
                                   use_local_alpha=use_local_alpha,
                                   alpha_hidden_dim=alpha_hidden_dim,
                                   num_slices=num_slices,
                                   alpha_init=alpha_init,
                                   w_init=w_init)
                    )
                else:
                    exit('Error: unrecognized model')

                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[layer+1]))

                if use_node_norm:
                    self.node_batch_norms.append(nn.BatchNorm1d(hidden_dims[layer]))

        self.linears_prediction = nn.ModuleList()
        for layer in range(self.num_layers + 1):
            if layer == 0:
                self.linears_prediction.append(MLP(num_mlp_layers, input_dim, mlp_hidden_dim, output_dim))
            else:
                self.linears_prediction.append(MLP(num_mlp_layers, hidden_dims[layer], mlp_hidden_dim, output_dim))
    
    def forward(self, adj, features, idxs, graph_indicator):
        device = features.device
        adj = adj.to(device)
        idxs = idxs.to(device)
        graph_indicator = graph_indicator.to(device)
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.size(0)

        hidden_rep = [features]

        for layer in range(self.num_layers):
            h = hidden_rep[layer]
            if self.use_node_norm:
                h = self.node_batch_norms[layer](h)

            h = self.ker_layers[layer](adj, h, idxs, device)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.zeros(n_graphs, h.shape[1], device=device).index_add_(0, graph_indicator, h)
            
            if not self.no_norm:
                norm = counts.unsqueeze(1).repeat(1, pooled_h.shape[1])
                pooled_h = pooled_h / norm

            score_over_layer += F.dropout(
                self.linears_prediction[layer](pooled_h),
                self.dropout_rate,
                training=self.training
            )

        return score_over_layer
