"""
Solvers from the Python Optimal Transport library 
https://pythonot.github.io/ (version POT=0.8.1)
 
adapted for the integration of (Fused) Gromov-Wasserstein in TFGW layers
"""

import numpy as np
import ot
import torch
from torch import vmap
from ot.bregman import sinkhorn
from ot.utils import dist, UndefinedParameter, list_to_array
from ot.optim import cg
from ot.lp import emd_1d, emd
from ot.utils import check_random_state
from ot.backend import get_backend
from ot.gromov import init_matrix, gwloss, gwggrad



def fast_squared_euclidean(X, Y):
    """
    2D mode:
      X: (N, D)
      Y: (M, D)
      Returns: (N, M) squared Euclidean distances
    
    3D mode:
      X: (B, N, D)
      Y: (B, M, D)
      Returns: (B, N, M) squared Euclidean distances
    """
    if X.dim() == 2:
        # 2D mode: (N, D) x (M, D) -> (N, M)
        XX = (X * X).sum(-1, keepdim=True)  # (N, 1)
        YY = (Y * Y).sum(-1, keepdim=True).T  # (1, M)
        XY = X @ Y.T  # (N, M)
        return XX + YY - 2 * XY
    elif X.dim() == 3:
        # 3D mode: (B, N, D) x (B, M, D) -> (B, N, M)
        XX = (X * X).sum(-1, keepdim=True)  # (B, N, 1)
        YY = (Y * Y).sum(-1, keepdim=True)  # (B, M, 1)
        XY = torch.bmm(X, Y.transpose(-2, -1))  # (B, N, M)
        return XX + YY.transpose(-2, -1) - 2 * XY  # (B, N, M)
    else:
        raise ValueError(f"Input must be 2D or 3D, got {X.dim()}D")

#%%


def entropic_fused_gromov_wasserstein(
    C_filter, F_filter, h_filter, C_sub, F_sub, h_sub, alpha,
    epsilon=1e-2, gw_iter=2, sinkhorn_iter=5):
    """
    Computes the batched Fused Gromov-Wasserstein distance with COST NORMALIZATION.
    
    Args:
        alpha (float or tensor): trade-off parameter between feature cost and structure cost.
            Can be a scalar or a tensor broadcastable to shape (D_out, batch_size, N_sub, N_filter).
            When tensor, enables local/adaptive alpha values per filter-graph pair.
    """
    # 1. --- SETUP: Reshape inputs ---
    if C_sub.dim() == 2:
        C_sub = C_sub.unsqueeze(0)
        F_sub = F_sub.unsqueeze(0)
        h_sub = h_sub.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, N_sub, _ = C_sub.shape
    N_filter, D_feat, D_out = F_filter.shape
    device = C_sub.device

    # Expand subgraph tensors
    C1 = C_sub.unsqueeze(0).expand(D_out, -1, -1, -1)
    F1 = F_sub.unsqueeze(0).expand(D_out, -1, -1, -1)
    p = h_sub.unsqueeze(0).expand(D_out, -1, -1)

    # Permute filter tensors
    C2 = C_filter.permute(2, 0, 1)
    F2 = F_filter.permute(2, 0, 1)
    q = h_filter.unsqueeze(0).expand(D_out, -1, -1)

    # 2. --- NORMALIZATION STEP 1: Structure Matrices ---
    C1_max = C1.amax(dim=(-2, -1), keepdim=True)
    C1 = C1 / (C1_max + 1e-8)
    
    C2_max = C2.amax(dim=(-2, -1), keepdim=True)
    C2 = C2 / (C2_max + 1e-8)

    # 3. --- PRE-COMPUTATION: Feature distance matrix ---
    F1_reshaped = F1.reshape(D_out * batch_size, N_sub, D_feat)
    F2_expanded = F2.unsqueeze(1).expand(-1, batch_size, -1, -1).reshape(D_out * batch_size, N_filter, D_feat)
    
    M_f = fast_squared_euclidean(F1_reshaped, F2_expanded)
    M_f = M_f.reshape(D_out, batch_size, N_sub, N_filter)

    # 4. --- NORMALIZATION STEP 2: Feature Cost Matrix ---
    M_f_max = M_f.amax(dim=(-2, -1), keepdim=True)
    M_f = M_f / (M_f_max + 1e-8)

    # 5. --- CORE ALGORITHM: Iterative Entropic FGW Solver ---
    # Initialize transport plan
    T = torch.einsum('dbi,dbj->dbij', p, q)

    def batch_sinkhorn(cost_matrix, p_dist, q_dist, reg, num_iters):
        log_p = torch.log(p_dist + 1e-20)
        log_q = torch.log(q_dist + 1e-20)
        log_K = -cost_matrix / reg
        u = torch.zeros_like(p_dist, device=device)
        for _ in range(num_iters):
            v = log_q - torch.logsumexp(log_K + u.unsqueeze(-1), dim=2)
            u = log_p - torch.logsumexp(log_K + v.unsqueeze(2), dim=3)
        return torch.exp(u.unsqueeze(-1) + v.unsqueeze(2) + log_K)

    # Handle alpha: convert to tensor if needed and ensure proper broadcasting
    alpha_tensor = alpha
    if not isinstance(alpha_tensor, torch.Tensor):
        alpha_tensor = torch.tensor(alpha_tensor, device=device, dtype=M_f.dtype).unsqueeze(0)
    
    cost = None
    for _ in range(gw_iter):
        temp = torch.einsum('dbij,dbjk->dbik', C1, T)
        M_s = 2 * torch.einsum('dbik,dkl->dbil', temp, C2) 
        

        cost = (1 - alpha_tensor) * M_f + alpha_tensor * M_s
        
        T = batch_sinkhorn(cost, p, q, epsilon, sinkhorn_iter)

    fgw_dist = torch.sum(cost * T, dim=(-2, -1))
    result = fgw_dist.transpose(0, 1)
    if squeeze_output:
        result = result.squeeze(0)
    return result


#%%

from torch.autograd import Function
class ValFunction(Function):

    @staticmethod
    def forward(ctx, val, grads, *inputs):
        ctx.grads = grads
        return val

    @staticmethod
    def backward(ctx, grad_output):
        # the gradients are grad
        return (None, None) + tuple(g * grad_output for g in ctx.grads)

    

def set_gradients(Func, val, inputs, grads):

    res = Func.apply(val, grads, *inputs)

    return res

#%%

def probability_simplex_projection(x):
    descending_idx = torch.argsort(x, descending=True)
    u = x[descending_idx]
    rho= 0.
    lambda_= 1.
    for i in range(u.shape[0]):
        value = u[i] + (1- u[:(i+1)].sum())/(i+1)
        if value>0:
            rho+=1
            lambda_-=u[i]
        else:
            break
    return torch.max(x + lambda_/rho, torch.zeros_like(x))

