"""
Solvers from the Python Optimal Transport library 
https://pythonot.github.io/ (version POT=0.8.1)
 
adapted for the integration of (Fused) Gromov-Wasserstein in TFGW layers
"""

import numpy as np
import ot
import torch as th
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
        # 使用 batch matrix multiplication，GPU 高度优化
        XX = (X * X).sum(-1, keepdim=True)  # (B, N, 1)
        YY = (Y * Y).sum(-1, keepdim=True)  # (B, M, 1)
        XY = th.bmm(X, Y.transpose(-2, -1))  # (B, N, M)
        return XX + YY.transpose(-2, -1) - 2 * XY  # (B, N, M)
    else:
        raise ValueError(f"Input must be 2D or 3D, got {X.dim()}D")

#%%

    


def parallel_gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', log=False, armijo=False, G0=None, **kwargs):
    r"""
    Returns the gromov-wasserstein discrepancy between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    The function solves the following optimization problem:
    .. math::
        GW = \min_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}
    Where :
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity
      matrices
    Note that when using backends, this loss function is differentiable wrt the
    marices and weights for quadratic loss using the gradients from [38]_.
    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Distribution in the source space.
    q :  array-like, shape (nt,)
        Distribution in the target space.
    loss_fun :  str
        loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    log : dict
        convergence information and Coupling marix
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    .. [13] Mémoli, Facundo. Gromov-Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20 = p, q, C1, C2
    nx = get_backend(p0, q0, C10, C20)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    
    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-04)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-04)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_gw = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)

    #T0 = nx.from_numpy(T, type_as=C10)

    #log_gw['gw_dist'] = nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10)
    gp = nx.from_numpy(log_gw['u'] - log_gw['u'].mean())
    gq = nx.from_numpy(log_gw['v'] - log_gw['v'].mean())
    #log_gw['T'] = T0

    if loss_fun == 'square_loss':
        
        gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
        gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
        #gw = nx.set_gradients(gw, (p0, q0, C10, C20),
        #                     (log_gw['u'], log_gw['v'], gC1, gC2))
    #for checking backprop manually
    #return T0, nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10), nx.from_numpy(log_gw['u']), nx.from_numpy(log_gw['v']), gC1, gC2
    return nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10), gp, gq, gC1, gC2

def parallel_fused_gromov_wasserstein2_learnablealpha( C1, C2, F1, F2, M, p, q, loss_fun='square_loss', alpha=0.5, compute_gradients=True, learn_alpha=False, armijo=False, log=False, G0=None, **kwargs):
    r"""
    Computes the FGW distance between two graphs see (see :ref:`[24] <references-fused-gromov-wasserstein2>`)
    .. math::
        \min_\gamma \quad (1 - \alpha) \langle \gamma, \mathbf{M} \rangle_F + \alpha \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}
        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}
             \mathbf{\gamma}^T \mathbf{1} &= \mathbf{q}
             \mathbf{\gamma} &\geq 0
    where :
    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are source and target weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices
    The algorithm used for solving the problem is conditional gradient as
    discussed in :ref:`[24] <references-fused-gromov-wasserstein2>`
    Note that when using backends, this loss function is differentiable wrt the
    marices and weights for quadratic loss using the gradients from [38]_.
    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space.
    p :  array-like, shape (ns,)
        Distribution in the source space.
    q :  array-like, shape (nt,)
        Distribution in the target space.
    loss_fun : str, optional
        Loss function used for the solver.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    log : bool, optional
        Record log if True.
    **kwargs : dict
        Parameters can be directly passed to the ot.optim.cg solver.
    Returns
    -------
    fgw-distance : float
        Fused gromov wasserstein distance for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.
    .. _references-fused-gromov-wasserstein2:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20, F10, F20, M0, alpha0 = p, q, C1, C2, F1, F2, M, alpha
    nx = get_backend(p0, q0, C10, C20, F10, F20, M0, alpha0)

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    F1 = nx.to_numpy(F10)
    F2 = nx.to_numpy(F20)
    M = nx.to_numpy(M0)
    alpha = nx.to_numpy(alpha0)
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-04)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-04)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_fgw = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

    fgw_dist = nx.from_numpy(log_fgw['loss'][-1], type_as=C10)
    if not compute_gradients:
        return fgw_dist
    else:
        #T0 = nx.from_numpy(T, type_as=C10)
    
        #log_fgw['fgw_dist'] = fgw_dist
        #log_fgw['u'] = nx.from_numpy(log_fgw['u'], type_as=C10)
        #log_fgw['v'] = nx.from_numpy(log_fgw['v'], type_as=C10)
        #log_fgw['T'] = T0
    
        if loss_fun == 'square_loss':
            gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T), type_as=C10)
            gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T), type_as=C10)
            if learn_alpha:
                gwloss_ = gwloss(constC, hC1, hC2, T)
                galpha = nx.from_numpy(gwloss_ - (M*T).sum(), type_as=C10)
            else:
                galpha = None
            #fgw_dist = nx.set_gradients(fgw_dist, (p0, q0, C10, C20, M0, alpha0),
            #                            (log_fgw['u'], log_fgw['v'], alpha0 * gC1, alpha0 * gC2, (1 - alpha0) * T0, galpha))
        gp = nx.from_numpy(log_fgw['u'] - log_fgw['u'].mean(), type_as=C10)
        gq = nx.from_numpy(log_fgw['v'] - log_fgw['v'].mean(), type_as=C10)
        gF1 = nx.from_numpy(2 * F1 * p[:, None] - 2 * T.dot(F2), type_as=C10)
        gF2 = nx.from_numpy(2 * F2 * q[:, None] - 2 * (T.T).dot(F1), type_as=C10)
        
        # device = gC1.device  # All grads are already on the correct device
        # # Ensure alpha0 is a tensor on the correct device and type
        # if not th.is_tensor(alpha0):
        #     alpha0_tensor = th.tensor(alpha0, dtype=gC1.dtype, device=device)
        # else:
        #     alpha0_tensor = alpha0.to(device=device, dtype=gC1.dtype)
        # one_minus_alpha0 = 1. - alpha0

        # return (
        #     fgw_dist,
        #     gp,
        #     gq,
        #     alpha0_tensor * gC1,
        #     alpha0_tensor * gC2,
        #     one_minus_alpha0 * gF1,
        #     one_minus_alpha0 * gF2,
        #     galpha if galpha is None else galpha.to(device)
        # )
        return fgw_dist, gp, gq, alpha0 * gC1, alpha0 * gC2, (1. - alpha0) * gF1, (1. - alpha0) * gF2, galpha

def batch_parallel_fused_gromov_wasserstein2_learnablealpha(C_filter, F_filter, h_filter, 
    C_sub, F_sub, h_sub, alpha, loss_fun='square_loss', compute_gradients=True, 
    learn_alpha=False, max_iter=50, tol=1e-6):
    """
    Optimized batched FGW solver with early stopping and efficient memory usage.
    
    C_filter: tensor (N_filter, N_filter, D_out) - Filter structure matrices
    F_filter: tensor (N_filter, D_feat, D_out) - Filter features
    h_filter: tensor (batch, N_filter) - Filter distribution
    C_sub: tensor (batch, N_sub, N_sub) - Input subgraph structure matrices
    F_sub: tensor (batch, N_sub, D_feat) - Input subgraph features
    h_sub: tensor  (batch, N_sub) - Input subgraph distribution
    alpha: float or tensor (1,) - Trade-off parameter
    """
    # Get dimensions and device
    batch_size, N_sub, _ = C_sub.shape
    N_filter, _, D_out = C_filter.shape
    device = C_sub.device

    # Reshape inputs for parallel processing across all dimensions
    # Shape: (D_out, batch_size, ...)
    C1 = C_sub.unsqueeze(0).expand(D_out, -1, -1, -1)  
    F1 = F_sub.unsqueeze(0).expand(D_out, -1, -1, -1)
    p = h_sub.unsqueeze(0).expand(D_out, -1, -1)  # (D_out, batch, N_sub)

    # Prepare filter matrices - avoid repeat operations
    C2 = C_filter.permute(2, 0, 1)  # (D_out, N_filter, N_filter)
    F2 = F_filter.permute(2, 0, 1)  # (D_out, N_filter, D_feat)
    q = h_filter.unsqueeze(0).expand(D_out, -1, -1)  # (D_out, batch, N_filter)

    # Initial coupling matrix
    G = th.einsum('dbi,dbj->dbij', p, q)  # (D_out, batch, N_sub, N_filter)

    # Pre-compute feature distances efficiently
    M = th.cdist(
        F1.reshape(D_out * batch_size, N_sub, -1),  # Changed (D_out*batch, N_sub, D_feat)
        F2.unsqueeze(1).expand(-1, batch_size, -1, -1).reshape(D_out * batch_size, N_filter, -1) # (D_out*batch, N_filter, D_feat)
    ).reshape(D_out, batch_size, N_sub, N_filter)  # (D_out, batch, N_sub, N_filter)

    # Optimization parameters
    beta = 0.9  # Momentum
    prev_deltaG = 0
    prev_loss = None
    no_improve_count = 0

    def batch_gwloss(G):
        """Vectorized GW loss computation"""
        struct_term = th.einsum('dbij,djk,dblk,dbil->db',
                            G, C2, G, C1)  # (D_out, batch)
        feat_term = th.sum(M * G, dim=[-2,-1])  # (D_out, batch)
        return (1-alpha) * feat_term + alpha * struct_term

    def batch_gwgrad(G):
        """Vectorized gradient computation"""
        struct_grad = 2 * alpha * th.einsum('dbij,dbjk,dkl->dbil',
                                        C1, G, C2)  # (D_out, batch, N_sub, N_filter)
        return struct_grad + (1-alpha) * M

    def batch_sinkhorn(cost, p, q, reg=5e-2, num_iters=20):
        """Stabilized Sinkhorn in log-space"""
        """
        Args:
            cost: (D_out, batch, N_sub, N_filter)
            p: (D_out, batch, N_sub)
            q: (D_out, batch, N_filter)
        """
        log_p = th.log(p)
        log_q = th.log(q)
        log_K = -cost/reg
        
        u = th.zeros_like(p, device=device)
        for _ in range(num_iters):
            v = log_q - th.logsumexp(log_K + u.unsqueeze(-1), dim=2)  # sum over N_sub
            u = log_p - th.logsumexp(log_K + v.unsqueeze(2), dim=3)  # sum over N_filter
        
        return th.exp(u.unsqueeze(-1) + v.unsqueeze(2) + log_K)

    # Main optimization loop with early stopping
    for it in range(max_iter):
        grad = batch_gwgrad(G)
        Gc = batch_sinkhorn(-grad, p, q)
        
        # Momentum update
        deltaG = Gc - G
        deltaG = beta * prev_deltaG + (1-beta) * deltaG
        prev_deltaG = deltaG
        
        # Efficient line search
        step_sizes = th.linspace(0, 1, 10, device=device)
        losses = th.stack([batch_gwloss(G + s * deltaG) for s in step_sizes], dim=-1)
        best_steps = step_sizes[th.argmin(losses, dim=-1)]
        
        # Update G
        G = G + best_steps.view(D_out, batch_size, 1, 1) * deltaG
        
        # Early stopping check
        curr_loss = losses[...,0]
        if prev_loss is not None:
            improve = th.abs(curr_loss - prev_loss).max()
            if improve < tol:
                no_improve_count += 1
                if no_improve_count >= 3:  # Stop after 3 iterations with no improvement
                    break
            else:
                no_improve_count = 0
        prev_loss = curr_loss

    # Compute final distance
    fgw_dist = batch_gwloss(G).transpose(0,1)  # (batch, D_out)
    
    if not compute_gradients:
        return fgw_dist

    # Efficient gradient computation
    q_mean = q.mean(1)  # (D_out, N_filter)  # Average over batch
    
    # Compute all gradients in parallel
    gC_filter = (2 * alpha * (
        th.einsum('di,dj->dij', q_mean, q_mean) * C2 -
        th.einsum('dbji,dbjl,dblk->dbik', G, C1, G).mean(1)
    )).permute(1,2,0)  # (N_filter, N_filter, D_out)

    gF_filter = (2 * (1-alpha) * (
        F2 * q_mean.unsqueeze(-1) -
        th.einsum('dbji,dbjk->dbik', G, F1).mean(1)
    )).permute(1,2,0)  # (N_filter, D_feat, D_out)
    
    gh_filter = -th.sum(grad * G, dim=[-2,-1]).mean([0,1])  # (N_filter,)

    # Add alpha gradient computation
    if learn_alpha:
        # Compute structure term
        struct_term = th.einsum('dbij,djk,dblk,dbil->db', 
                              G, C2, G, C1)  # (D_out, batch)
        # Compute feature term
        feat_term = th.sum(M * G, dim=[-2,-1])  # (D_out, batch)
        # Gradient of alpha is structure_term - feature_term
        galpha = (struct_term - feat_term).mean()  # scalar
    else:
        galpha = None

    # Return shapes:
    # fgw_dist: (batch, D_out)
    # gC_filter: (N_filter, N_filter, D_out)
    # gF_filter: (N_filter, D_feat, D_out)
    # gh_filter: (N_filter,)
    # galpha: scalar or None
    return fgw_dist, gC_filter, gF_filter, gh_filter, galpha

def batch_parallel_fused_gromov_wasserstein2_entropic(C_filter, F_filter, h_filter, 
    C_sub, F_sub, h_sub, alpha, epsilon=0.01, loss_fun='square_loss', 
    compute_gradients=True, learn_alpha=False, max_iter=50, inner_iter=20, tol=1e-6):
    """
    Optimized batched FGW solver using entropic regularization.
    
    Args:
        C_filter: tensor (N_filter, N_filter, D_out) - Filter structure matrices
        F_filter: tensor (N_filter, D_feat, D_out) - Filter features 
        h_filter: tensor (batch, N_filter) - Filter distribution
        C_sub: tensor (batch, N_sub, N_sub) - Input subgraph structure matrices
        F_sub: tensor (batch, N_sub, D_feat) - Input subgraph features
        h_sub: tensor  (batch, N_sub) - Input subgraph distribution
        alpha: float or tensor (1,) - Trade-off parameter
        epsilon: float - Entropic regularization parameter
    """
    # Get dimensions and device
    batch_size, N_sub, D_feat = F_sub.shape
    N_filter, _, D_out = C_filter.shape
    device = C_sub.device

    # Scale C1, C2, F1, F2 to [0, 1]
    C1 = C_sub
    C2 = C_filter
    F1 = F_sub
    F2 = F_filter

    C1_scale = C1.max() + 1e-8
    C2_scale = C2.max() + 1e-8
    F_scale = max(F1.max(), F2.max()) + 1e-8
    # F2_scale = F2.max() + 1e-8

    C1 = C1 / C1_scale
    C2 = C2 / C2_scale
    F1 = F1 / F_scale
    F2 = F2 / F_scale

    # Reshape inputs for parallel processing
    C1_expanded = C1.unsqueeze(0).expand(D_out, -1, -1, -1)  # (D_out, batch, N_sub, N_sub)
    F1_expanded = F1.unsqueeze(0).expand(D_out, -1, -1, -1)  # (D_out, batch, N_sub, D_feat)
    p = h_sub.unsqueeze(0).expand(D_out, -1, -1)  # (D_out, batch, N_sub)

    # Prepare filter matrices
    C2_permuted = C2.permute(2, 0, 1)  # (D_out, N_filter, N_filter)
    F2_permuted = F2.permute(2, 0, 1)  # (D_out, N_filter, D_feat)
    q = h_filter.unsqueeze(0).expand(D_out, -1, -1)  # (D_out, batch, N_filter)

    # Pre-compute feature distances
    # F1: (D_out, batch, N_sub, D_feat) -> (D_out * batch, N_sub, D_feat)
    # F2: (D_out, N_filter, D_feat) -> (D_out * batch, N_filter, D_feat)
    M = th.cdist(
        F1_expanded.reshape(D_out * batch_size, N_sub, D_feat),
        F2_permuted.unsqueeze(1).expand(-1, batch_size, -1, -1).reshape(D_out * batch_size, N_filter, D_feat)
    ).reshape(D_out, batch_size, N_sub, N_filter) # (D_out, batch, N_sub, N_filter)

    def entropic_sinkhorn(cost, p, q, epsilon, num_iter=20):
        """
        Stabilized entropic regularization Sinkhorn in log-space
        
        Args:
            cost: (D_out, batch, N_sub, N_filter) - Cost matrix
            p: (D_out, batch, N_sub) - Source distribution
            q: (D_out, batch, N_filter) - Target distribution
            epsilon: float - Regularization parameter
        """
        # Initialize dual variables in log space
        u = th.zeros_like(p, device=device)
        v = th.zeros_like(q, device=device)
        
        # Compute kernel in log space
        log_K = -cost / epsilon
        
        # Sinkhorn iterations
        for _ in range(num_iter):
            # Update u and v in log space (stabilized version)
            v_exp = log_K + v.unsqueeze(2)  # (D_out, batch, N_sub, N_filter)
            u = epsilon * (th.log(p + 1e-8) - th.logsumexp(v_exp, dim=-1))  # (D_out, batch, N_sub)

            u_exp = log_K + u.unsqueeze(-1)  # (D_out, batch, N_sub, N_filter)
            v = epsilon * (th.log(q + 1e-8) - th.logsumexp(u_exp, dim=-2))  # (D_out, batch, N_filter)
            
        # Return transport plan in log space
        log_P = log_K + u.unsqueeze(-1) + v.unsqueeze(2)
        return th.exp(log_P)

    def batch_gwloss(G):
        """Vectorized GW loss computation with entropic regularization"""
        struct_term = th.einsum('dbij,djk,dblk,dbil->db', G, C2_permuted, G, C1_expanded)
        feat_term = th.sum(M * G, dim=[-2,-1])
        # Add entropy regularization term
        entropy = -th.sum(G * th.log(G + 1e-8), dim=[-2,-1])
        return (1-alpha) * feat_term + alpha * struct_term + epsilon * entropy

    def batch_gwgrad(G):
        """Vectorized gradient computation with entropic term"""
        struct_grad = 2 * alpha * th.einsum('dbij,dbjk,dkl->dbil', C1_expanded, G, C2_permuted)
        feat_grad = (1-alpha) * M
        # Add gradient of entropy term
        entropy_grad = epsilon * (1 + th.log(G + 1e-8))
        return struct_grad + feat_grad + entropy_grad

    # Initialize transport plan
    G = th.einsum('dbi,dbj->dbij', p, q)
    prev_loss = None
    no_improve_count = 0

    # Main optimization loop
    for it in range(max_iter):
        # Compute gradient
        grad = batch_gwgrad(G)
        
        # Update using entropic Sinkhorn
        G_new = entropic_sinkhorn(-grad, p, q, epsilon, inner_iter)
        
        # Line search
        step_sizes = th.linspace(0, 1, 5, device=device)
        losses = th.stack([
            batch_gwloss(G + s * (G_new - G)) 
            for s in step_sizes
        ], dim=-1)
        best_steps = step_sizes[th.argmin(losses, dim=-1)]
        
        # Update transport plan
        G = G + best_steps.view(D_out, batch_size, 1, 1) * (G_new - G)
        
        # Early stopping check
        curr_loss = losses[..., 0]
        if prev_loss is not None:
            improve = th.abs(curr_loss - prev_loss).max()
            if improve < tol:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
            else:
                no_improve_count = 0
        prev_loss = curr_loss

    # Compute final FGW distance
    fgw_dist = batch_gwloss(G).transpose(0, 1)

    if not compute_gradients:
        return fgw_dist

    # Compute gradients
    q_mean = q.mean(1)
    
    # Scale G before einsum
    # G_scaled = G / (G.max() + 1e-8)  # Scale G to [0, 1]

    gC_filter = (2 * alpha * (
        th.einsum('di,dj->dij', q_mean, q_mean) * C2_permuted -
        th.einsum('dbji,dbjl,dblk->dbik', G, C1_expanded, G).mean(1)
    )).permute(1, 2, 0)
    gC_filter = gC_filter

    gF_filter = (2 * (1-alpha) * (
        F2_permuted * q_mean.unsqueeze(-1) -
        th.einsum('dbji,dbjk->dbik', G, F1_expanded).mean(1)
    )).permute(1, 2, 0)
    gF_filter = gF_filter
    
    gh_filter = -th.sum(grad * G, dim=[-2,-1]).mean([0,1])

    if learn_alpha:
        struct_term = th.einsum('dbij,djk,dblk,dbil->db', G, C2_permuted, G, C1_expanded)
        feat_term = th.sum(M * G, dim=[-2,-1])
        galpha = (struct_term - feat_term).mean()
    else:
        galpha = None

    return fgw_dist, gC_filter, gF_filter, gh_filter, galpha

def entropic_fused_gromov_wasserstein(
    C_filter, F_filter, h_filter, C_sub, F_sub, h_sub, alpha,
    epsilon=1e-2, gw_iter=5, sinkhorn_iter=5):
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
    # 将结构矩阵归一化到 [0, 1] (防止不同图尺度带来的结构项波动)
    # 如果是邻接矩阵(0/1)这步没影响，但如果是最短路径矩阵，这步至关重要
    C1_max = C1.amax(dim=(-2, -1), keepdim=True)
    C1 = C1 / (C1_max + 1e-8)
    
    C2_max = C2.amax(dim=(-2, -1), keepdim=True)
    C2 = C2 / (C2_max + 1e-8)

    # 3. --- PRE-COMPUTATION: Feature distance matrix ---
    F1_reshaped = F1.reshape(D_out * batch_size, N_sub, D_feat)
    F2_expanded = F2.unsqueeze(1).expand(-1, batch_size, -1, -1).reshape(D_out * batch_size, N_filter, D_feat)
    
    # 计算特征距离矩阵 M_f
    M_f = fast_squared_euclidean(F1_reshaped, F2_expanded)
    M_f = M_f.reshape(D_out, batch_size, N_sub, N_filter)

    # 4. --- NORMALIZATION STEP 2: Feature Cost Matrix ---
    # 关键！将特征搬运代价归一化到 [0, 1]
    # 这消除了特征数值大小(Scale)对 alpha 的影响，也解决了 "Hub +100" 导致代价过大的问题
    M_f_max = M_f.amax(dim=(-2, -1), keepdim=True)
    M_f = M_f / (M_f_max + 1e-8)

    # 5. --- CORE ALGORITHM: Iterative Entropic FGW Solver ---
    # Initialize transport plan
    T = th.einsum('dbi,dbj->dbij', p, q)

    def batch_sinkhorn(cost_matrix, p_dist, q_dist, reg, num_iters):
        log_p = th.log(p_dist + 1e-20)
        log_q = th.log(q_dist + 1e-20)
        log_K = -cost_matrix / reg
        u = th.zeros_like(p_dist, device=device)
        for _ in range(num_iters):
            v = log_q - th.logsumexp(log_K + u.unsqueeze(-1), dim=2)
            u = log_p - th.logsumexp(log_K + v.unsqueeze(2), dim=3)
        return th.exp(u.unsqueeze(-1) + v.unsqueeze(2) + log_K)

    # Handle alpha: convert to tensor if needed and ensure proper broadcasting
    alpha_tensor = alpha
    if not isinstance(alpha_tensor, th.Tensor):
        alpha_tensor = th.tensor(alpha_tensor, device=device, dtype=M_f.dtype)
    
    cost = None
    for _ in range(gw_iter):
        # 计算结构代价 (基于归一化后的 C1, C2)
        temp = th.einsum('dbij,dbjk->dbik', C1, T)
        M_s = 2 * th.einsum('dbik,dkl->dbil', temp, C2) # 这里 M_s 近似也在 [0, 2] 范围内
        
        # 如果 gw_iter > 1，建议也对 M_s 做归一化，但通常 C1/C2 归一化后 M_s 也是受控的
        # 组合特征代价和结构代价，alpha_tensor 自动广播到 M_f 的形状
        cost = (1 - alpha_tensor) * M_f + alpha_tensor * M_s
        
        T = batch_sinkhorn(cost, p, q, epsilon, sinkhorn_iter)

    fgw_dist = th.sum(cost * T, dim=(-2, -1))
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
    descending_idx = th.argsort(x, descending=True)
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
    return th.max(x + lambda_/rho, th.zeros_like(x))

#%% 


# ============================================================================

# ============================================================================



