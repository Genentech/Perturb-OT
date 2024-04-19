from typing import Dict, Tuple, Union
from numbers import Number
import numpy as np
import time
import ot
from scipy import stats
from scipy.sparse import random
from ott.solvers import linear
from ott.geometry import geometry
from .utils import init_matrix_np, random_gamma_init
from perturbot.utils import mdict_to_matrix


def fot_numpy(
    X1,
    X2,
    Ts,
    v1=None,
    v2=None,
    niter=10,
    algo="emd",
    reg=0,
    algo2="emd",
    reg2=0,
    verbose=True,
    log=False,
    random_init=False,
    C_lin=None,
):
    """Returns COOT between two datasets X1,X2 (see [1])

    The function solves the following optimization problem:
    .. math::

        FOT = \min_{Tv} \sum_{i,j,k,l} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}*Tv_{k,l}

    Where :
    - X1 : The source dataset
    - X2 : The target dataset
    - w1,w2  : weights (histograms) on the samples (rows) of resp. X1 and X2
    - v1,v2  : weights (histograms) on the features (columns) of resp. X1 and X2

    Parameters
    ----------
    X1 : numpy array, shape (n, d)
         Source dataset
    X2 : numpy array, shape (n', d')
         Target dataset
    w1 : numpy array, shape (n,)
        Weight (histogram) on the samples of X1. If None uniform distribution is considered.
    w2 : numpy array, shape (n',)
        Weight (histogram) on the samples of X2. If None uniform distribution is considered.
    v1 : numpy array, shape (d,)
        Weight (histogram) on the features of X1. If None uniform distribution is considered.
    v2 : numpy array, shape (d',)
        Weight (histogram) on the features of X2. If None uniform distribution is considered.
    niter : integer
            Number max of iterations of the BCD for solving COOT.
    algo : string
            Choice of algorithm for solving OT problems on samples each iteration. Choice ['emd','sinkhorn'].
            If 'emd' returns sparse solution
            If 'sinkhorn' returns regularized solution
    algo2 : string
            Choice of algorithm for solving OT problems on features each iteration. Choice ['emd','sinkhorn'].
            If 'emd' returns sparse solution
            If 'sinkhorn' returns regularized solution
    reg : float
            Regularization parameter for samples coupling matrix. Ignored if algo='emd'
    reg2 : float
            Regularization parameter for features coupling matrix. Ignored if algo='emd'
    eps : float
        Threshold for the convergence
    random_init : bool
            Wether to use random initialization for the coupling matrices. If false identity couplings are considered.
    log : bool, optional
         record log if True
    C_lin : numpy array, shape (n, n')
            Prior on the sample correspondences. Added to the cost for the samples transport

    Returns
    -------
    Ts : numpy array, shape (n,n')
           Optimal Transport coupling between the samples
    Tv : numpy array, shape (d,d')
           Optimal Transport coupling between the features
    cost : float
            Optimization value after convergence
    log : dict
        convergence information and coupling marices
    References
    ----------
    .. [1] Redko Ievgen, Vayer Titouan, Flamary R{\'e}mi and Courty Nicolas
          "CO-Optimal Transport"
    Example
    ----------
    import numpy as np
    from cot import cot_numpy

    n_samples=300
    Xs=np.random.rand(n_samples,2)
    Xt=np.random.rand(n_samples,1)
    cot_numpy(Xs,Xt)
    """
    if v1 is None:
        v1 = np.ones(X1.shape[1]) / X1.shape[1]  # is (d,)
    if v2 is None:
        v2 = np.ones(X2.shape[1]) / X2.shape[1]  # is (d',)
    Ts = Ts / Ts.sum()
    w1 = Ts.sum(axis=0)
    w2 = Ts.sum(axis=1)
    if not random_init:
        Tv = np.ones((X1.shape[1], X2.shape[1])) / (
            X1.shape[1] * X2.shape[1]
        )  # is (d,d')
    else:
        Tv = random_gamma_init(v1, v2)

    constC_v, hC1_v, hC2_v = init_matrix_np(X1.T, X2.T, w1, w2)
    cost = np.inf

    log_out = {}
    log_out["cost"] = []

    for i in range(niter):
        Tvold = Tv
        costold = cost

        M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)
        Tv = np.array(
            linear.solve(
                geometry.Geometry(cost_matrix=M, epsilon=reg2, scale_cost="max_cost"),
                max_iterations=2000,
            ).matrix
        )

        delta = np.linalg.norm(Tv - Tvold)
        cost = np.sum(M * Tv)

        if log:
            log_out["cost"].append(cost)

        if verbose:
            print("Delta: {0}  Loss: {1}".format(delta, cost))

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            if verbose:
                print("converged at iter ", i)
            break
    if log:
        return Tv, cost, log_out
    else:
        return Tv, cost


def get_coupling_fot(
    data: Tuple[Dict[Number, np.ndarray], Dict[Number, np.ndarray]],
    Ts: Union[Dict[Number, np.ndarray], np.ndarray],
    eps=5e-3,
):
    r"""Returns GW coupling between features given two datasets X, Y and the sample coupling.

    The function solves the following optimization problem:

    .. math::

        FOT = \min_{Tv} \sum_{i,j,k,l} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}^l*Tv_{k,l} - \epsilon H(T_v)


    Parameters
    ----------
    data :
        (source dataset, target dataset) where source and target datasets
        are the dictionaries mapping label to np.ndarray with matched labels.
    Ts:
        Sample-to-sample transport.
        Per-label transport matched with source dataset, target dataset
        or a global coupling matrix where the samples are concatenated by
        the order of labels in data[0].keys().
    eps:
        Regularization parameter, relative to the max cost.

    Returns
    -------
    Tv :
        Feature-to-feature coupling.
    log :
        Running log

    Example
    ----------
    .. code-block:: python

        import numpy as np
        from perturbot.match import get_coupling_egw_labels_ott, get_coupling_fot

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,2) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        Ts, log = get_coupling_egw_labels_ott((Xs_dict, Xt_dict), 0.05)
        Tv, feature_matching_log = get_coupling_fot((Xs_dict, Xt_dict), Ts, 0.05)
    """

    X_dict = data[0]
    Y_dict = data[1]
    if isinstance(Ts, dict):
        Ts = mdict_to_matrix(
            Ts,
            np.concatenate([np.ones(X_dict[l].shape[0]) * l for l in X_dict.keys()]),
            np.concatenate([np.ones(Y_dict[l].shape[0]) * l for l in X_dict.keys()]),
        )
    X = np.concatenate([X_dict[l] for l in X_dict.keys()])
    Y = np.concatenate([Y_dict[l] for l in X_dict.keys()])
    start = time.time()
    try:
        Tv, cost, log = fot_numpy(X, Y, Ts, log=True, reg=eps, reg2=eps, niter=2000)
    except FloatingPointError:
        return -1, -1
    log["time"] = time.time() - start
    return Tv, log
