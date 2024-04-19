# Adapted from https://github.com/PythonOT/COOT
from typing import Dict, Optional, Tuple, Union
from numbers import Number
import pandas as pd
import numpy as np
import time
import ot as pot
import matplotlib.pyplot as plt
from ott.solvers import linear
from ott.geometry import geometry
from .utils import random_gamma_init, init_matrix_np


def cotl_numpy(
    X_dict: Dict[Number, np.ndarray],
    Y_dict: Dict[Number, np.ndarray],
    w1: Dict[Number, np.ndarray] = None,
    w2: Dict[Number, np.ndarray] = None,
    v1: Optional[np.ndarray] = None,
    v2: Optional[np.ndarray] = None,
    niter: int = 100,
    algo: str = "emd",
    reg: float = 0.1,
    algo2: str = "emd",
    reg2: float = 10.0,
    verbose: bool = True,
    log: bool = False,
    random_init: bool = False,
    C_lin: bool = None,
):
    r"""Returns COOT between two datasets X, Y given labels.

    The function solves the following optimization problem:

    .. math::

        COOTL = \min_{Ts^1,..Ts^{L},Tv} \sum_{t=1}^L \sum_{i,k \in {i|l_{x_i}=t}, j,l \in {j|l_{y_j}=t}} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}^l*Tv_{k,l}


    Parameters
    ----------
    X1 : numpy array, shape (n, d)
         Source dataset
    X2 : numpy array, shape (n', d')
         Target dataset
    y1 : numpy array, shape (n,)
    y2 : numpy array, shape (n',)
    w1 : numpy array, shape (n,)
        Weight (histogram) on the samples of X1. If None uniform distribution is considered.
    w2 : Ditionary of numpy array, shape (n',)
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
    .. code-block:: python

        import numpy as np
        from perturbot.match import cotl_numpy

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,2) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        cotl_numpy(Xs_dict, Xt_dict)
    """
    assert sorted(X_dict.keys()) == sorted(
        Y_dict.keys()
    ), "Labels don't match in y1 & y2."
    labels = list(X_dict.keys())
    if v1 is None:
        X = np.concatenate([X_dict[k] for k in labels], axis=0)
        if (X >= 0).all():
            v1 = X.sum(axis=0) / X.sum()
        else:
            v1 = np.ones(X.shape[1]) / X.shape[1]

    if v2 is None:
        Y = np.concatenate([Y_dict[k] for k in labels], axis=0)
        if (Y >= 0).all():
            v2 = Y.sum(axis=0) / Y.sum()
        else:
            v2 = np.ones(Y.shape[1]) / Y.shape[1]

    if w1 is None:
        w1 = {
            k: np.ones(X_dict[k].shape[0]) / X_dict[k].shape[0] for k in labels
        }  # is (n',)
    if w2 is None:
        w2 = {
            k: np.ones(Y_dict[k].shape[0]) / Y_dict[k].shape[0] for k in labels
        }  # is (n,)

    if not random_init:
        Ts = {
            k: np.ones((X_dict[k].shape[0], Y_dict[k].shape[0]))
            / (X_dict[k].shape[0] * Y_dict[k].shape[0])
            for k in labels
        }  # is (n,n')
        Tv = np.ones((X_dict[labels[0]].shape[1], Y_dict[labels[0]].shape[1])) / (
            X_dict[labels[0]].shape[1] * Y_dict[labels[0]].shape[1]
        )  # is (d,d')
    else:
        Ts = {k: random_gamma_init(w1[k], w2[k]) for k in labels}
        Tv = random_gamma_init(v1, v2)

    constC_s_dict = {}
    hC1_s_dict = {}
    hC2_s_dict = {}

    constC_v_dict = {}
    hC1_v_dict = {}
    hC2_v_dict = {}

    for k in labels:
        constC_s_dict[k], hC1_s_dict[k], hC2_s_dict[k] = init_matrix_np(
            X_dict[k], Y_dict[k], v1, v2
        )
        constC_v_dict[k], hC1_v_dict[k], hC2_v_dict[k] = init_matrix_np(
            X_dict[k].T, Y_dict[k].T, w1[k], w2[k]
        )
        cost = np.inf

    log_out = {}
    log_out["cost"] = []

    for i in range(niter):
        Tsold = Ts
        Tvold = Tv
        costold = cost

        # Sample OT for each label
        for k in labels:
            M_k = constC_s_dict[k] - np.dot(hC1_s_dict[k], Tv).dot(hC2_s_dict[k].T)
            print(f"M_{k}:{M_k.min()} - {M_k.max()}")
            if C_lin is not None:
                M_k = M_k + C_lin
            if algo == "emd":
                Ts[k] = pot.emd(w1[k], w2[k], M_k, numItermax=1e7)
            elif algo == "sinkhorn":
                Ts[k] = np.array(
                    linear.solve(
                        geometry.Geometry(
                            cost_matrix=M_k, epsilon=reg, scale_cost="max_cost"
                        ),
                        max_iterations=2000,
                    ).matrix
                )

        # Global feature OT
        M = 0
        for k in labels:
            M += constC_v_dict[k] - np.dot(hC1_v_dict[k], Ts[k]).dot(hC2_v_dict[k].T)
        print(f"M:{M.min()} - {M.max()}")
        if algo2 == "emd":
            Tv = pot.emd(v1, v2, M, numItermax=1e7)
        elif algo2 == "sinkhorn":
            Tv = np.array(
                linear.solve(
                    geometry.Geometry(
                        cost_matrix=M, epsilon=reg, scale_cost="max_cost"
                    ),
                    max_iterations=2000,
                ).matrix
            )
        if not np.abs(Tv.sum() - 1.0) < 1e-8:
            Tv = Tv / Tv.sum()
        delta = sum(
            [np.linalg.norm(Ts[k] - Tsold[k]) for k in labels]
        ) + np.linalg.norm(Tv - Tvold)
        cost = np.sum(M * Tv)

        if log:
            log_out["cost"].append(cost)

        if verbose:
            print(f"It {i} Delta: {delta}  Loss: {cost}")

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            if verbose:
                print("converged at iter ", i)
            break
    if log:
        return Ts, Tv, cost, log_out
    else:
        return Ts, Tv, cost


def get_coupling_cotl(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]],
) -> Tuple[Union[int, Dict[Number, np.array]], Union[int, Dict]]:
    """Returns sample coupling between two datasets X, Y given the labels, disregarding label information.

    The function solves the following optimization problem:

    .. math::

        COOTL = \min_{Ts^1,..Ts^{L},Tv} \sum_{t=1}^L \sum_{i,k \in {i|l_{x_i}=t}, j,l \in {j|l_{y_j}=t}} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}^l*Tv_{k,l}


    Parameters
    ----------
    data :
        (source dataset, target dataset) where source and target datasets
        are the dictionaries mapping label to np.ndarray with matched labels.
    eps:
        Regularization parameter, relative to the max cost.

    Returns
    -------
    T_dict :
        Optimal Transport coupling between the samples per label
    log :
        Running log

    Example
    ----------
    .. code-block:: python

        import numpy as np
        from perturbot.match import get_coupling_eot_ott

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,1) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_eot_ott((Xs_dict, Xt_dict), 0.05)
    """
    X_dict = data[0]
    Y_dict = data[1]
    start = time.time()
    try:
        Ts, Tv, cost, log = cotl_numpy(X_dict, Y_dict, log=True, niter=2000)
    except FloatingPointError:
        return -1, -1
    log["time"] = time.time() - start
    return Ts, log


def get_coupling_cotl_sinkhorn(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]],
    eps: float = 5e-3,
    eps2: float = None,
) -> Tuple[Dict[Number, np.array], Dict]:
    """Returns sample coupling between two datasets X, Y given the labels.

    The function solves the following optimization problem:

    .. math::

        COOTL = \min_{Ts^1,..Ts^{L},Tv} \sum_{t=1}^L \sum_{i,k \in {i|l_{x_i}=t}, j,l \in {j|l_{y_j}=t}} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}^l*Tv_{k,l} - \epsilon_1 H(Ts) -\epsilon_2 H(Tv)

    Parameters
    ----------
    data :
        (source dataset, target dataset) where source and target datasets
        are the dictionaries mapping label to np.ndarray with matched labels.
    eps:
        Regularization parameter, relative to the max cost.

    Returns
    -------
    T_dict :
        Optimal Transport coupling between the samples per label
    log :
        Running log

    Example
    ----------
    .. code-block:: python

        import numpy as np
        from perturbot.match import get_coupling_eot_ott

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,1) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_eot_ott((Xs_dict, Xt_dict), 0.05)
    """
    print(f"calculating with eps {eps}")
    X_dict = data[0]
    Y_dict = data[1]
    start = time.time()
    if eps2 is None:
        eps2 = eps
    try:
        Ts, Tv, cost, log = cotl_numpy(
            X_dict,
            Y_dict,
            algo="sinkhorn",
            reg=eps,
            algo2="sinkhorn",
            reg2=eps2,
            log=True,
            niter=2000,
        )
    except FloatingPointError:
        return -1, -1
    log["time"] = time.time() - start
    return Ts, log
