from typing import Union, Dict, Tuple, Optional
from numbers import Number
import numpy as np
import time
import ot
from ott.solvers import linear
from ott.geometry import geometry
from .utils import random_gamma_init, init_matrix_np
import jax


def cot_numpy(
    X1,
    X2,
    w1=None,
    w2=None,
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
    """Returns COOT between two datasets X1,X2 (see [1]), Sinkhorn reimplemented with OTT

    The function solves the following optimization problem:

    .. math::

        COOT = \min_{Ts,Tv} \sum_{i,j,k,l} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}*Tv_{k,l}

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

    Examples
    ----------
    .. code-block:: python

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
    if w1 is None:
        w1 = np.ones(X1.shape[0]) / X1.shape[0]  # is (n',)
    if w2 is None:
        w2 = np.ones(X2.shape[0]) / X2.shape[0]  # is (n,)

    if not random_init:
        Ts = np.ones((X1.shape[0], X2.shape[0])) / (
            X1.shape[0] * X2.shape[0]
        )  # is (n,n')
        Tv = np.ones((X1.shape[1], X2.shape[1])) / (
            X1.shape[1] * X2.shape[1]
        )  # is (d,d')
    else:
        Ts = random_gamma_init(w1, w2)
        Tv = random_gamma_init(v1, v2)

    constC_s, hC1_s, hC2_s = init_matrix_np(X1, X2, v1, v2)

    constC_v, hC1_v, hC2_v = init_matrix_np(X1.T, X2.T, w1, w2)
    cost = np.inf

    log_out = {}
    log_out["cost"] = []

    for i in range(niter):
        Tsold = Ts
        Tvold = Tv
        costold = cost

        M = constC_s - np.dot(hC1_s, Tv).dot(hC2_s.T)
        if C_lin is not None:
            M = M + C_lin
        if algo == "emd":
            Ts = ot.emd(w1, w2, M, numItermax=1e7)
        elif algo == "sinkhorn":
            Ts = np.array(
                linear.solve(
                    geometry.Geometry(
                        cost_matrix=M, epsilon=reg, scale_cost="max_cost"
                    ),
                    max_iterations=2000,
                ).matrix
            )

        M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)

        if algo2 == "emd":
            Tv = ot.emd(v1, v2, M, numItermax=1e7)
        elif algo2 == "sinkhorn":
            Tv = np.array(
                linear.solve(
                    geometry.Geometry(
                        cost_matrix=M, epsilon=reg2, scale_cost="max_cost"
                    ),
                    max_iterations=2000,
                ).matrix
            )

        delta = np.linalg.norm(Ts - Tsold) + np.linalg.norm(Tv - Tvold)
        cost = np.sum(M * Tv)

        if log:
            log_out["cost"].append(cost)

        if verbose:
            print("Delta: {0}  Loss: {1}".format(delta, cost))

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            if verbose:
                print("converged at iter ", i)
            break
        jax.clear_caches()
    if log:
        return Ts, Tv, cost, log_out
    else:
        return Ts, Tv, cost


def predict_with_cot(Xs, ys, Xt, yt, Xs_test, log=True):
    """Learn mapping from Xs, Ys, Yt, Yt and predict Xt_test from Xs_test."""
    vs = Xs.sum(axis=0)  # set the weights on the features
    vs /= vs.sum()
    vt = Xt.sum(axis=0)
    vt /= vt.sum()

    Ts, Tv, _, log = cot_numpy(Xs, Xt, y1=ys, y2=yt, v1=vs, v2=vt, niter=100, log=True)

    log["Ts"] = Ts
    log["Tv"] = Tv
    Xt_pred = Xs_test @ (Tv / Tv.sum(axis=-1)[:, None])
    return Xt_pred, log


def get_coupling_cot(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]],
) -> Tuple[Union[int, Dict[Number, np.array]], Union[int, Dict]]:
    """Returns sample coupling between two datasets X, Y given the labels, disregarding label information.

    The function solves the following optimization problem:

    .. math::

        COOT = \min_{Ts,Tv} \sum_{i,j,k,l} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}^l*Tv_{k,l}

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
        from perturbot.match import get_coupling_cot

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,1) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,2) for k in labels}
        get_coupling_cot((Xs_dict, Xt_dict))
    """
    X_dict = data[0]
    Y_dict = data[1]
    X = np.concatenate([X_dict[l] for l in X_dict.keys()])
    Y = np.concatenate([Y_dict[l] for l in X_dict.keys()])
    start = time.time()
    try:
        T, Tv, cost, log = cot_numpy(X, Y, log=True, niter=2000)
    except FloatingPointError:
        return -1, -1
    log["time"] = time.time() - start
    return T, log


def get_coupling_cot_sinkhorn(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]],
    eps: float = 5e-3,
    eps2: Optional[float] = None,
) -> Tuple[Union[int, Dict[Number, np.array]], Union[int, Dict]]:
    """Returns sample coupling between two datasets X, Y given the labels, disregarding label information.

    The function solves the following optimization problem:

    .. math::

        ECOOT = \min_{Ts,Tv} \sum_{i,j,k,l} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}^l*Tv_{k,l} - \epsilon H(T_s)

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
        from perturbot.match import get_coupling_cot_sinkhorn

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,1) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,2) for k in labels}
        get_coupling_cot_sinkhorn((Xs_dict, Xt_dict), 0.05)
    """
    print(f"calculating with eps {eps}")
    X_dict = data[0]
    Y_dict = data[1]
    X = np.concatenate([X_dict[l] for l in X_dict.keys()])
    Y = np.concatenate([Y_dict[l] for l in X_dict.keys()])
    if eps2 is None:
        eps2 = eps
    start = time.time()
    try:
        T, Tv, cost, log = cot_numpy(
            X,
            Y,
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
    return T, log


def get_coupling_each_cot_sinkhorn(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]],
    eps: float = 5e-3,
    eps2: Optional[float] = None,
) -> Tuple[Union[int, Dict[Number, np.array]], Union[int, Dict]]:
    """Returns sample coupling between two datasets X, Y given the labels, disregarding label information.

    The function solves the following optimization problem:

    .. math::

        ECOOT = \min_{Ts,Tv} \sum_{i,j,k,l} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}^l*Tv_{k,l} - \epsilon H(T_s)

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
        from perturbot.match import get_coupling_cot_sinkhorn

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,1) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,2) for k in labels}
        get_coupling_cot_sinkhorn((Xs_dict, Xt_dict), 0.05)
    """
    print(f"calculating with eps {eps}")
    X_dict = data[0]
    Y_dict = data[1]
    X = np.concatenate([X_dict[l] for l in X_dict.keys()])
    Y = np.concatenate([Y_dict[l] for l in X_dict.keys()])
    if eps2 is None:
        eps2 = eps
    start = time.time()
    try:
        T_dict = {}
        for l in X_dict.keys():
            T, Tv, cost, log = cot_numpy(
                X_dict[l],
                Y_dict[l],
                algo="sinkhorn",
                reg=eps,
                algo2="sinkhorn",
                reg2=eps2,
                log=True,
                niter=2000,
            )
            T_dict[l] = T
        print(f"Done calculating with eps {eps}")
    except FloatingPointError:
        return -1, -1
    log["time"] = time.time() - start
    return T_dict, log
