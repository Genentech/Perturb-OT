from typing import Tuple, Dict
from numbers import Number
import numpy as np
import scipy as sp
import ot
import time


def get_coupling_gw_labels(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]],
) -> Tuple[Dict[Number, np.array], Dict]:
    r"""Returns GW coupling between two datasets X, Y given the labels.

    The function solves the following optimization problem:

    .. math::

        GWL = \min_{T\in C_{p,q}^\ell} \sum_{i,k \in \{i|l_{x_i}=t\}, j,l \in \{j|l_{y_j}=t}\} |(x_i-x_k)^2 - (y_j-y_l)^2|^{2}*T_{i,j}T_{k,l} \\
        C_{p,q}^\ell = \{T | T \in C{p,q}, T_{ij} > 0 \implies l_{x_i} = l_{y_j}\}

    Parameters
    ----------
    data : 
        (source dataset, target dataset) where source and target datasets 
        are the dictionaries mapping label to np.ndarray with matched labels.

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
        from perturbot.match import get_coupling_gw_labels

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,2) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_gw_labels((Xs_dict, Xt_dict))
    """
    X_dict = data[0]
    Y_dict = data[1]
    Xs_tot = np.concatenate([X_dict[l] for l in X_dict.keys()], axis=0)
    Xt_tot = np.concatenate([Y_dict[l] for l in X_dict.keys()], axis=0)
    source_labels = np.concatenate(
        [np.repeat(l, X_dict[l].shape[0]) for l in X_dict.keys()]
    )
    target_labels = np.concatenate(
        [np.repeat(l, Y_dict[l].shape[0]) for l in X_dict.keys()]
    )
    start = time.time()
    C1_tot = sp.spatial.distance.cdist(Xs_tot, Xs_tot)
    C2_tot = sp.spatial.distance.cdist(Xt_tot, Xt_tot)
    C1_tot /= C1_tot.max()
    C2_tot /= C2_tot.max()
    cost_time = time.time() - start
    start = time.time()
    T, log = ot.gromov.gromov_wasserstein_labeled(
        C1_tot, C2_tot, source_labels, target_labels, log=True
    )
    end = time.time()
    log["time"] = end - start
    log["cosst_time"] = cost_time
    T_dict = {}
    for l in np.unique(source_labels):
        T_dict[l] = T[source_labels == l, :][:, target_labels == l]
    return T_dict, log


def get_coupling_egw_labels(data, eps=5e-3):
    """Returns GW coupling between two datasets X, Y given the labels.

    The function solves the following optimization problem:

    .. math::
    
        GWL = \min_{T\in C_{p,q}^\ell} \sum_{i,k \in \{i|l_{x_i}=t\} j,l \in \{j|l_{y_j}=t}\} |(x_i-x_k)^2 - (y_j-y_l)^2|^{2}*T_{i,j}T_{k,l} - \epsilon H(T)\\
        C_{p,q}^\ell = \{T | T \in C{p,q}, T_{ij} > 0 \implies l_{x_i} = l_{y_j}\}

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
        from perturbot.match import get_coupling_egw_labels

        n_samples = 300
        labels = [0,1,2,3]
        Xs_dict = {k: np.random.rand(n_samples,2) for k in labels}
        Xt_dict = {k: np.random.rand(n_samples,1) for k in labels}
        get_coupling_egw_labels((Xs_dict, Xt_dict), 0.05)
    """
    X_dict = data[0]
    Y_dict = data[1]
    Xs_tot = np.concatenate([X_dict[l] for l in X_dict.keys()], axis=0)
    Xt_tot = np.concatenate([Y_dict[l] for l in X_dict.keys()], axis=0)
    source_labels = np.concatenate(
        [np.repeat(l, X_dict[l].shape[0]) for l in X_dict.keys()]
    )
    target_labels = np.concatenate(
        [np.repeat(l, Y_dict[l].shape[0]) for l in X_dict.keys()]
    )
    start = time.time()
    C1_tot = sp.spatial.distance.cdist(Xs_tot, Xs_tot)
    C2_tot = sp.spatial.distance.cdist(Xt_tot, Xt_tot)
    C1_tot /= C1_tot.max()
    C2_tot /= C2_tot.max()
    cost_time = time.time() - start
    start = time.time()
    print("running LEOT")
    T, log = ot.gromov.entropic_gromov_wasserstein_labeled(
        C1_tot,
        C2_tot,
        source_labels,
        target_labels,
        epsilon=eps,
        log=True,
        verbose=True,
    )
    end = time.time()
    print("Done running LEOT")
    log["time"] = end - start
    log["cost_time"] = cost_time
    T_dict = {}
    for l in np.unique(source_labels):
        T_dict[l] = T[source_labels == l, :][:, target_labels == l]
    return T_dict, log
