import numpy as np
import scipy as sp
import ot
import time


def get_coupling_ot_labels(data, eps=5e-3):
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
    C = sp.spatial.distance.cdist(Xs_tot, Xs_tot)
    C = C / C.max()
    cost_time = time.time() - start
    start = time.time()
    p = np.ones(C.shape[0]) / C.shape[0]
    q = np.ones(C.shape[1]) / C.shape[1]
    T, log = ot.bergman.sinkhorn_labeled(
        p,
        q,
        T,
        source_labels,
        target_labels,
        reg=eps,
        log=True,
        verbose=True,
    )
    end = time.time()
    log["time"] = end - start
    log["cost_time"] = cost_time
    T_dict = {}
    for l in np.unique(source_labels):
        T_dict[l] = T[source_labels == l, :][:, target_labels == l]
    return T_dict, log


def get_coupling_eot_labels(data, eps=5e-3):
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
    C = sp.spatial.distance.cdist(Xs_tot, Xs_tot)
    C = C / C.max()
    cost_time = time.time() - start
    start = time.time()
    p = np.ones(C.shape[0]) / C.shape[0]
    q = np.ones(C.shape[1]) / C.shape[1]
    T, log = ot.bergman.sinkhorn_labeled(
        p,
        q,
        T,
        source_labels,
        target_labels,
        reg=eps,
        log=True,
        verbose=True,
    )
    end = time.time()
    log["time"] = end - start
    log["cost_time"] = cost_time
    T_dict = {}
    for l in np.unique(source_labels):
        T_dict[l] = T[source_labels == l, :][:, target_labels == l]
    return T_dict, log
