import numpy as np
import scipy as sp
import ot
import time


def gw_cg(X_dict, Y_dict):
    """GW with conditional gradient algorithm"""
    labels = X_dict.keys()
    Ts = {}
    log = {"cost_time": {}, "time": {}}
    for l in labels:
        start = time.time()
        C1 = sp.spatial.distance.cdist(X_dict[l], X_dict[l])
        C2 = sp.spatial.distance.cdist(Y_dict[l], Y_dict[l])

        C1 /= C1.max()
        C2 /= C2.max()
        log["cost_time"][l] = time.time() - start
        p = ot.unif(C1.shape[0])
        q = ot.unif(C2.shape[0])

        start = time.time()
        Ts[l], log[l] = ot.gromov.gromov_wasserstein(
            C1, C2, p, q, "square_loss", verbose=True, log=True
        )
        log["time"][l] = time.time() - start
    return Ts, log


def egw_pgd(X_dict, Y_dict, epsilon=5e-3):
    """EGW with projected gradient descent algorithm"""
    labels = X_dict.keys()
    Ts = {}
    log = {"cost_time": {}, "time": {}}
    for l in labels:
        start = time.time()
        C1 = sp.spatial.distance.cdist(X_dict[l], X_dict[l])
        C2 = sp.spatial.distance.cdist(Y_dict[l], Y_dict[l])

        C1 /= C1.max()
        C2 /= C2.max()
        log["cost_time"][l] = time.time() - start
        p = ot.unif(C1.shape[0])
        q = ot.unif(C2.shape[0])
        start = time.time()
        Ts[l], log[l] = ot.gromov.entropic_gromov_wasserstein(
            C1,
            C2,
            p,
            q,
            "square_loss",
            epsilon=epsilon,
            solver="PGD",
            log=True,
            verbose=True,
        )
        log["time"][l] = time.time() - start
    return Ts, log


def gw_all(Xtot, Ytot):
    start = time.time()
    C1 = sp.spatial.distance.cdist(Xtot, Xtot)
    C2 = sp.spatial.distance.cdist(Ytot, Ytot)
    C1 /= C1.max()
    C2 /= C2.max()
    cost_time = time.time() - start
    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])
    start = time.time()
    Ts, log = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, "square_loss", verbose=True, log=True, max_iter=1e8
    )
    log["time"] = time.time() - start
    log["cost_time"] = cost_time
    return Ts, log


def egw_all(Xtot, Ytot, epsilon=5e-3):
    start = time.time()
    C1 = sp.spatial.distance.cdist(Xtot, Xtot)
    C2 = sp.spatial.distance.cdist(Ytot, Ytot)
    C1 /= C1.max()
    C2 /= C2.max()
    cost_time = time.time() - start
    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])
    start = time.time()
    Ts, log = ot.gromov.entropic_gromov_wasserstein(
        C1,
        C2,
        p,
        q,
        "square_loss",
        verbose=True,
        log=True,
        epsilon=epsilon,
    )
    log["time"] = time.time() - start
    log["cost_time"] = cost_time
    return Ts, log


def get_coupling_gw_cg(data):
    X_dict = data[0]
    Y_dict = data[1]
    Ts, log = gw_cg(X_dict, Y_dict)
    return Ts, log


def get_coupling_egw(data, eps=5e-3):
    X_dict = data[0]
    Y_dict = data[1]
    Ts, log = egw_pgd(X_dict, Y_dict, epsilon=eps)
    return Ts, log


def get_coupling_gw_all(data):
    """Run GW all-to-all, ignore labels."""
    X_dict = data[0]
    Y_dict = data[1]
    Xtot = np.concatenate([X_dict[l] for l in X_dict.keys()], axis=0)
    Ytot = np.concatenate([Y_dict[l] for l in X_dict.keys()], axis=0)
    T, log = gw_all(Xtot, Ytot)
    return T, log


def get_coupling_egw_all(data, eps=5e-3):
    """Run GW all-to-all, ignore labels."""
    X_dict = data[0]
    Y_dict = data[1]
    Xtot = np.concatenate([X_dict[l] for l in X_dict.keys()], axis=0)
    Ytot = np.concatenate([Y_dict[l] for l in X_dict.keys()], axis=0)
    T, log = egw_all(Xtot, Ytot, eps)

    return T, log
