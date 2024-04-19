from typing import Sequence, List
import numpy as np
from sklearn.neighbors import NearestNeighbors


def make_G(size, label, k):
    assert size == len(label), (k, size, len(label))
    G = np.zeros((size, size))
    for l in np.unique(label):
        l_idx = np.where(label == l)[0]
        for i in l_idx:
            for j in l_idx:
                G[i, j] = 1.0
    assert (G.sum(axis=0) > 0).all(), (k, (G.sum(axis=0) == 0).sum())
    return G


def foscttm(Y_pred, Y_true, idx=None) -> List[float]:
    """
    From https://github.com/rsinghlab/SCOT/blob/9787d0a6a1a494059cb3fb49e8ceda9318273c06/src/.ipynb_checkpoints/evals-checkpoint.py#L32
        Returns fraction closer than true match for each sample (as an array)
    """
    if idx is not None:
        Y_pred = Y_pred[:, idx]
        Y_true = Y_true[:, idx]
    fracs = []
    nsamp = Y_pred.shape[0]
    rank = 0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(
            np.sum(np.square(np.subtract(Y_pred[row_idx, :], Y_true)), axis=1)
        )
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        try:
            rank = np.where(sort_euc_dist == true_nbr)[0].mean()
        except FloatingPointError:
            print(true_nbr)
            print(sort_euc_dist)
            np.where(np.abs(sort_euc_dist - true_nbr < 1e-8))[0]
            rank = np.where(np.abs(sort_euc_dist - true_nbr < 1e-8))[0].mean()

        frac = float(rank) / (nsamp - 1)
        fracs.append(frac)
    return fracs


def get_T_from_nn(X, Y, k):
    """Obtain kNN label of observation in X from the k nearest neighbors in Y"""
    nsamp = X.shape[0]
    T = np.zeros((X.shape[0], Y.shape[0]))
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(X[row_idx, :], Y)), axis=1))
        smallest_k_idx = np.argpartition(euc_dist, k)[:k]
        T[row_idx, smallest_k_idx] = 1.0 / (nsamp * k)
    return T


def get_Ts_from_nn_multKs(X_dict, Y_dict, ks):
    """Obtain kNN label of observation in X from the k nearest neighbors in Y"""
    X = np.concatenate([X_dict[l] for l in X_dict.keys()])
    Y = np.concatenate([Y_dict[l] for l in X_dict.keys()])
    k_to_T = {k: np.zeros((X.shape[0], Y.shape[0])) for k in ks}
    nsamp = X.shape[0]
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(X[row_idx, :], Y)), axis=1))
        for k in ks:
            smallest_k_idx = np.argpartition(euc_dist, k)[:k]
            k_to_T[k][row_idx, smallest_k_idx] = 1.0 / (nsamp * k)
    k_to_Tdict = {}
    for k, T in k_to_T.items():
        i = 0
        j = 0
        k_to_Tdict[k] = {
            l: np.zeros((X_dict[l].shape[0], Y_dict[l].shape[0])) for l in X_dict.keys()
        }
        for l, X in X_dict.items():
            k_to_Tdict[k][l] = T[
                i : (i + X_dict[l].shape[0]), j : (j + Y_dict[l].shape[0])
            ]
            i += X_dict[l].shape[0]
            j += Y_dict[l].shape[0]
            # k_to_Tdict[k][l] = k_to_Tdict[k][l] / k_to_Tdict[k][l].sum()

    return k_to_Tdict


def _pop_key(d, k, sub_key=None):
    d = d.copy()
    if sub_key is None:
        del d[k]
    else:
        del d[sub_key][k]
    return d


def _pop_keys(d, ks, sub_key=None):
    d = d.copy()
    if sub_key is None:
        for k in ks:
            del d[k]
    else:
        for k in ks:
            del d[sub_key][k]
    return d
