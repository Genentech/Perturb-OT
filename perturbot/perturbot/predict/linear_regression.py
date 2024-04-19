import numpy as np
from typing import Dict, Union
from numpy.linalg import inv
from tqdm.auto import tqdm


def ols(X, Y):
    """Regular OLS where we know item-to. Note that given different number of perturbation,
    each data points are weighted as the same."""
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return inv(X.T @ X) @ (X.T @ Y)


def ols_label(X_dict, Y_dict):
    assert X_dict.keys() == Y_dict.keys()
    xtx = 0
    xty = 0
    for l, X in X_dict.items():
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        xtx += X.T @ X
        xty += X.T @ Y_dict[l]
    """Equivalent to ols"""
    return inv(xtx) @ (xty)


def weighted_ols(
    X_dict: Dict[int, np.ndarray],
    Y_dict: Dict[int, np.ndarray],
    G_dict: Dict[int, np.ndarray],
):
    """OLS for weighted matching between X and Y provided as G.
    Return B = (\sum_l{(X^l)'Diag(G_x)(X^l)})^{-1}(\sum_l{\sum_i{(X_i^l)'(G_i)(Y^l)}}).
    Each sample from the source X has the same weight.
    where
    l: label (keys of the dictionaries),
    i: sample index in X^l,
    G_i: i th row of G,
    G_x: Marginal distribution of x from G
    """
    assert X_dict.keys() == Y_dict.keys()
    xtx = 0
    xty = 0
    for l, X in tqdm(X_dict.items()):
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        Y = Y_dict[l]
        G = G_dict[l]
        G = G / G.sum(axis=-1)[:, None]
        assert X.shape[0] == Y.shape[0]
        assert X.shape[0] == G.shape[0]
        assert Y.shape[0] == G.shape[1]
        xtx += X.T @ np.diag(G.sum(axis=-1)) @ X
        for i in range(X.shape[0]):
            xty += X[[i], :].T @ (G[[i], :] @ Y)
    return inv(xtx) @ xty


def weight_1_ols(X_dict: Dict[int, np.ndarray], Y_dict: Dict[int, np.ndarray]):
    """OLS for all-to-all matching given label.
    Return B = (\sum_l{(X^l)'X^l)})^{-1}(\sum_l (X^l)'Y^l).
    Each sample from the source X has the same weight.
    """
    assert X_dict.keys() == Y_dict.keys()
    xtx = 0
    xty = 0
    for l, X in X_dict.items():
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        Y = Y_dict[l]
        xtx += X.T @ X
        xty += (
            X.sum(axis=0, keepdims=True).T @ Y.sum(axis=0, keepdims=True) / Y.shape[0]
        )
    return inv(xtx) @ xty


def predict(X, params):
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return X @ params


def weighted_ols_normed(
    X_dict: Dict[int, np.ndarray],
    Y_dict: Dict[int, np.ndarray],
    G_dict: Union[np.ndarray, Dict[int, np.ndarray]],
) -> np.ndarray:
    """OLS for weighted matching between X and Y provided as G.
    Return B = (\sum_l{(X^l)'Diag(G_x)(X^l)})^{-1}(\sum_l{\sum_i{(X_i^l)'(G_i)(Y^l)}}).
    Each sample from the source X has the same weight.
    where
    l: label (keys of the dictionaries),
    i: sample index in X^l,
    G_i: i th row of G,
    G_x: Marginal distribution of x from G
    """
    assert X_dict.keys() == Y_dict.keys()
    xtx = 0
    xty = 0
    if isinstance(G_dict, dict):
        for l, X in tqdm(X_dict.items()):
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
            Y = Y_dict[l]
            G = G_dict[l]
            G = G / G.sum()
            assert X.shape[0] == Y.shape[0]
            assert X.shape[0] == G.shape[0]
            assert Y.shape[0] == G.shape[1]
            xtx += X.T @ np.diag(G.sum(axis=-1)) @ X
            for i in range(X.shape[0]):
                xty += X[[i], :].T @ (G[[i], :] @ Y)
    else:
        try:
            G = G_dict / G_dict.sum()
        except FloatingPointError:
            G = (G_dict + 1e-6) / (G_dict.sum() + 1e-6)
        labels = X_dict.keys()
        X = np.concatenate(
            [
                np.ones((sum([X_dict[l].shape[0] for l in labels]), 1)),
                np.concatenate([X_dict[l] for l in labels], axis=0),
            ],
            axis=1,
        )
        Y = np.concatenate([Y_dict[l] for l in labels], axis=0)
        xtx += X.T @ np.diag(G.sum(axis=-1)) @ X
        for i in range(X.shape[0]):
            xty += X[[i], :].T @ (G[[i], :] @ Y)
        xty += X[[i], :].T @ (G[[i], :] @ Y)
    return inv(xtx) @ xty


def weight_1_ols_normed(X_dict, Y_dict, *args):
    G_dict = {
        k: np.ones((X_dict[k].shape[0], Y_dict[k].shape[0])) for k in X_dict.keys()
    }
    return weighted_ols_normed(X_dict, Y_dict, G_dict)


def ols_normed(X_dict, Y_dict, *args):
    G_dict = {k: np.identity(X_dict[k].shape[0]) for k in X_dict.keys()}
    return weighted_ols_normed(X_dict, Y_dict, G_dict)


def weight_conc_normed(X_dict, Y_dict, Z_dict, z_key="dosage"):
    def make_G(size, label):
        G = np.zeros((size, size))
        for l in np.unique(label):
            l_idx = np.where(label == l)[0]
            for i in l_idx:
                for j in l_idx:
                    G[i, j] = 1
        return G

    if z_key in Z_dict:
        Z_dict = Z_dict[z_key]
    G_dict = {k: make_G(X_dict[k].shape[0], Z_dict[k]) for k in X_dict.keys()}
    return weighted_ols_normed(X_dict, Y_dict, G_dict)
