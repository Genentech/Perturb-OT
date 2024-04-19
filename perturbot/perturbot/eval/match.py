"""Evaluate the quality of OT"""

from typing import Dict, Tuple, Union, Sequence, List
import numpy as np
import pandas as pd
from perturbot.eval.utils import foscttm
from perturbot.utils import mdict_to_matrix


def get_rel_mse(T_dict):
    rel_err = {}
    for k, T in T_dict.items():
        T = T / T.sum()
        perfect_match = np.identity(
            T_dict[k].shape[0],
        )
        perfect_match /= perfect_match.sum()
        err = np.mean((np.diag(T_dict[k]) - np.diag(perfect_match)) ** 2)

        all2all = np.ones((T_dict[k].shape[0], T_dict[k].shape[1]))
        all2all /= all2all.sum()
        worst_err = np.mean((np.diag(all2all) - np.diag(perfect_match)) ** 2)

        rel_err[k] = err / worst_err

    return rel_err


def get_confusion_matrix(
    T_dict,
    Xs_dict,
    Xt_dict,
    Zs_dict,
    Zt_dict,
    norm=True,
) -> Tuple[Dict[Union[float, int], np.ndarray], pd.Series]:
    """
    Ts: Dictionary of n_l x n'_l matrices for each label l
    zs: m x m Confusion matrix is calculated where k is the number of unique values in z.
       Must contain integer values (0, ..., m-1)
    """
    labels = list(Xs_dict.keys())
    if not isinstance(T_dict, dict):
        return get_confusion_matrix_single(T_dict, Xs_dict, Xt_dict, Zs_dict, Zt_dict)
    classes = set(val for vals in Zs_dict.values() for val in vals)
    Cmat_dict = np.zeros((len(classes), len(classes)))
    diag_frac = {}
    for k in labels:
        # Cmat_dict[k] = np.zeros((len(classes), len(classes)))
        try:
            Zs = Zs_dict[k]
        except KeyError as e:
            print(e)
            print(Zs_dict.keys())
            raise e
        Zt = Zt_dict[k]
        # if norm:
        #     T = T_dict[k] / T_dict[k].sum()
        # else:
        T = T_dict[k]
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if T[i, j]:
                    Cmat_dict[int(Zs[i]), int(Zt[j])] += T[i, j]
        diag_frac = np.diag(Cmat_dict).sum()
    return Cmat_dict, diag_frac


def get_confusion_matrix_single(
    T,
    Xs_dict,
    Xt_dict,
    Zs_dict,
    Zt_dict,
) -> Tuple[np.ndarray, float]:
    classes = set(val for vals in Zs_dict.values() for val in vals)
    Cmat = np.zeros((len(classes), len(classes)))
    Zs = np.concatenate([Zs_dict[k] for k in Xs_dict.keys()])
    Zt = np.concatenate([Zt_dict[k] for k in Xs_dict.keys()])
    T = T / T.sum()
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j]:
                Cmat[int(Zs[i]), int(Zt[j])] += T[i, j]
    diag_frac = np.diag(Cmat).sum()
    return Cmat, diag_frac


def get_diag_fracs(
    T_dict: Union[np.ndarray, Dict[float, np.ndarray]],
    Xs_dict,
    Xt_dict,
    Zs_dict,
    Zt_dict,
) -> Union[Tuple[pd.Series, pd.Series], Tuple[float, float]]:
    if not isinstance(T_dict, dict):
        T = T_dict.copy()
        sidx = 0
        tidx = 0
        T_dict = {}
        for k, v in Xs_dict.items():
            T_dict[k] = T[
                sidx : (sidx + v.shape[0]), tidx : (tidx + Xt_dict[k].shape[0])
            ]
            sidx += v.shape[0]
            tidx += Xt_dict[k].shape[0]
        # return get_diag_fracs_single(T_dict, Xs_dict, Xt_dict, Zs_dict, Zt_dict)
    Cmat_dict, diag_fracs = get_confusion_matrix(
        T_dict, Xs_dict, Xt_dict, Zs_dict, Zt_dict
    )
    total_size_perfect = sum([T_dict[k].shape[0] for k in T_dict.keys()])
    T_perfect = {}
    for k in T_dict.keys():
        T_perfect[k] = (
            np.identity(
                T_dict[k].shape[0],
            )
            / total_size_perfect
        )
    Cmats_perfect, perfect_diag_fracs = get_confusion_matrix(
        T_perfect, Xs_dict, Xt_dict, Zs_dict, Zt_dict
    )
    total_size_random = sum([T_dict[k].size for k in T_dict.keys()])
    T_random = {k: np.ones(T_dict[k].shape) / total_size_random for k in T_dict.keys()}
    Cmats_random, random_diag_fracs = get_confusion_matrix(
        T_random, Xs_dict, Xt_dict, Zs_dict, Zt_dict
    )

    return diag_fracs, (diag_fracs - random_diag_fracs) / (
        perfect_diag_fracs - random_diag_fracs
    )


def get_diag_fracs_single(
    T,
    Xs_dict,
    Xt_dict,
    Zs_dict,
    Zt_dict,
):
    Cmat, diag_fracs = get_confusion_matrix_single(
        T, Xs_dict, Xt_dict, Zs_dict, Zt_dict
    )
    T_perfect = {}
    for k in Xs_dict.keys():
        T_perfect[k] = (
            np.identity(
                Xs_dict[k].shape[0],
            )
            / Xs_dict[k].shape[0]
        )
    T_perfect = mdict_to_matrix(
        T_perfect,
        np.concatenate([np.ones(Xs_dict[l].shape[0]) * l for l in Xs_dict.keys()]),
        np.concatenate([np.ones(Xt_dict[l].shape[0]) * l for l in Xs_dict.keys()]),
    )
    Cmats_perfect, perfect_diag_fracs = get_confusion_matrix_single(
        T_perfect, Xs_dict, Xt_dict, Zs_dict, Zt_dict
    )

    T_random = {
        k: np.ones((Xs_dict[k].shape[0], Xt_dict[k].shape[0])) for k in Xs_dict.keys()
    }
    T_random = mdict_to_matrix(
        T_random,
        np.concatenate([np.ones(Xs_dict[l].shape[0]) * l for l in Xs_dict.keys()]),
        np.concatenate([np.ones(Xt_dict[l].shape[0]) * l for l in Xs_dict.keys()]),
    )
    Cmats_random, random_diag_fracs = get_confusion_matrix_single(
        T_random, Xs_dict, Xt_dict, Zs_dict, Zt_dict
    )

    return diag_fracs, (diag_fracs - random_diag_fracs) / (
        perfect_diag_fracs - random_diag_fracs
    )


def get_FOSCTTM(
    T_dict, Xs_dict, Xt_dict, use_barycenter=True, use_agg="mean"
) -> Union[Tuple[Dict[float, Sequence], pd.Series], Tuple[List[float], float]]:
    """Obtain the fraction of cells with higher assignment probability than the
    true match (assume the diagonal is the true match.). Barycenter is used as the prediction.

    Returns
    foscttm_dict: label to the list of FOSCTTM per cells
    median_foscttm_dict: label to the aggregated FOSCTTM
    """
    agg_fn = np.nanmedian if use_agg == "median" else np.nanmean
    if isinstance(T_dict, dict):
        T = mdict_to_matrix(
            T_dict,
            np.concatenate([np.ones(Xs_dict[l].shape[0]) * l for l in Xs_dict.keys()]),
            np.concatenate([np.ones(Xt_dict[l].shape[0]) * l for l in Xs_dict.keys()]),
        )
    else:
        T = T_dict

    foscttm_dict = {}
    median_foscttm_dict = {}
    Xs_true = np.concatenate([Xs_dict[l] for l in Xs_dict.keys()])
    Xt_true = np.concatenate([Xt_dict[l] for l in Xt_dict.keys()])
    if use_barycenter:
        marg = T.sum(axis=-1)
        marg[marg == 0] = 1e-30
        Xt_pred = (T / marg[:, None]) @ Xt_true
        foscttm_ = foscttm(Xt_pred, Xt_true)
    else:
        foscttm_ = foscttm(Xs_true, Xt_true)
    return foscttm_, agg_fn(foscttm_)
    for l, Xt_true in Xt_dict.items():
        Xs_true = Xs_dict[l]
        if use_barycenter:
            T = T_dict[l]
            marg = T.sum(axis=-1)
            marg[marg == 0] = 1e-30
            Xt_pred = (T / marg[:, None]) @ Xt_true
            foscttm_ = foscttm(Xt_pred, Xt_true)
        else:
            foscttm_ = foscttm(Xs_true, Xt_true)
        foscttm_dict[l] = foscttm_
        median_foscttm_dict[l] = agg_fn(foscttm_)
    return foscttm_dict, pd.Series(median_foscttm_dict)


def get_FOSCTTM_single(
    T, Xs_dict, Xt_dict, use_barycenter=True
) -> Tuple[List[float], float]:
    """Obtain the fraction of cells with higher assignment probability than the
    true match (assume the diagonal is the true match.). Barycenter is used as the prediction.
    """
    Xs = np.concatenate([Xs_dict[l] for l in Xs_dict.keys()], axis=0)
    Xt = np.concatenate([Xt_dict[l] for l in Xt_dict.keys()], axis=0)

    if use_barycenter:
        marg = T.sum(axis=-1)
        marg[marg == 0] = 1e-30
        Xt_pred = (T / marg[:, None]) @ Xt
        foscttm_ = foscttm(Xt_pred, Xt)
    else:
        foscttm_ = foscttm(Xs, Xt)

    return foscttm_, np.nanmedian(foscttm_)
