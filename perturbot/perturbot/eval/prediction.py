import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from perturbot.eval.utils import foscttm


def _pearson_rowwise(A, B, eps=1e-8):
    assert A.shape[0] == B.shape[0]
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = np.einsum("ij,ij->i", A_mA, A_mA)
    ssB = np.einsum("ij,ij->i", B_mB, B_mB)
    return np.einsum("ij,ij->i", A_mA, B_mB) / (np.sqrt(ssA * ssB) + eps)


def _spearman_rowwise(A, B):
    assert A.shape[0] == B.shape[0]
    scorr = []
    for i in range(A.shape[0]):
        scorr.append(
            spearmanr(
                A[i, :],
                B[i, :],
            )[0]
        )
    return scorr


def get_corrs(Y_pred, Y_true, idx=None):
    if idx is not None:
        Y_pred = Y_pred[:, idx]
        Y_true = Y_true[:, idx]
    pearson = _pearson_rowwise(Y_pred, Y_true)
    spearman = _spearman_rowwise(Y_pred, Y_true)
    return pearson, spearman


def mse(Y_pred, Y_true, idx=None):
    if idx is None:
        return (np.abs(Y_pred - Y_true) ** 2).mean(axis=1)
    else:
        return (np.abs(Y_pred[:, idx] - Y_true[:, idx]) ** 2).mean(axis=1)


def get_evals(
    Y_pred,
    Y_true,
    idx=None,
    idx_id="subset",
    prediction_id="pred",
    full=True,
    agg_method="mean",
    norm_Y: np.ndarray = None,
):
    if agg_method == "median":
        agg = np.median
    elif agg_method == "mean":
        agg = np.mean
    else:
        raise ValueError()
    if norm_Y is not None:
        pearson, spearman = get_corrs(
            Y_pred / norm_Y[None, :],
            Y_true / norm_Y[None, :],
        )
    else:
        pearson, spearman = get_corrs(Y_pred, Y_true)
    pearson_c, spearman_c = get_corrs(Y_pred.T, Y_true.T)
    _mse = mse(Y_pred, Y_true)
    metrics = pd.Series(
        [agg(pearson), agg(spearman), agg(pearson_c), agg(spearman_c), agg(_mse)],
        index=[
            "Pearson_corr",
            "Spearman_corr",
            "Pearson_samples",
            "Spearman_samples",
            "MSE",
        ],
    ).to_frame()
    if idx is not None:
        if norm_Y is not None:
            pearson_idx, spearman_idx = get_corrs(
                Y_pred / norm_Y[None, :], Y_true / norm_Y[None, :], idx
            )
        else:
            pearson_idx, spearman_idx = get_corrs(Y_pred, Y_true, idx)
        pearson_idx_c, spearman_idx_c = get_corrs(Y_pred, Y_true, idx)
        _foscttm_idx = foscttm(Y_pred, Y_true, idx)
        _mse_idx = mse(Y_pred, Y_true, idx)
        metrics = pd.concat(
            [
                metrics,
                pd.Series(
                    [
                        agg(pearson_idx),
                        agg(spearman_idx),
                        agg(pearson_idx_c),
                        agg(spearman_idx_c),
                        agg(_mse_idx),
                    ],
                    index=[
                        f"Pearson_corr_{idx_id}",
                        f"Spearman_corr_{idx_id}",
                        f"Pearson_samples_{idx_id}",
                        f"Spearman_samples_{idx_id}",
                        f"MSE_{idx_id}",
                    ],
                ).to_frame(),
            ],
            axis=0,
        )
    metrics.columns = [prediction_id]
    if full:
        if idx is not None:
            full_metrics = pd.DataFrame(
                {
                    "metric": np.concatenate(
                        [
                            pearson,
                            spearman,
                            pearson_c,
                            spearman_c,
                            _mse,
                            pearson_idx,
                            spearman_idx,
                            pearson_idx_c,
                            spearman_idx_c,
                            _mse_idx,
                        ]
                    ),
                    "group": np.repeat(
                        [
                            "Pearson_corr",
                            "Spearman_corr",
                            "Pearson_samples",
                            "Spearman_samples",
                            "MSE",
                            f"Pearson_corr_{idx_id}",
                            f"Spearman_corr_{idx_id}",
                            f"Pearson_samples_{idx_id}",
                            f"Spearman_samples_{idx_id}",
                            f"MSE_{idx_id}",
                        ],
                        len(pearson),
                    ),
                    "pred_label": prediction_id,
                }
            )
        else:
            full_metrics = pd.DataFrame(
                {
                    "metric": np.concatenate(
                        [
                            pearson,
                            spearman,
                            pearson_c,
                            spearman_c,
                            _mse,
                        ]
                    ),
                    "group": np.repeat(
                        [
                            "Pearson_corr",
                            "Spearman_corr",
                            "Pearson_samples",
                            "Spearman_samples",
                            "MSE",
                        ],
                        len(pearson),
                    ),
                    "pred_label": prediction_id,
                }
            )

        return metrics, full_metrics
    return metrics


def get_evals_preds(
    Y_true, Y_preds, pred_labels, idx=None, idx_id="subset", full=False
):
    metric_dfs = []
    if full:
        foscttm_dfs = []
    for Y_pred, pred_label in zip(Y_preds, pred_labels):
        if full:
            metrics, foscttm_df = get_evals(
                Y_true,
                Y_pred,
                idx=idx,
                idx_id=idx_id,
                full=full,
                prediction_id=pred_label,
            )
            metric_dfs.append(metrics)
            foscttm_dfs.append(foscttm_df)
        else:
            metric_dfs.append(
                get_evals(
                    Y_true,
                    Y_pred,
                    idx=idx,
                    idx_id=idx_id,
                    full=full,
                    prediction_id=pred_label,
                )
            )
    if full:
        return pd.concat(metric_dfs, axis=1), pd.conat(foscttm_dfs, axis=1)
    return pd.concat(metric_dfs, axis=1)
