"""For 5-fold CV, run the single inner loop of train-validation to select hyperparameter."""

from typing import Dict, Any
import sys
import argparse
import pickle as pkl
import jax
from functools import partial
from itertools import product
from multiprocessing import Pool
import numpy as np
import pandas as pd
from perturbot.eval.utils import get_Ts_from_nn_multKs
from sklearn.model_selection import KFold
import torch
from tqdm import tqdm
import psutil
import gc
from perturbot.eval.utils import _pop_keys, _pop_key
from perturbot.eval.match import get_FOSCTTM, get_diag_fracs
from perturbot.eval.prediction import get_evals_preds, get_evals
from perturbot.preprocess.vae import (
    train_vae_rna,
    train_vae_acc,
    train_vae_prot,
    SCVI_LATENT_KEY,
)
from perturbot.match.cot_labels import get_coupling_cotl, get_coupling_cotl_sinkhorn
from perturbot.match.gw import (
    get_coupling_gw_cg,
    get_coupling_egw,
    get_coupling_egw_all,
    get_coupling_gw_all,
)
from perturbot.match.ott_egwl import (
    get_coupling_egw_labels_ott,
    get_coupling_egw_all_ott,
    get_coupling_eot_ott,
    get_coupling_leot_ott,
    get_coupling_egw_ott,
)
from perturbot.match.cot import (
    get_coupling_cot,
    get_coupling_cot_sinkhorn,
    get_coupling_each_cot_sinkhorn,
)
from perturbot.match.gw_labels import get_coupling_gw_labels, get_coupling_egw_labels
from perturbot.predict.scvi_vae import train_vae_model
from perturbot.predict.linear_regression import (
    ols_normed,
    weight_conc_normed,
    weighted_ols_normed,
    weight_1_ols_normed,
    predict,
)
from perturbot.predict.mlp import train_mlp
from perturbot.predict.scvi_vae import predict_from_model, infer_from_Xs, infer_from_Ys

ot_method_map = {
    "ECOOTL": get_coupling_cotl_sinkhorn,
    "ECOOT_each": get_coupling_each_cot_sinkhorn,
    "ECOOT": get_coupling_cot_sinkhorn,
    "EGWL": get_coupling_egw_labels,
    "EOT_ott": get_coupling_eot_ott,
    "LEOT_ott": get_coupling_leot_ott,
    "EGW_ott": get_coupling_egw_ott,
    "EGW_all_ott": get_coupling_egw_all_ott,
    "EGWL_ott": get_coupling_egw_labels_ott,
    "VAE_label": train_vae_model,
    "VAE": partial(train_vae_model, use_label=False),
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Run inner validation loop of CV",
        "Run sample-matching OT in parallel and fit prediction model in leave-one-out mannter",
    )
    parser.add_argument("method", type=str)
    parser.add_argument("test_idx", type=int)
    parser.add_argument("filepath", type=str)
    parser.add_argument("-m", "--mlp", action="store_true")
    parser.add_argument("--baseline", action="store_true")

    parser.add_argument(
        "-p",
        "--pred-filepath",
        type=str,
        default=None,
        help="Path to data.pkl file with full features.",
    )
    parser.add_argument(
        "-l",
        "--log-filepath",
        type=str,
        default=None,
        help="Path to log.pkl if transport was already run. If not None, reevaluates but does not calculates the transport again.",
    )
    return parser.parse_args()


ot_method_hyperparams = {}
for method in [
    "EGWL",
    "EOT_ott",
    "LEOT_ott",
    "EGW_ott",
    "EGW_all_ott",
    "EGWL_ott",
    "ECOOT",
    "ECOOT_each",
    "ECOOTL",
]:
    ot_method_hyperparams[method] = [
        0.1,
        1e-2,
        1e-3,
        1e-4,
        1e-5,
    ]
vae_adv_loss = [1, 5, 10, 50, 100]
vae_latent_dims = [128]
vae_learning_rates = [1e-4]
# vae_latent_dims = [32]
# vae_learning_rates = [1e-4]
for method in ["VAE", "VAE_label"]:
    ot_method_hyperparams[method] = list(
        product(vae_adv_loss, vae_latent_dims, vae_learning_rates)
    )

ot_method_all_to_all = ["GW_all", "EGW_all_ott", "EOT_all_ott"]
pred_method = weighted_ols_normed
baseline_pred_methods = [ols_normed, weight_1_ols_normed, weight_conc_normed]
baseline_pred_method_labels = ["perfect", "random", "by_conc"]
pred_from_param = predict


def main(args):
    epsilons = ot_method_hyperparams[args.method]
    all_to_all = args.method in ot_method_all_to_all
    with open(args.filepath, "rb") as f:
        data_dict = pkl.load(f)
    if args.pred_filepath is not None:
        with open(args.pred_filepath, "rb") as f:
            pred_data_dict = pkl.load(f)

    X_dict = data_dict["Xs_dict"]
    Y_dict = data_dict["Xt_dict"]
    Zs_dict = data_dict["Zs_dict"]["dosage"]
    Zt_dict = data_dict["Zt_dict"]["dosage"]

    labels = list(X_dict.keys())
    dim_X = X_dict[labels[0]].shape[1]
    dim_Y = Y_dict[labels[0]].shape[1]
    cv = KFold(n_splits=5)
    train_val_idx, test_idx = list(cv.split(labels))[args.test_idx]
    cv_inner = KFold(n_splits=5)

    test_labels = [labels[tidx] for tidx in test_idx]
    train_val_X = _pop_keys(X_dict, test_labels)
    train_val_Y = _pop_keys(Y_dict, test_labels)
    if args.pred_filepath is not None:
        train_val_X_full = _pop_keys(pred_data_dict["Xs_dict"], test_labels)
        train_val_Y_full = _pop_keys(pred_data_dict["Xt_dict"], test_labels)
    train_val_Z = _pop_keys(Zs_dict, test_labels)

    train_val_idx_split = cv_inner.split(train_val_idx)
    val_labels_all = []
    train_Zs = []
    train_data = []
    train_val_labels = [labels[tvidx] for tvidx in train_val_idx]
    for train_idx, val_idx in train_val_idx_split:
        val_labels = tuple([train_val_labels[vidx] for vidx in val_idx])
        val_labels_all.append(val_labels)
        train_X = _pop_keys(train_val_X, val_labels)
        train_Y = _pop_keys(train_val_Y, val_labels)
        train_Z = _pop_keys(train_val_Z, val_labels)
        train_Zs.append(train_Z)
        train_data.append((train_X, train_Y))

    # Run 5-fold inner CV for all eps
    eps_prod, train_data_prod = zip(*product(epsilons, train_data))
    val_labels_prod = val_labels_all * len(epsilons)
    train_Z_prod = train_Zs * len(epsilons)
    print("Len eps", eps_prod)
    if args.log_filepath is None and "VAE" not in args.method:
        Ts_list = []
        logs = []
        # for i, (_train_data, _eps) in tqdm(enumerate(zip(train_data_prod, eps_prod))):
        #     print(f"{i}th iteration with {_eps}")
        #     _T, _log = ot_method_map[args.method](_train_data, _eps)
        #     Ts_list.append(_T)
        #     logs.append(_log)
        #     jax.clear_caches()
        try:
            with Pool(5 * len(epsilons)) as p:
                # with Pool(5) as p:
                Ts_list, logs = zip(
                    *p.starmap(
                        ot_method_map[args.method], zip(train_data_prod, eps_prod)
                    )
                )

        except KeyboardInterrupt:
            p.terminate()
        finally:
            p.join()
    elif args.log_filepath is not None:
        with open(args.log_filepath, "rb") as f:
            prev_log = pkl.load(f)
        Ts_list = [None] * len(train_data_prod)
        logs = [None] * len(train_data_prod)
        if "VAE" not in args.method:
            epsilons = [
                e for e in epsilons if e in np.unique(list(prev_log["T"].keys()))
            ]
            eps_prod, train_data_prod = zip(*product(epsilons, train_data))
            val_labels_prod = val_labels_all * len(epsilons)
    else:
        Ts_list = []
        logs = []
        for train_data, eps in zip(train_data_prod, eps_prod):
            t, l = ot_method_map[args.method](train_data, eps)
            Ts_list.append(t)
            logs.append(l)

    # Evaluate & select best eps
    eps_to_pred_evals = {eps: [] for eps in epsilons}
    eps_to_pred_full_evals = {eps: [] for eps in epsilons}
    eps_to_matching_evals = {eps: [] for eps in epsilons}
    eps_to_dfracs_evals = {eps: [] for eps in epsilons}
    eps_to_val_to_T = {eps: {} for eps in epsilons}
    eps_to_val_to_log = {eps: {} for eps in epsilons}
    iters = zip(eps_prod, val_labels_prod, train_data_prod, Ts_list, logs)
    assert len(list(iters)) == len(eps_prod)
    print(f"Ts_list: (len {len(Ts_list)})")
    for eps, val_labels, train_pair, Ts, train_Z, log in zip(
        eps_prod, val_labels_prod, train_data_prod, Ts_list, train_Z_prod, logs
    ):
        if args.log_filepath is not None:
            Ts = prev_log["T"][eps][val_labels]
            log = prev_log["log"][eps][val_labels]
        # Ts: Dict[Union[float,int], np.ndarray] for each label, transport plan
        val_X_dict = {tidx: train_val_X[tidx] for tidx in val_labels}
        val_Y_dict = {tidx: train_val_Y[tidx] for tidx in val_labels}
        if args.pred_filepath is not None:
            val_X_dict_full = {tidx: train_val_X_full[tidx] for tidx in val_labels}
            val_Y_dict_full = {tidx: train_val_Y_full[tidx] for tidx in val_labels}
        train_X, train_Y = train_pair
        print(f"in the loop; {eps}, {val_labels}")

        if isinstance(Ts, int):
            # COOT underflow
            print("Underflow", eps, val_labels)
            eps_to_matching_evals[eps].append(100)
            for tidx in val_labels:
                eps_to_pred_evals[eps].append(
                    pd.Series(
                        [np.nan, np.nan, np.nan, np.nan, np.nan],
                        index=[
                            "Pearson_corr",
                            "Spearman_corr",
                            "Pearson_samples",
                            "Spearman_samples",
                            "MSE",
                        ],
                    ).to_frame()
                )
            if args.pred_filepath is not None:
                for tidx in val_labels:
                    eps_to_pred_full_evals[eps].append(
                        pd.Series(
                            [np.nan, np.nan, np.nan, np.nan, np.nan],
                            index=[
                                "Pearson_corr",
                                "Spearman_corr",
                                "Pearson_samples",
                                "Spearman_samples",
                                "MSE",
                            ],
                        ).to_frame()
                    )
            eps_to_val_to_T[eps][val_labels] = Ts
            eps_to_val_to_log[eps][val_labels] = {}
            continue
        # Evaluate matching
        if "VAE" in args.method:
            ks = [5, 10, 25, 50]
            latent_Y = infer_from_Ys(train_Y, Ts, dim_X)
            latent_X = infer_from_Xs(train_X, Ts, dim_Y)
            _, mean_foscttm = get_FOSCTTM(
                Ts, latent_X, latent_Y, use_agg="mean", use_barycenter=False
            )

            Ts_multK = get_Ts_from_nn_multKs(latent_X, latent_Y, ks)  # k -> T
            diag_fracs_dict = {}
            for k, T_k in Ts_multK.items():
                dfracs, rel_dfracs = get_diag_fracs(
                    T_k, train_X, train_Y, train_Z, train_Z
                )
                diag_fracs_dict[k] = rel_dfracs
            eps_to_dfracs_evals[eps].append(diag_fracs_dict)
        else:
            _, mean_foscttm = get_FOSCTTM(Ts, train_X, train_Y, use_agg="mean")
            dfracs, rel_dfracs = get_diag_fracs(Ts, train_X, train_Y, train_Z, train_Z)

            eps_to_dfracs_evals[eps].append(rel_dfracs)
        eps_to_matching_evals[eps].append(mean_foscttm.mean())
        print("map", eps_to_matching_evals[eps])
        print("foscttm", mean_foscttm)

        # Evaluate prediction
        for tidx in val_labels:
            val_X = val_X_dict[tidx]
            val_Y = val_Y_dict[tidx]
            if "VAE" in args.method:
                pred = predict_from_model(val_X, Ts, dim_Y)
            else:
                param = pred_method(train_X, train_Y, Ts)
                pred = pred_from_param(val_X, param)
            try:
                df = get_evals(
                    val_Y,
                    pred,
                    prediction_id=(eps, val_labels),
                    full=False,
                    agg_method="mean",
                )
            except:
                df = pd.Series(
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    index=[
                        "Pearson_corr",
                        "Spearman_corr",
                        "Pearson_samples",
                        "Spearman_samples",
                        "MSE",
                    ],
                ).to_frame()
            eps_to_pred_evals[eps].append(df)
        if args.pred_filepath is not None:
            for tidx in val_labels:
                train_X_full = _pop_keys(train_val_X_full, val_labels)
                train_Y_full = _pop_keys(train_val_Y_full, val_labels)
                val_X_full = val_X_dict_full[tidx]
                val_Y_full = val_Y_dict_full[tidx]
                if "VAE" in args.method:
                    pred = predict_from_model(val_X, Ts, dim_Y)
                else:
                    param = pred_method(train_X_full, train_Y_full, Ts)
                    pred = pred_from_param(val_X_full, param)
                eps_to_pred_full_evals[eps].append(
                    get_evals(
                        val_Y_full,
                        pred,
                        prediction_id=(eps, val_labels),
                        full=False,
                        agg_method="mean",
                    )
                )
        eps_to_val_to_T[eps][val_labels] = Ts
        eps_to_val_to_log[eps][val_labels] = log
    print(eps_to_matching_evals)
    eps_to_matching_evals = pd.Series(
        {k: np.nanmean(v) for k, v in eps_to_matching_evals.items()},
    )
    max_matching_eps = eps_to_matching_evals.idxmin()
    eps_to_pred_evals = pd.DataFrame(
        {k: pd.concat(v, axis=1).mean(axis=1) for k, v in eps_to_pred_evals.items()}
    )
    max_pred_eps = eps_to_pred_evals.loc["MSE"].idxmin()
    best_eps = {"matching": max_matching_eps, "pred": max_pred_eps}
    if args.pred_filepath is not None:
        eps_to_pred_full_evals = pd.DataFrame(
            {
                k: pd.concat(v, axis=1).mean(axis=1)
                for k, v in eps_to_pred_full_evals.items()
            }
        )
        max_pred_full_eps = eps_to_pred_full_evals.loc["Pearson_samples"].idxmax()
        best_eps["pred_full"] = eps_to_pred_full_evals.loc["Pearson_samples"].idxmax()
    val_logs = {
        "matching_evals": eps_to_matching_evals,
        "dfracs": eps_to_dfracs_evals,
        "pred_evals": eps_to_pred_evals,
        "T": eps_to_val_to_T,
        "log": eps_to_val_to_log,
        "best_eps": best_eps,
    }
    if args.pred_filepath is not None:
        val_logs["pred_full_evals"] = eps_to_pred_full_evals

    edited_label = "e." if args.log_filepath is not None else ""
    with open(
        f"val_CV_{args.method}.{args.test_idx}.best_eps.{edited_label}pkl",
        "wb",
    ) as f:
        pkl.dump(best_eps, f)

    with open(
        f"val_CV_{args.method}.{args.test_idx}.{edited_label}pkl",
        "wb",
    ) as f:
        pkl.dump(val_logs, f)


def run_mlp(args):
    epsilons = [e for e in ot_method_hyperparams[args.method] if e != 0.1 and e != 1e-6]
    with open(args.log_filepath, "rb") as f:
        log = pkl.load(f)
    with open(args.filepath, "rb") as f:
        data_dict = pkl.load(f)

    X_dict = data_dict["Xs_dict"]
    Y_dict = data_dict["Xt_dict"]
    Z_dict = data_dict["Zs_dict"]["dosage"]

    labels = list(X_dict.keys())
    dim_X = X_dict[labels[0]].shape[1]
    dim_Y = Y_dict[labels[0]].shape[1]
    cv = KFold(n_splits=5)
    train_val_idx, test_idx = list(cv.split(labels))[args.test_idx]
    cv_inner = KFold(n_splits=5)

    test_labels = [labels[tidx] for tidx in test_idx]
    train_val_X = _pop_keys(X_dict, test_labels)
    train_val_Y = _pop_keys(Y_dict, test_labels)
    train_val_Z = _pop_keys(Z_dict, test_labels)

    train_val_idx_split = cv_inner.split(train_val_idx)
    val_labels_all = []
    train_data = []
    val_data = []
    train_val_labels = [labels[tvidx] for tvidx in train_val_idx]
    for train_idx, val_idx in train_val_idx_split:
        val_labels = tuple([train_val_labels[vidx] for vidx in val_idx])
        val_labels_all.append(val_labels)
        train_X = _pop_keys(train_val_X, val_labels)
        train_Y = _pop_keys(train_val_Y, val_labels)
        val_X = np.concatenate([train_val_X[l] for l in val_labels], axis=0)
        val_Y = np.concatenate([train_val_Y[l] for l in val_labels], axis=0)
        val_Z = np.concatenate([train_val_Z[l] for l in val_labels], axis=0)
        train_data.append((train_X, train_Y))
        val_data.append((val_X, val_Y, val_Z))

    # Run 5-fold inner CV for all eps
    eps_prod, train_data_prod = zip(*product(epsilons, train_data))
    val_labels_prod = val_labels_all * len(epsilons)
    val_data_prod = val_data * len(epsilons)
    Ts_list = []
    for eps, val_labels in zip(eps_prod, val_labels_prod):
        Ts_list.append(log["T"][eps][val_labels])

    trained_models = []
    logs = []
    for train_data, Ts in zip(train_data_prod, Ts_list):
        t, l = train_mlp(train_data, Ts)
        trained_models.append(t)
        logs.append(l)

    # Evaluate & select best eps
    eps_to_pred_evals = {eps: [] for eps in epsilons}
    eps_to_val_to_preds = {eps: {} for eps in epsilons}
    eps_to_matching_evals = {eps: [] for eps in epsilons}
    eps_to_val_to_model = {eps: {} for eps in epsilons}
    eps_to_val_to_log = {eps: {} for eps in epsilons}
    iters = zip(val_data_prod, train_data_prod, trained_models)
    assert len(list(iters)) == len(eps_prod)
    print(f"Ts_list: (len {len(Ts_list)})")
    for eps, val_labels, val_pair, model in zip(
        eps_prod, val_labels_prod, val_data_prod, trained_models
    ):
        val_X, val_Y, val_Z = val_pair
        print(f"in the loop; {val_labels}")

        # Evaluate prediction
        Y_pred = model(torch.tensor(val_X)).detach().numpy()
        eval_df = get_evals(
            val_Y,
            Y_pred,
            prediction_id=(eps, val_labels),
            full=False,
            agg_method="mean",
        )
        print(eval_df)
        eps_to_pred_evals[eps] = eps_to_pred_evals[eps] + [eval_df]
        eps_to_val_to_preds[eps][val_labels] = (Y_pred, val_Y, val_Z)
        eps_to_val_to_model[eps][val_labels] = model

    def concat_if_possible(v):
        if len(v) > 0:
            return pd.concat(v, axis=1).mean(axis=1)
        else:
            return v

    eps_to_pred_evals = pd.DataFrame(
        {k: concat_if_possible(v) for k, v in eps_to_pred_evals.items()}
    )
    # max_pred_eps = eps_to_pred_evals.loc["Pearson_c"].idxmax()
    val_logs = {
        "pred": eps_to_val_to_preds,
        "pred_evals": eps_to_pred_evals,
        "model": eps_to_val_to_model,
    }
    # with open(f"val_CV_MLP_{args.method}.{args.test_idx}.best_eps.pkl", "wb") as f:
    #     pkl.dump({"eps": max_pred_eps}, f)

    with open(f"val_CV_MLP_{args.method}.{args.test_idx}.pkl", "wb") as f:
        pkl.dump(val_logs, f)


def run_mlp_control(args):
    with open(args.filepath, "rb") as f:
        data_dict = pkl.load(f)

    X_dict = data_dict["Xs_dict"]
    Y_dict = data_dict["Xt_dict"]
    Z_dict = data_dict["Zs_dict"]["dosage"]

    labels = list(X_dict.keys())
    dim_X = X_dict[labels[0]].shape[1]
    dim_Y = Y_dict[labels[0]].shape[1]
    cv = KFold(n_splits=5)
    train_val_idx, test_idx = list(cv.split(labels))[args.test_idx]
    cv_inner = KFold(n_splits=5)

    test_labels = [labels[tidx] for tidx in test_idx]
    train_val_X = _pop_keys(X_dict, test_labels)
    train_val_Y = _pop_keys(Y_dict, test_labels)
    train_val_Z = _pop_keys(Z_dict, test_labels)

    train_val_idx_split = cv_inner.split(train_val_idx)
    val_labels_all = []
    train_data = []
    val_data = []
    train_Z = []
    train_val_labels = [labels[tvidx] for tvidx in train_val_idx]
    for train_idx, val_idx in train_val_idx_split:
        val_labels = tuple([train_val_labels[vidx] for vidx in val_idx])
        val_labels_all.append(val_labels)
        train_X = _pop_keys(train_val_X, val_labels)
        train_Y = _pop_keys(train_val_Y, val_labels)
        val_X = np.concatenate([train_val_X[l] for l in val_labels], axis=0)
        val_Y = np.concatenate([train_val_Y[l] for l in val_labels], axis=0)
        val_Z = np.concatenate([train_val_Z[l] for l in val_labels], axis=0)
        train_data.append((train_X, train_Y))
        train_Z.append(train_val_Z)
        val_data.append((val_X, val_Y, val_Z))

    # Make baseline Ts
    perf_T = {k: np.diag(np.ones(v.shape[0])) for k, v in X_dict.items()}

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

    cond_T = {k: make_G(X_dict[k].shape[0], Z_dict[k], k) for k in X_dict.keys()}
    rand_T = {k: np.ones((v.shape[0], v.shape[0])) for k, v in X_dict.items()}
    Ts_list = [perf_T, cond_T, rand_T]
    Ts_prod, train_data_prod = zip(*product(Ts_list, train_data))
    Ts_label = ["perfect", "dosage", "random"]
    Ts_label_prod = np.repeat(Ts_label, len(train_data))
    train_Z_prod = train_Z * len(Ts_label)
    val_labels_prod = val_labels_all * len(Ts_label)
    val_data_prod = val_data * len(Ts_label)

    trained_models = []
    logs = []
    for train_data, val_labels, Ts, tlab in zip(
        train_data_prod, val_labels_prod, Ts_prod, Ts_label_prod
    ):
        T = _pop_keys(Ts, test_labels)
        T = _pop_keys(T, val_labels)
        try:
            t, l = train_mlp(train_data, T)
        except Exception as e:
            print(f"error at {tlab}, {val_labels}")
            raise e
        trained_models.append(t)
        logs.append(l)

    tlab_to_pred_evals = {tlab: [] for tlab in Ts_label}
    tlab_to_matching_evals = {tlab: [] for tlab in Ts_label}
    tlab_to_val_to_model = {tlab: {} for tlab in Ts_label}
    tlab_to_val_to_log = {tlab: {} for tlab in Ts_label}
    tlab_to_val_to_pred = {tlab: {} for tlab in Ts_label}
    iters = zip(val_data_prod, train_data_prod, trained_models)
    assert len(list(iters)) == len(Ts_label_prod)
    print(f"Ts_list: (len {len(Ts_list)})")
    for Ts, tlab, val_labels, val_pair, train_data, model, train_Z, log in zip(
        Ts_prod,
        Ts_label_prod,
        val_labels_prod,
        val_data_prod,
        train_data_prod,
        trained_models,
        train_Z_prod,
        logs,
    ):
        val_X, val_Y, val_Z = val_pair
        train_X, train_Y = train_data
        print(f"in the loop; {val_labels}")
        # Evaluate matching
        T = _pop_keys(Ts, test_labels)
        T = _pop_keys(T, val_labels)
        try:
            _, mean_foscttm = get_FOSCTTM(T, train_X, train_Y, use_agg="mean")
        except KeyError as e:
            print(Ts.keys())
            print(T.keys())
            print(train_X.keys())
            raise e
        dfracs, rel_dfracs = get_diag_fracs(T, train_X, train_Y, train_Z, train_Z)
        tlab_to_matching_evals[tlab] = {
            "foscttm": mean_foscttm,
            "dfracs": dfracs,
            "rel_dfracs": rel_dfracs,
        }
        # Evaluate prediction
        Y_pred = model(torch.tensor(val_X)).detach().numpy()
        eval_df = get_evals(
            val_Y,
            Y_pred,
            prediction_id=(tlab, val_labels),
            full=False,
            agg_method="mean",
            norm_Y=Y_dict[3].mean(axis=0),
        )
        tlab_to_val_to_log[tlab] = log
        tlab_to_val_to_pred[tlab][val_labels] = (val_Y, Y_pred, val_Z)
        print(eval_df)
        tlab_to_pred_evals[tlab] = tlab_to_pred_evals[tlab] + [eval_df]
        tlab_to_val_to_model[tlab][val_labels] = model

    def concat_if_possible(v):
        if len(v) > 0:
            return pd.concat(v, axis=1).mean(axis=1)
        else:
            return v

    tlab_to_pred_evals = pd.DataFrame(
        {k: concat_if_possible(v) for k, v in tlab_to_pred_evals.items()}
    )
    val_logs = {
        "pred_evals": tlab_to_pred_evals,
        "match_evals": tlab_to_matching_evals,
        "model": tlab_to_val_to_model,
        "logs": tlab_to_val_to_log,
        "preds": tlab_to_val_to_pred,
    }

    with open(f"val_CV_MLP_baseline.{args.test_idx}.pkl", "wb") as f:
        pkl.dump(val_logs, f)


if __name__ == "__main__":
    args = parse_args()
    if args.mlp:
        if args.baseline:
            run_mlp_control(args)
        else:
            if args.log_filepath is None:
                args.log_filepath = f"val_CV_{args.method}.{args.test_idx}.pkl"
            run_mlp(args)
    else:
        main(args)
