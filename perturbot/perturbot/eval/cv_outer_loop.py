"""For 5-fold CV, run the single inner loop of train-validation to select hyperparameter."""

from typing import Dict, Any
import sys
import argparse
import pickle as pkl
from functools import partial
from itertools import product
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch

from perturbot.eval.utils import _pop_keys, _pop_key, get_Ts_from_nn_multKs, make_G
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
from perturbot.match.cot import get_coupling_cot, get_coupling_cot_sinkhorn
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
    parser.add_argument("eps", type=str)
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
    parser.add_argument(
        "-b",
        "--baseline",
        type=str,
        default=None,
        help="One of perfect, random, by_conc",
    )
    return parser.parse_args()


ot_method_all_to_all = ["GW_all", "EGW_all_ott", "EOT_all_ott"]
pred_from_param = predict


def main(args):
    if args.log_filepath is not None and args.log_filepath != "None":
        with open(args.log_filepath, "rb") as f:
            prev_logs = pkl.load(f)
    logs = {
        "matching_evals": {},
        "pred_evals": {},
        "T": {},
        "pred": {},
        "log": {},
        "eps": {},
    }
    try:
        match_eps, lin_eps, pred_eps = tuple(map(float, args.eps.split(",")))
        logs["eps"]["match"] = match_eps
        logs["eps"]["lin"] = lin_eps
        logs["eps"]["pred"] = pred_eps
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
        dim_X = pred_data_dict["Xs_dict"][labels[0]].shape[1]
        dim_Y = pred_data_dict["Xt_dict"][labels[0]].shape[1]
        cv = KFold(n_splits=5)
        train_idx, test_idx = list(cv.split(labels))[args.test_idx]

        test_labels = [labels[tidx] for tidx in test_idx]
        train_X = _pop_keys(X_dict, test_labels)
        train_Y = _pop_keys(Y_dict, test_labels)
        train_Z = _pop_keys(Zs_dict, test_labels)
        if args.pred_filepath is not None:
            train_X_full = _pop_keys(pred_data_dict["Xs_dict"], test_labels)
            train_Y_full = _pop_keys(pred_data_dict["Xt_dict"], test_labels)
        train_Z = _pop_keys(Zs_dict, test_labels)
        train_data = (train_X, train_Y)
        train_data_full = (train_X_full, train_Y_full)
        print(f"Calculating matching with {match_eps}")

        # Get matching
        if args.log_filepath is not None and args.log_filepath != "None":
            Ts_matching = prev_logs["T"]["match"]
            Ts_pred = prev_logs["T"]["pred"]
            try:
                log_matching = prev_logs["log"]["match"]
            except:
                log_matching = None
            try:
                log_matching_predeps = prev_logs["log"]["match_pred"]
            except:
                log_matching_predeps = None
        elif args.baseline is not None:
            if args.baseline == "perfect":
                Ts_matching = Ts_pred = {
                    k: np.diag(np.ones(v.shape[0])) for k, v in train_X.items()
                }
            elif args.baseline == "random":
                Ts_matching = Ts_pred = {
                    k: np.ones((v.shape[0], v.shape[0])) for k, v in train_X.items()
                }
            elif args.baseline == "by_conc":
                Ts_matching = Ts_pred = {
                    k: make_G(train_X[k].shape[0], train_Z[k], k)
                    for k in train_X.keys()
                }
            log_matching = None
            log_matching_predeps = None
        else:
            if "VAE" in args.method:
                Ts_matching, log_matching = ot_method_map[args.method](
                    train_data_full, match_eps
                )
            else:
                Ts_matching, log_matching = ot_method_map[args.method](
                    train_data, match_eps
                )

            if "VAE" in args.method:
                if match_eps == pred_eps:
                    Ts_pred = Ts_matching
                    log_matching_predeps = None
                else:
                    Ts_pred, log_matching_predeps = ot_method_map[args.method](
                        train_data_full, pred_eps
                    )
            else:
                if match_eps != pred_eps:
                    print(f"Calculating pred matching with {pred_eps}")
                    Ts_pred, log_matching_predeps = ot_method_map[args.method](
                        train_data, pred_eps
                    )
                else:
                    Ts_pred = Ts_matching
                    log_matching_predeps = None
        logs["T"]["match"] = Ts_matching
        logs["T"]["pred"] = Ts_pred

        if "VAE" in args.method:
            latent_Y = infer_from_Ys(train_Y_full, Ts_matching, dim_X)
            latent_X = infer_from_Xs(train_X_full, Ts_matching, dim_Y)
            _, mean_foscttm = get_FOSCTTM(
                Ts_matching,
                latent_X,
                latent_Y,
                use_agg="mean",
                use_barycenter=False,
            )
            ks = [1, 5, 10, 50, 100]
            k_to_Ts = get_Ts_from_nn_multKs(latent_X, latent_Y, ks)  # k -> T
            dfracs = {}
            rel_dfracs = {}
            for k, Ts in k_to_Ts.items():
                dfracs[k], rel_dfracs[k] = get_diag_fracs(
                    Ts, train_X, train_Y, train_Z, train_Z
                )

        else:
            # normalize
            if isinstance(Ts_matching, dict):
                total_sum = 0
                for k, v in Ts_matching.items():
                    total_sum += v.sum()
                Ts_matching = {
                    k: v.astype(np.double) / total_sum for k, v in Ts_matching.items()
                }
            else:
                Ts_matching = Ts_matching.astype(np.double) / Ts_matching.sum()
            _, mean_foscttm = get_FOSCTTM(Ts_matching, train_X, train_Y, use_agg="mean")
            dfracs, rel_dfracs = get_diag_fracs(
                Ts_matching, train_X, train_Y, train_Z, train_Z
            )

        # if not all_to_all:
        mean_mean_foscttm = mean_foscttm.mean()
        # else:
        # mean_mean_foscttm = mean_foscttm
        print("foscttm", mean_foscttm)
        logs["matching_evals"] = (
            {
                "foscttm": mean_foscttm,
                "mean_foscttm": mean_mean_foscttm,
                "dfracs": dfracs,
                "rel_dfracs": rel_dfracs,
            },
        )
        # Eval prediction: full features
        test_X_full = np.concatenate(
            [pred_data_dict["Xs_dict"][l] for l in test_labels], axis=0
        )
        test_Y_full = np.concatenate(
            [pred_data_dict["Xt_dict"][l] for l in test_labels], axis=0
        )
        if "VAE" not in args.method:
            trained_model, log_pred = train_mlp((train_X_full, train_Y_full), Ts_pred)
            Y_pred_full = trained_model(torch.tensor(test_X_full)).detach().numpy()
        else:
            Y_pred_full = predict_from_model(test_X_full, Ts_pred, dim_Y)
        logs["pred"] = {
            "model": trained_model if "VAE" not in args.method else None,
            "Y_pred": Y_pred_full,
            "Y_true": test_Y_full,
            "train_Y": train_Y_full,
            "test_Z": {k: Zs_dict[k] for k in test_labels},
        }
        eval_df_full = get_evals(
            test_Y_full,
            Y_pred_full,
            prediction_id="eval",
            full=False,
            agg_method="mean",
        )
        print(eval_df_full)
        logs["pred_evals"]["full"] = eval_df_full

        # # Eval prediction - PC
        # test_X = np.concatenate([X_dict[l] for l in test_labels], axis=0)
        # test_Y = np.concatenate([Y_dict[l] for l in test_labels], axis=0)
        # if "VAE" in args.method:
        #     pred_Ys = predict_from_model(test_X, Ts_lin, dim_Y)

        # else:
        #     params = [
        #         lin_method(train_X, train_Y, Ts_lin)
        #         for lin_method in [ols_normed, weight_1_ols_normed, weight_conc_normed]
        #     ]
        #     pred_Ys = [pred_from_param(test_X, param) for param in params]
        #     pred_labels = ["ot", "perfect", "random", "by_conc"]
        # eval_df = get_evals_preds(
        #     test_Y,
        #     pred_Ys,
        #     pred_labels=pred_labels,
        #     full=False,
        # )
        # logs["pred_evals"]["PC"] = eval_df
        # print(eval_df)

        logs["log"] = (
            {
                "match": log_matching,
                "match_pred": log_matching_predeps,
                # "match_lin": log_matching_lin,
                "mlp": log_pred if "VAE" not in args.method else None,
            },
        )
    except Exception as e:
        with open(f"test_{args.method}.{args.test_idx}.tmp.pkl", "wb") as f:
            pkl.dump(logs, f)
        raise e

    with open(f"test_{args.method}.{args.test_idx}.e.pkl", "wb") as f:
        pkl.dump(logs, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
