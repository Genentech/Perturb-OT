"""For 5-fold CV, run the single inner loop of train-validation to select hyperparameter."""

import os
import argparse
import pickle as pkl
from functools import partial
import numpy as np
import torch
from perturbot.eval.match import get_FOSCTTM, get_diag_fracs
from perturbot.eval.utils import get_Ts_from_nn_multKs
from perturbot.match.cot_labels import get_coupling_cotl_sinkhorn

from perturbot.match.ott_egwl import (
    get_coupling_egw_labels_ott,
    get_coupling_egw_all_ott,
    get_coupling_eot_ott,
    get_coupling_leot_ott,
    get_coupling_egw_ott,
)
from perturbot.match.cot import (
    get_coupling_cot_sinkhorn,
    get_coupling_each_cot_sinkhorn,
)
from perturbot.match.gw_labels import get_coupling_egw_labels
from perturbot.predict.scvi_vae import train_vae_model, infer_from_Xs, infer_from_Ys

ot_method_map = {
    "ECOOTL": get_coupling_cotl_sinkhorn,
    "ECOOT": get_coupling_cot_sinkhorn,
    "ECOOT_each": get_coupling_each_cot_sinkhorn,
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
    parser.add_argument("filepath", type=str)
    parser.add_argument("eps", type=float)

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


def main(args):
    # if args.log_filepath is not None and args.log_filepath != "None":
    #     with open(args.log_filepath, "rb") as f:
    #         prev_logs = pkl.load(f)
    logs = {
        "matching_evals": {},
        "pred_evals": {},
        "T": {},
        "pred": {},
        "log": {},
        "eps": {},
    }
    try:
        match_eps = args.eps
        logs["eps"] = match_eps
        with open(args.filepath, "rb") as f:
            data_dict = pkl.load(f)

        X_dict = data_dict["Xs_dict"]
        Y_dict = data_dict["Xt_dict"]
        Zs_dict = data_dict["Zs_dict"]["dosage"]
        Zt_dict = data_dict["Zt_dict"]["dosage"]

        labels = list(X_dict.keys())

        train_X = X_dict
        train_Y = Y_dict
        train_Z = Zs_dict
        train_data = (train_X, train_Y)
        print(f"Calculating matching with {match_eps}")
        if args.log_filepath is not None:
            with open(args.log_filepath, "rb") as f:
                d = pkl.load(f)
            Ts_matching = d["T"]
            log_matching = ""
        else:
            Ts_matching, log_matching = ot_method_map[args.method](
                train_data, match_eps
            )
        if "VAE" in args.method:
            dim_X = data_dict["Xs_dict"][labels[0]].shape[1]
            dim_Y = data_dict["Xt_dict"][labels[0]].shape[1]
            latent_Y = infer_from_Ys(train_Y, Ts_matching, dim_X)
            latent_X = infer_from_Xs(train_X, Ts_matching, dim_Y)
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
        logs["T"] = Ts_matching
        logs["log"] = (log_matching,)
    except Exception as e:
        with open(f"all_{args.method}.{args.eps}.tmp.pkl", "wb") as f:
            pkl.dump({"log": logs, "T": Ts_matching}, f)
        raise e

    with open(f"all_{args.method}.{args.eps}.pkl", "wb") as f:
        pkl.dump(logs, f)


def submit_all_run(data_path, ot_method_label, load_existing=False):
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    for eps in epsilons:
        run_label = f"all.{ot_method_label}.{eps}"
        f = open(f"{run_label}.bsub", "w")
        f.write("#!/bin/bash\n")
        f.write("source ~/.bashrc\n")
        f.write("pwd\n")
        f.write("conda activate ot \n")
        command = f"python /gpfs/scratchfs01/site/u/ryuj6/OT/software/perturbot/perturbot/eval/all.py {ot_method_label} {data_path} {eps}"
        f.write(f"echo {command}\n")
        f.write(f"{command}\n")
        f.close()
        command = f"bsub -M 10G -n 10 -q long -J {run_label} -o log/log.{run_label}.o%J -e log/log.{run_label}.e%J < {run_label}.bsub"
        print(command)
        os.system(command)


if __name__ == "__main__":
    args = parse_args()
    main(args)
