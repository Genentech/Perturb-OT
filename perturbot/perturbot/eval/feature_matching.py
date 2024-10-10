"""For 5-fold CV, run the single inner loop of train-validation to select hyperparameter."""

import os
import argparse
import pickle as pkl
from functools import partial
import numpy as np
from perturbot.match.fot import get_coupling_fot
from perturbot.eval.utils import make_G
from perturbot.predict.scvi_vae import train_vae_model, infer_from_Xs, infer_from_Ys
from perturbot.eval.utils import get_Ts_from_nn_multKs


def parse_args():
    parser = argparse.ArgumentParser(
        "Run inner validation loop of CV",
        "Run sample-matching OT in parallel and fit prediction model in leave-one-out mannter",
    )
    parser.add_argument("method", type=str)
    parser.add_argument("filepath", type=str)
    parser.add_argument("best_eps", type=float)
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
    parser.add_argument(
        "--best-k",
        type=int,
        default=None,
        help="Best k to use for VAE",
    )
    return parser.parse_args()


ot_method_all_to_all = ["GW_all", "EGW_all_ott", "EOT_all_ott"]


def main(args):
    logs = {
        "matching_evals": {},
        "pred_evals": {},
        "T": {},
        "pred": {},
        "log": {},
        "eps": {},
    }
    try:
        if args.best_eps != 0:
            with open(f"all_{args.method}.{args.best_eps}.pkl", "rb") as f:
                d = pkl.load(f)
            Ts = d["T"]
            logs["sample_eps"] = args.best_eps
            # logs['min_foscttm'] = d[]
        else:
            Ts = None
        with open(args.filepath, "rb") as f:
            data_dict = pkl.load(f)  # full features

        X_dict = data_dict["Xs_dict"]
        Y_dict = data_dict["Xt_dict"]
        Z_dict = data_dict["Zs_dict"]["dosage"]

        if "VAE" in args.method:
            labels = list(X_dict.keys())
            dim_X = data_dict["Xs_dict"][labels[0]].shape[1]
            dim_Y = data_dict["Xt_dict"][labels[0]].shape[1]
            latent_Y = infer_from_Ys(Y_dict, Ts, dim_X)
            latent_X = infer_from_Xs(X_dict, Ts, dim_Y)
            Ts = get_Ts_from_nn_multKs(latent_X, latent_Y, [args.best_k])[args.best_k]

        if Ts is None:
            if args.method == "random":
                Ts = {
                    k: np.ones((X_dict[k].shape[0], Y_dict[k].shape[0]))
                    / (X_dict[k].shape[0] * Y_dict[k].shape[0])
                    for k in X_dict.keys()
                }
            elif args.method == "perfect":
                Ts = {
                    k: np.diag(
                        np.ones(X_dict[k].shape[0]),
                    )
                    / X_dict[k].shape[0]
                    for k in X_dict.keys()
                }
            elif args.method == "by_conc":
                Ts = {
                    k: make_G(X_dict[k].shape[0], Z_dict[k], k) for k in X_dict.keys()
                }
        Tv, log = get_coupling_fot((X_dict, Y_dict), Ts, args.eps)
        logs["log"] = log
        logs["Tv"] = Tv
    except Exception as e:
        with open(f"features_{args.method}.{args.eps}.tmp.pkl", "wb") as f:
            pkl.dump(logs, f)
        raise e

    with open(f"features_{args.method}.{args.eps}.pkl", "wb") as f:
        pkl.dump(logs, f)


def submit_feature_run(data_path, ot_method_label):
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    rel_dfracs = []
    if ot_method_label in ["perfect", "random", "by_conc"]:
        best_eps = 0
    else:
        best_k_dict = {}
        for eps in epsilons:
            # try:
            with open(f"all_{ot_method_label}.{eps}.pkl", "rb") as f:
                d = pkl.load(f)
            _rel_dfracs = d["matching_evals"][0]["rel_dfracs"]
            if isinstance(_rel_dfracs, dict):
                max_rel_dfracs = -10
                for k, v in _rel_dfracs.items():
                    if max_rel_dfracs < v:
                        max_rel_dfracs = v
                        best_k_dict[eps] = k
                _rel_dfracs = max_rel_dfracs
            rel_dfracs.append(_rel_dfracs)
            # except Exception as e:
            #     print(e)
            #     rel_dfracs.append(-1)
        best_eps = epsilons[rel_dfracs.index(max(rel_dfracs))]
        if "VAE" in ot_method_label:
            best_k = best_k_dict[best_eps]
    for eps in epsilons:
        run_label = f"FM.{ot_method_label}.{eps}"
        f = open(f"{run_label}.bsub", "w")
        f.write("source ~/.bashrc\n")
        f.write("pwd\n")
        f.write("conda activate ot \n")
        command = f"python /gpfs/scratchfs01/site/u/ryuj6/OT/software/perturbot/perturbot/eval/feature_matching.py {ot_method_label} {data_path} {best_eps} {eps}"

        if "VAE" in ot_method_label:
            command += f" --best-k {best_k}"
        f.write(f"echo {command}\n")
        f.write(f"{command}\n")
        f.close()
        command = f"bsub -M 10G -n 1 -q short -J {run_label} -o log/log.{run_label}.o%J -e log/log.{run_label}.e%J < {run_label}.bsub"
        print(command)
        os.system(command)


if __name__ == "__main__":
    args = parse_args()
    main(args)
