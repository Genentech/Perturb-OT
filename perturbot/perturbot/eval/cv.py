import os
from typing import List, Dict, Any
from functools import partial
from itertools import product
import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from perturbot.eval.prediction import get_evals_preds, get_evals
from multiprocessing import Pool
import pickle as pkl
from perturbot.eval.match import get_FOSCTTM, get_diag_fracs
from perturbot.eval.utils import get_Ts_from_nn_multKs, _pop_key
from perturbot.predict.scvi_vae import predict_from_model, infer_from_Xs, infer_from_Ys
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

np.seterr(all="raise")

vae_trainer_dict = {
    "rna": train_vae_rna,
    "accessibility": train_vae_acc,
    "protein": train_vae_prot,
}

ot_method_map = {
    "COOTL": get_coupling_cotl,
    "ECOOTL": get_coupling_cotl_sinkhorn,
    "COOT": get_coupling_cot,
    "ECOOT": get_coupling_cot_sinkhorn,
    "ECOOT_each": get_coupling_each_cot_sinkhorn,
    "GW_all": get_coupling_gw_all,
    # "EGW_all": get_coupling_egw_all,
    "GW": get_coupling_gw_cg,
    # "EGW": get_coupling_egw,
    "GWL": get_coupling_gw_labels,
    "EGWL": get_coupling_egw_labels,
    "EOT_ott": get_coupling_eot_ott,
    "LEOT_ott": get_coupling_leot_ott,
    "EGW_ott": get_coupling_egw_ott,
    "EGW_all_ott": get_coupling_egw_all_ott,
    "EGWL_ott": get_coupling_egw_labels_ott,
    "VAE_label": train_vae_model,
    "VAE": partial(train_vae_model, use_label=False),
}
ot_method_hyperparams = {}
for method in ["EGWL", "EOT_ott", "LEOT_ott", "EGW_ott", "EGW_all_ott", "EGWL_OTT"]:
    ot_method_hyperparams[method] = [
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
    ]
for method in ["ECOOTL", "ECOOT", "ECOOT_each"]:
    ot_method_hyperparams[method] = [0.1, 0.05, 0.01, 0.005, 0.001]
ot_method_all_to_all = ["GW_all", "EGW_all_ott", "EOT_all_ott"]


def run_cv_models(
    filepath,
    ot_method_label,
    data_label="k",
    log_filepath=None,
    full_filepath=None,
    run_mlp=False,
    mlp_baseline=False,
    rerun=False,
):
    if ot_method_label == "VAE":
        assert full_filepath is not None
    with open(filepath, "rb") as f:
        data_dict = pkl.load(f)
    labels = list(data_dict["Xs_dict"].keys())
    cv = KFold(n_splits=5)
    # Outer loop will evaluate final performance.
    for i, (train_val_idx, test_idx) in enumerate(cv.split(labels)):
        # Inner loop will select the best-performing hyperparameter.
        # Run CV for multiple hyperparameters (eps)
        run_label = f"{'mlp.' if run_mlp else ''}{ot_method_label}.{i}"

        if (
            run_mlp
            and os.path.isfile(f"val_mlp.{ot_method_label}.{i}.best_eps.pkl")
            and not rerun
        ):
            print(f"val_mlp.{ot_method_label}.{i}.best_eps.pkl exists. Skipping")
            continue
        elif (
            (not run_mlp)
            and os.path.isfile(f"val_CV_{ot_method_label}.{i}.best_eps.pkl")
            and log_filepath is None
        ):
            print(f"val_CV_{ot_method_label}.{i}.best_eps.pkl exists. Skipping")
            continue
        # Write job script
        f = open(f"{run_label}.bsub", "w")
        f.write("#!/bin/bash\n")
        f.write("source ~/.bashrc\n")
        f.write("pwd\n")
        f.write("conda activate ot \n")
        if "VAE" in ot_method_label:
            command = f"python /gpfs/scratchfs01/site/u/ryuj6/OT/software/perturbot/perturbot/eval/cv_inner_loop.py {ot_method_label} {i} {full_filepath} {'' if log_filepath is None else f'-l val_CV_{ot_method_label}.{i}.pkl'} {'--mlp' if run_mlp else ''} "
        else:
            command = f"python /gpfs/scratchfs01/site/u/ryuj6/OT/software/perturbot/perturbot/eval/cv_inner_loop.py {ot_method_label} {i} {full_filepath if run_mlp else filepath} {'' if log_filepath is None else f'-l val_CV_{ot_method_label}.{i}.pkl'} {'' if full_filepath is None else '-p ' + full_filepath} {'--mlp' if run_mlp else ''} {'--baseline' if mlp_baseline else ''}"

        f.write(f"echo {command}\n")
        f.write(f"{command}\n")
        f.close()
        label = "mlp" if run_mlp else "cv"
        if log_filepath is not None:
            label = "reval"
        command = f"bsub {'-M 20G -n 10' if 'VAE' in ot_method_label or run_mlp else '-M 10G -n 25'} -q {'short' if log_filepath is not None else 'long'} -J {data_label}{'mlp' if run_mlp else 'CV'}.{run_label} -o log/log_{label}.{run_label}.o%J -e log/log_{label}.{run_label}.e%J < {run_label}.bsub"
        print(command)
        os.system(command)


def get_test_eval(
    filepath,
    ot_method_label,
    data_label="k",
    log_filepath=None,
    full_filepath=None,
    run_mlp=False,
    baseline=None,
):
    if ot_method_label == "VAE":
        assert full_filepath is not None
    with open(filepath, "rb") as f:
        data_dict = pkl.load(f)
    labels = list(data_dict["Xs_dict"].keys())
    cv = KFold(n_splits=5)
    # Outer loop will evaluate final performance.
    if baseline is not None:
        for i, (train_val_idx, test_idx) in enumerate(cv.split(labels)):
            if log_filepath is not None:
                log_filepath = f"test_{baseline}.{i}.pkl"
            run_label = f"TM.{baseline}.{i}"
            f = open(f"{run_label}.bsub", "w")
            f.write("#!/bin/bash\n")
            f.write("source ~/.bashrc\n")
            f.write("pwd\n")
            f.write("conda activate ot \n")
            command = f"python /gpfs/scratchfs01/site/u/ryuj6/OT/software/perturbot/perturbot/eval/cv_outer_loop.py {baseline} {i} {filepath} 0,0,0 -p {full_filepath} -l {log_filepath} --baseline {baseline}"
            f.write(f"echo {command}\n")
            f.write(f"{command}\n")
            f.close()
            command = f"bsub -M 10G -n 10 -q long -J {data_label}{run_label} -o test_log/log.{run_label}.o%J -e test_log/log.{run_label}.e%J < {run_label}.bsub"
            print(command)
            os.system(command)
    else:
        for i, (train_val_idx, test_idx) in enumerate(cv.split(labels)):
            # Evaluate matching
            try:
                with open(f"val_CV_{ot_method_label}.{i}.best_eps.pkl", "rb") as f:
                    log = pkl.load(f)

            except FileNotFoundError:
                print(f"val_CV_{ot_method_label}.{i}.best_eps.pkl not found. Skipping")
                continue
            if "VAE" not in ot_method_label:
                try:
                    with open(f"val_mlp.{ot_method_label}.{i}.best_eps.pkl", "rb") as f:
                        log2 = pkl.load(f)
                    eps_pred = log2["eps"]
                except FileNotFoundError:
                    try:
                        with open(f"val_CV_MLP_{ot_method_label}.{i}.pkl", "rb") as f:
                            log2 = pkl.load(f)
                        eps_pred = log2["pred_evals"].T["Pearson_samples"].idxmax()
                    except FileNotFoundError:
                        print(
                            f"val_CV_MLP_{ot_method_label}.{i}.pkl and val_mlp.{ot_method_label}.{i}.best_eps.pkl not found. Skipping"
                        )
                        continue
            # elif os.path.exists(f"test_{ot_method_label}.{i}.pkl"):
            #     print(f"test_{ot_method_label}.{i}.pkl already exists. Skipping")
            # elif os.path.exists(f"test_log/test_{ot_method_label}.{i}.tmp.pkl"):
            #     print(
            #         f"test_log/test_{ot_method_label}.{i}.tmp.pkl already exists. Skipping"
            # )
            else:
                eps_pred = np.nan

            allowed_eps = [1e-2, 1e-3, 1e-4, 1e-5]
            eps_lin = log["pred"]

            eps_matching = log["matching"]
            if isinstance(eps_lin, tuple):
                eps_lin = eps_lin[0]
            if isinstance(eps_matching, tuple):
                eps_matching = eps_matching[0]

            if "VAE" not in ot_method_label and (
                eps_lin not in allowed_eps or eps_matching not in allowed_eps
            ):
                try:
                    with open(f"val_CV_{ot_method_label}.{i}.pkl", "rb") as f:
                        eval_log = pkl.load(f)
                except FileNotFoundError:
                    print(f"val_CV_{ot_method_label}.{i}.pkl not found. Skipping")
                eps_lin = eval_log["pred_evals"].loc["MSE", allowed_eps].idxmin()
                eps_matching = eval_log["matching_evals"][allowed_eps].idxmin()
            elif "VAE" in ot_method_label:
                if isinstance(eps_lin, tuple):
                    eps_lin = eps_lin[0]
                if isinstance(eps_matching, tuple):
                    eps_matching = eps_matching[0]
            if log_filepath is not None:
                log_filepath = f"test_{ot_method_label}.{i}.pkl"
            run_label = f"TM.{ot_method_label}.{i}"
            f = open(f"{run_label}.bsub", "w")
            f.write("#!/bin/bash\n")
            f.write("source ~/.bashrc\n")
            f.write("pwd\n")
            f.write("conda activate ot \n")
            command = f"python /gpfs/scratchfs01/site/u/ryuj6/OT/software/perturbot/perturbot/eval/cv_outer_loop.py {ot_method_label} {i} {filepath} {eps_matching},{eps_lin},{eps_pred} -p {full_filepath} -l {log_filepath} {'' if baseline is None else '--baseline '+baseline}"
            f.write(f"echo {command}\n")
            f.write(f"{command}\n")
            f.close()
            command = f"bsub -M 10G -n 10 -q long -J {data_label}{run_label} -o test_log/log.{run_label}.o%J -e test_log/log.{run_label}.e%J < {run_label}.bsub"
            print(command)
            os.system(command)
