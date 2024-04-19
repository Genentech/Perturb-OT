from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from .prediction import get_evals_preds
from multiprocessing import Pool
import pickle as pkl
from .match import get_FOSCTTM, get_diag_fracs
from .utils import get_Ts_from_nn_multKs
from perturbot.predict.scvi_vae import predict_from_model, infer_from_Xs, infer_from_Ys
from perturbot.preprocess.vae import (
    train_vae_rna,
    train_vae_acc,
    train_vae_prot,
    SCVI_LATENT_KEY,
)
from .utils import _pop_key

np.seterr(all="raise")

vae_trainer_dict = {
    "rna": train_vae_rna,
    "accessibility": train_vae_acc,
    "protein": train_vae_prot,
}


def run_models(
    X_dict,
    Y_dict,
    ot_method,
    pred_method,
    pred_method_label: str,
    baseline_pred_methods: List,
    baseline_pred_method_labels: List[str],
    pred_from_param,
    Ts_dict=None,
    Z_dict=None,
    *args,
    **kwargs,
):
    if pred_method_label == "VAE":
        return run_models_vae(
            X_dict,
            Y_dict,
            ot_method,
            Ts_dict=None,
            *args,
            **kwargs,
        )
    labels = list(X_dict.keys())
    loo = LeaveOneOut()
    log = {"ot_couplings": {}, "params": {}, "preds": {}, "logs": {}}
    eval_dfs = []
    train_data = []
    test_data = {}

    for i, (train_idx, test_idx) in enumerate(loo.split(labels)):
        test_idx = test_idx.item()
        test_X = X_dict[test_idx]
        test_Y = Y_dict[test_idx]
        train_X = X_dict.copy()
        train_Y = Y_dict.copy()
        del train_X[test_idx]
        del train_Y[test_idx]
        train_data.append((train_X, train_Y))
        test_data[test_idx] = (test_X, test_Y)
    if Ts_dict is None:
        try:
            with Pool(10) as p:
                Ts_list, logs = zip(*p.map(ot_method, train_data))
        except KeyboardInterrupt:
            p.terminate()
        finally:
            p.join()

    for i, (test_idx, (test_X, test_Y)) in enumerate(test_data.items()):
        if Ts_dict is None:
            Ts = Ts_list[i]
            log["logs"][i] = logs[i]
        else:
            Ts = Ts_dict[test_idx]
        log["ot_couplings"][test_idx] = Ts
        train_X = X_dict.copy()
        train_Y = Y_dict.copy()
        if Z_dict is not None:
            train_Z = Z_dict.copy()
        del train_X[test_idx]
        del train_Y[test_idx]
        params = [pred_method(train_X, train_Y, Ts)]
        if Z_dict is not None:
            params += [
                baseline_method(train_X, train_Y, train_Z)
                for baseline_method in baseline_pred_methods
            ]
        else:
            params += [
                baseline_method(train_X, train_Y)
                for baseline_method in baseline_pred_methods
            ]
        log["params"][test_idx] = params

        preds = [pred_from_param(test_X, param) for param in params]
        log["preds"][test_idx] = preds
        method_labels = [pred_method_label] + baseline_pred_method_labels
        assert len(preds) == len(method_labels)
        eval_df = get_evals_preds(test_Y, preds, method_labels)
        eval_df["loo_test_idx"] = test_idx
        eval_dfs.append(eval_df)

    return pd.concat(eval_dfs, axis=0), log


def run_models_vae(
    X_dict,
    Y_dict,
    ot_method,
    Ts_dict=None,
    *args,
    **kwargs,
):
    labels = list(X_dict.keys())
    loo = LeaveOneOut()
    log = {
        "models": {},
        "params": {},
        "preds": {},
        "latent_X": {},
        "latent_Y": {},
        "logs": {},
    }
    eval_dfs = []
    train_data = []
    test_data = {}

    for i, (train_idx, test_idx) in enumerate(loo.split(labels)):
        test_idx = test_idx.item()
        test_X = X_dict[test_idx]
        test_Y = Y_dict[test_idx]
        train_X = X_dict.copy()
        train_Y = Y_dict.copy()
        del train_X[test_idx]
        del train_Y[test_idx]
        train_data.append((train_X, train_Y))
        test_data[test_idx] = (test_X, test_Y)
    if Ts_dict is None:
        try:
            with Pool(10) as p:
                model_list, logs = zip(
                    *p.map(
                        ot_method,
                        train_data,
                    )
                )
        except KeyboardInterrupt:
            p.terminate()
        finally:
            p.join()
    ks = [5, 10, 50, 100]
    for k in ks:
        log[f"pred_T_k{k}"] = {}  # test_idx -> train_idx -> T
    for i, (test_idx, (test_X, test_Y)) in enumerate(test_data.items()):
        if Ts_dict is None:
            model = model_list[i]
            log["models"][test_idx] = model
            log["logs"][test_idx] = logs[i]
        else:
            model = Ts_dict[test_idx]
            log["models"][test_idx] = model
        dX = test_X.shape[1]
        dY = test_Y.shape[1]
        latent_Y = infer_from_Ys(_pop_key(Y_dict, test_idx), model, dX)
        latent_X = infer_from_Xs(_pop_key(X_dict, test_idx), model, dY)
        pred_Y = predict_from_model(test_X, model, test_Y.shape[1])
        log["preds"][test_idx] = pred_Y
        log["latent_X"][test_idx] = latent_X
        log["latent_Y"][test_idx] = latent_Y
        Ts = get_Ts_from_nn_multKs(latent_X, latent_Y, ks)  # k -> T
        for k, v in Ts.items():
            log[f"pred_T_k{k}"][test_idx] = v
        eval_df = get_evals_preds(test_Y, [pred_Y], ["VAE"])
        eval_df["loo_test_idx"] = test_idx
        eval_dfs.append(eval_df)

    return pd.concat(eval_dfs, axis=0), log


def run_models_vae_then_ot(
    source_adata,
    target_adata,
    source_modality,
    target_modality,
    label_col,
    ot_method,
    pred_method,
    pred_method_label: str,
    baseline_pred_methods: List,
    baseline_pred_method_labels: List[str],
    pred_from_param,
    T_dict=None,
    model_dict=None,
    Z_dict=None,
    n_threads=1,
    *args,
    **kwargs,
):
    labels = source_adata.obs[label_col].unique().values
    loo = LeaveOneOut()
    log = {"ot_couplings": {}, "params": {}, "preds": {}, "logs": {}}
    eval_dfs = []
    test_data = {}
    source_adatas_train = []
    target_adatas_train = []
    for i, (train_idx, test_idx) in enumerate(loo.split(labels)):
        test_idx = test_idx.item()
        source_adata_test = source_adata.obs[label_col == test_idx].copy()
        target_adata_test = target_adata.obs[label_col == test_idx].copy()
        source_adata_train = source_adata.obs[label_col != test_idx].copy()
        target_adata_train = target_adata.obs[label_col != test_idx].copy()
        source_adatas_train.append(source_adata_train)
        target_adatas_train.append(target_adata_train)
        test_data[test_idx] = (source_adata_test, target_adata_test)

    # Train VAE
    if model_dict is None:
        try:
            with Pool(n_threads) as p:
                source_adatas, source_models = zip(
                    *p.map(vae_trainer_dict[source_modality], source_adatas_train)
                )
        except KeyboardInterrupt:
            p.terminate()
        finally:
            p.join()
        try:
            with Pool(n_threads) as p:
                target_adatas, target_models = zip(
                    *p.map(vae_trainer_dict[target_modality], target_adatas_train)
                )
        except KeyboardInterrupt:
            p.terminate()
        finally:
            p.join()

    # Train OT
    train_data = []
    for i, (test_idx, (test_X, test_Y)) in enumerate(test_data.items()):
        train_X = source_adatas[i].obsm[SCVI_LATENT_KEY]
        train_Y = target_adatas[i].obsm[SCVI_LATENT_KEY]
        train_data.append((train_X, train_Y))

    if T_dict is None:
        try:
            with Pool(1) as p:
                T_list, logs = zip(*p.map(ot_method, train_data))
        except KeyboardInterrupt:
            p.terminate()
        finally:
            p.join()

    for i, (test_idx, (test_X, test_Y)) in enumerate(test_data.items()):
        if T_dict is None:
            Ts = T_list[i]
            log["logs"][i] = logs[i]
        else:
            Ts = T_dict[test_idx]
        log["ot_couplings"][test_idx] = Ts

        # @TODO see from here
        pred_Y = predict_from_model_with_OT(
            test_X, source_models[i], target_models[i], Ts, test_Y.shape[1]
        )

        preds = [pred_from_param(test_X, param) for param in params]
        log["preds"][test_idx] = preds
        method_labels = [pred_method_label] + baseline_pred_method_labels
        assert len(preds) == len(method_labels)
        eval_df = get_evals_preds(test_Y, preds, method_labels)
        eval_df["loo_test_idx"] = test_idx
        eval_dfs.append(eval_df)

    return pd.concat(eval_dfs, axis=0), log


def evaluate_loo_vae(
    data_dict, logfile_name, test_idx_file, use_z_key="dosage", ks=[5, 10, 50, 100]
):
    Xs_dict = data_dict["Xs_dict"]
    Xt_dict = data_dict["Xt_dict"]
    Zs_dict = data_dict["Zs_dict"]
    Zt_dict = data_dict["Zt_dict"]
    with open(logfile_name, "rb") as f:
        log = pkl.load(f)
    dfracs = {k: [] for k in ks}
    rel_dfracs = {k: [] for k in ks}
    med_foscttms = []
    idx_name = pd.read_csv(test_idx_file, header=None)
    for test_idx in log["preds"].keys():
        model = log["models"]
        Xs_train = _pop_key(Xs_dict, test_idx)
        Xt_train = _pop_key(Xt_dict, test_idx)
        for k in ks:
            dfrac, rel_dfrac = get_diag_fracs(
                log[f"pred_T_k{k}"][test_idx],
                Xs_train,
                Xt_train,
                Zs_dict[use_z_key],
                Zt_dict[use_z_key],
            )
            dfracs[k].append(dfrac)
            rel_dfracs[k].append(rel_dfrac)
        _, median_foscttm = get_FOSCTTM(
            model,
            log["latent_X"][test_idx],
            log["latent_Y"][test_idx],
            use_barycenter=False,
        )
        med_foscttms.append(median_foscttm)
    med_foscttms = pd.concat(med_foscttms, axis=1).mean(axis=1)
    rel_dfracs_ks = {}
    dfracs_ks = {}
    for k in ks:
        rel_dfracs_ks[k] = pd.concat(rel_dfracs[k], axis=1).mean(axis=1)
        dfracs_ks[k] = pd.concat(dfracs[k], axis=1).mean(axis=1)
    main_k = 10
    other_ks = [k for k in ks if k != 3]
    metrics_list = [
        med_foscttms,
        dfracs_ks[main_k],
        rel_dfracs_ks[main_k],
    ]
    labels = ["treat_idx", "FOSCTTM", "DFracs", "Relative DFracs"]
    for k in other_ks:
        metrics_list = metrics_list + [dfracs_ks[k], rel_dfracs_ks[k]]
        labels = labels + [f"DFracs_{k}", f"Relative DFracs_{k}"]
    metrics = pd.concat(
        metrics_list,
        axis=1,
    ).reset_index()

    metrics.columns = labels
    metrics["treatment"] = idx_name.loc[metrics["treat_idx"].astype(int), 0].tolist()
    return metrics.set_index("treatment", drop=True)[labels[1:]].to_dict("series")


def evaluate_loo_run(
    data_dict,
    logfile_name,
    test_idx_file="/home/ryuj6/scratch/OT/data/chemical_screen/chemical_screen_pca_idx.txt",
    use_z_key="dosage",
):
    Xs_dict = data_dict["Xs_dict"]
    Xt_dict = data_dict["Xt_dict"]
    Zs_dict = data_dict["Zs_dict"]
    Zt_dict = data_dict["Zt_dict"]
    with open(logfile_name, "rb") as f:
        log = pkl.load(f)
    dfracs = []
    rel_dfracs = []
    med_foscttms = []
    idx_name = pd.read_csv(test_idx_file, header=None)
    for test_idx, Ts in log["ot_couplings"].items():
        Xs_train = _pop_key(Xs_dict, test_idx)
        Xt_train = _pop_key(Xt_dict, test_idx)

        dfrac, rel_dfrac = get_diag_fracs(
            Ts, Xs_train, Xt_train, Zs_dict[use_z_key], Zt_dict[use_z_key]
        )
        _, median_foscttm = get_FOSCTTM(Ts, Xs_train, Xt_train)
        med_foscttms.append(median_foscttm)
        dfracs.append(dfrac)
        rel_dfracs.append(rel_dfrac)
    if isinstance(Ts, dict):
        med_foscttms = pd.concat(med_foscttms, axis=1).mean(axis=1)
        rel_dfracs = pd.concat(rel_dfracs, axis=1).mean(axis=1)
        dfracs_df = pd.concat(dfracs, axis=1).mean(axis=1)
        metrics = pd.concat(
            [
                med_foscttms,
                dfracs_df,
                rel_dfracs,
            ],
            axis=1,
        ).reset_index()
    else:
        med_foscttms = pd.Series(med_foscttms, index=log["ot_couplings"].keys())
        rel_dfracs = pd.Series(rel_dfracs, index=log["ot_couplings"].keys())
        dfracs_df = pd.Series(dfracs_df, index=log["ot_couplings"].keys())
        metrics = pd.concat(
            [
                med_foscttms,
                dfracs_df,
                rel_dfracs,
            ],
            axis=1,
        ).reset_index()

    metrics.columns = ["treat_idx", "FOSCTTM", "DFracs", "Relative DFracs"]
    metrics["treatment"] = idx_name.loc[metrics["treat_idx"].astype(int), 0].tolist()
    return metrics.set_index("treatment", drop=True)[
        ["FOSCTTM", "DFracs", "Relative DFracs"]
    ].to_dict("series")


def evaluate_loo_runs_OT(
    data_dict,
    logfile_paths: List[str],
    labels: List[str],
    test_idx_file="/home/ryuj6/scratch/OT/data/chemical_screen/chemical_screen_pca_idx.txt",
):
    metric_names = ["FOSCTTM", "DFracs", "Relative DFracs"]
    metrics = {m: [] for m in metric_names}
    non_foscttm_labels = []
    for logfile_path, label in zip(logfile_paths, labels):
        if "vae" in label:
            evals = evaluate_loo_vae(data_dict, logfile_path, test_idx_file)
        else:
            non_foscttm_labels.append(label)
            evals = evaluate_loo_run(data_dict, logfile_path, test_idx_file)
        for m in metric_names:
            if m in evals:
                metrics[m].append(evals[m])
            else:
                metrics[m].append(pd.Series(np.zeros(evals["FOSCTTM"].shape)))

    def collect_series(list_series):
        df = pd.concat(list_series, axis=1)
        df.columns = labels
        return df

    metrics = {m: collect_series(metrics[m]) for m in metrics.keys()}
    return metrics
