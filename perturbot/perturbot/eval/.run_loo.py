# Run leave-one-out prediction & evaluation
from typing import Optional
import numpy as np
import argparse
import mudata as md
from sklearn.model_selection import LeaveOneOut
from software.perturbot.perturbot.ot.cot_labels import (
    predict_with_cot,
    evaluate_training,
)

method_dict = {"cot_labels": predict_with_cot}
get_train_metrics_dict = {"cot_labels": evaluate_training}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leave-one-out prediction & evaluation."
    )
    parser.add_argument("input_h5mu", type=str, help="Path to the input .h5mu file.")
    parser.add_argument(
        "--label-col",
        "-l",
        type=str,
        default="treatment",
        help="Column label in .obs of MuData that encodes the perturbation labels.",
    )
    parser.add_argument(
        "--within-label-group",
        "-g",
        type=str,
        default=None,
        help="If provided, evaluation includes the concordance of the grouping.",
    )
    parser.add_argument(
        "--within-label-group",
        "-g",
        type=str,
        default=None,
        help="If provided, evaluation includes the concordance of the grouping.",
    )
    parser.add_argument(
        "--train-with-group",
        "-tg",
        type=bool,
        action="store_true",
        help="train with `within_label_group`.",
    )
    parser.add_argument(
        "--source-modality",
        "-s",
        type=str,
        default="protein",
        help="Source modality in MuData",
    )
    parser.add_argument(
        "--target-modality",
        "-t",
        type=str,
        default="RNA",
        help="Target modality in MuData",
    )
    parser.add_argument(
        "--rng-seed",
        "-r",
        type=int,
        default=2024,
        help="random number generator seed for NumPy",
    )
    return parser.parse_args()


def get_data_with_group(
    x, y, include_y, z: Optional[np.ndarray] = None, nbperclass: Optional[int] = None
):
    # Randomly select nbperclass train datasets
    xr = np.zeros((0, x.shape[1]))
    yr = np.zeros((0))
    if z is not None:
        zr = np.zeros((0))
    else:
        zr = None
    for i in include_y:
        xi = x[y.ravel() == i, :]
        xr = np.concatenate((xr, xi), 0)
        yr = np.concatenate((yr, np.repeat(i, xi.shape[0])))
        if z is not None:
            zi = z[y.ravel() == i]
            zr = np.concatenate((zr, zi), 0)
    return xr, yr, zr


def main():
    args = parse_args()
    np.random.seed(args.rng_seed)
    mdata = md.read_h5ad(args.input_h5mu)
    Xtot1 = mdata[args.source_modality].X
    Xtot2 = mdata[args.target_modality].X
    labels = mdata.obs[args.label_col].values
    within_label_group = mdata.obs[args.within_label_group].values
    loo = LeaveOneOut()
    train_data = {}
    test_data = {}
    for i, (train_idx, test_idx) in enumerate(loo.split(labels)):
        Xs, Ys, Zs = get_data_with_group(
            Xtot1, labels, labels[train_idx], within_label_group
        )
        Xt, Yt, Zt = get_data_with_group(
            Xtot2, labels, labels[train_idx], within_label_group
        )
        Xs_test, Ys_test, Zs_test = get_data_with_group(
            Xtot1, labels, labels[test_idx], within_label_group
        )
        Xt_test, Yt_test, Zt_test = get_data_with_group(
            Xtot1, labels, labels[test_idx], within_label_group
        )
        for method_label, run_fn in method_dict.items():
            Xt_pred, log = run_fn(Xs, Ys, Xt, Yt, Xs_test)
            train_eval_metrics = get_train_metrics_dict[method_label](
                Xs, Ys, Xt, Yt, Zt, log
            )
            test_eval_metrics = get_test_metrics(Xt_pred, Xt_test, Zt_test)


if __name__ == "__main__":
    main()
