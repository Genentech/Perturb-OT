import argparse
from perturbot.eval.cv import run_cv_models, get_test_eval
from perturbot.eval.all import submit_all_run
from perturbot.eval.feature_matching import submit_feature_run


def parse_args():
    parser = argparse.ArgumentParser(
        "Run inner validation loop of CV",
        "Run sample-matching OT in parallel and fit prediction model in leave-one-out mannter",
    )
    parser.add_argument("data", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--baseline_test", default=None)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--feature", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    """Run as `python cv.py kinase EGWL"""
    data_paths = {
        "kinase": "/gpfs/scratchfs01/site/u/ryuj6/OT/data/chemical_screen/chemical_screen_pca_subsampled.pkl",
        "chromod": "/gpfs/scratchfs01/site/u/ryuj6/OT/data/chemical_screen/chromatin_modifier/chromod_pca_subsampled.pkl",
    }
    full_data_paths = {
        "kinase": "/gpfs/scratchfs01/site/u/ryuj6/OT/data/chemical_screen/chemical_screen_subsampled_2000.pkl",
        "chromod": "/gpfs/scratchfs01/site/u/ryuj6/OT/data/chemical_screen/chromatin_modifier/chromod_subsampled_2000.pkl",
    }
    args = parse_args()
    if args.test:
        get_test_eval(
            data_paths[args.data],
            args.method,
            data_label=args.data[0],
            full_filepath=full_data_paths[args.data],
            log_filepath=args.log,
            baseline=args.baseline_test,
        )
    elif args.all:
        submit_all_run(
            data_paths[args.data],
            args.method,
        )
    elif args.feature:
        submit_feature_run(
            full_data_paths[args.data],
            args.method,
        )
    else:
        run_cv_models(
            data_paths[args.data],
            args.method,
            data_label=args.data[0],
            full_filepath=full_data_paths[args.data],
            run_mlp=args.mlp,
            log_filepath=args.log,
            mlp_baseline=args.baseline,
            rerun=args.rerun,
        )
