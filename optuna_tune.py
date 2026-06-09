import argparse
import copy
import csv
import gc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

try:
    import optuna
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit(
        "Optuna is not installed. Install it with: pip install optuna"
    ) from exc

from run import build_parser, train_one_trial
from utils import load_mat


SEARCH_SPACE_KEYS = [
    "lr",
    "weight_decay",
    "embedding_dim",
    "batch_size",
    "subgraph_size",
    "readout",
    "negsamp_ratio",
    "alpha",
    "beta",
    "rwr_restart_prob",
]


def _positive_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers, e.g. 64,128,256")
    return values


def _str_list(text: str) -> List[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated values")
    return values


def build_tune_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Optuna hyper-parameter tuning for SL-GAD"

    # `run.py` already has --trials.  Keep it for backward compatibility, but expose
    # a clearer name for tuning-time repeated seeds.
    parser.add_argument("--n_trials", type=int, default=15, help="Number of Optuna configurations to evaluate.")
    parser.add_argument("--repeat_trials", type=int, default=1)
    parser.add_argument("--metric", choices=["auc", "auprc"], default="auprc", help="Metric optimized by Optuna.")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name. Defaults to slgad_<dataset>_<metric>.")
    parser.add_argument("--storage", type=str, default="sqlite:///tune/optuna.db", help="Optional Optuna storage URL, e.g. sqlite:///results/optuna.db")
    parser.add_argument("--resume", action="store_true", help="Resume an existing study with the same name/storage.")
    parser.add_argument("--sampler_seed", type=int, default=2026, help="Random seed for Optuna sampler.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional tuning timeout in seconds.")
    parser.add_argument("--jobs", type=int, default=1, help="Optuna parallel jobs. Use 1 for a single GPU.")
    parser.add_argument("--save_dir", type=str, default="tune")

    # Lightweight customization of the search space without editing this file.
    parser.add_argument("--embedding_dim_choices", type=_positive_int_list, default=[32, 64, 128, 256])
    parser.add_argument("--batch_size_choices", type=_positive_int_list, default=[128, 256, 300, 512, 1024])
    parser.add_argument("--subgraph_size_min", type=int, default=3)
    parser.add_argument("--subgraph_size_max", type=int, default=8)
    parser.add_argument("--readout_choices", type=_str_list, default=["avg", "max", "min", "weighted_sum"])
    return parser


def suggest_params(trial: "optuna.Trial", args: argparse.Namespace) -> Dict[str, object]:
    if args.subgraph_size_min > args.subgraph_size_max:
        raise ValueError("--subgraph_size_min cannot be greater than --subgraph_size_max")

    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "embedding_dim": trial.suggest_categorical("embedding_dim", args.embedding_dim_choices),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_size_choices),
        "subgraph_size": trial.suggest_int("subgraph_size", args.subgraph_size_min, args.subgraph_size_max),
        "readout": trial.suggest_categorical("readout", args.readout_choices),
        "negsamp_ratio": trial.suggest_int("negsamp_ratio", 1, 3),
        "alpha": trial.suggest_float("alpha", 0.1, 2.0),
        "beta": trial.suggest_float("beta", 0.1, 2.0),
        "rwr_restart_prob": trial.suggest_float("rwr_restart_prob", 0.5, 0.95),
    }
    return params


def apply_params(args: argparse.Namespace, params: Dict[str, object]) -> argparse.Namespace:
    tuned_args = copy.deepcopy(args)
    for key, value in params.items():
        setattr(tuned_args, key, value)
    return tuned_args


def append_trial_csv(path: str, row: Dict[str, object]):
    path_obj = Path(os.path.expanduser(path))
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path_obj.exists()
    with path_obj.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def objective_factory(args: argparse.Namespace, data_bundle, device):
    repeat_trials = args.repeat_trials
    if repeat_trials <= 0:
        raise ValueError("repeat_trials / trials must be positive")

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial, args)
        tuned_args = apply_params(args, params)
        auc_values: List[float] = []
        auprc_values: List[float] = []
        epoch_values: List[int] = []

        try:
            for repeat_idx in range(repeat_trials):
                seed = int(args.seed) + trial.number * repeat_trials + repeat_idx
                roc_auc, auprc, best_epoch = train_one_trial(tuned_args, seed, data_bundle, device)
                auc_values.append(float(roc_auc))
                auprc_values.append(float(auprc))
                epoch_values.append(int(best_epoch))

                score_so_far = float(np.mean(auprc_values if args.metric == "auprc" else auc_values))
                trial.report(score_so_far, step=repeat_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except RuntimeError as exc:
            # Treat OOM as an invalid trial instead of killing the whole study.
            if "out of memory" in str(exc).lower():
                cleanup_cuda()
                raise optuna.TrialPruned(f"CUDA OOM for params: {params}") from exc
            raise
        finally:
            cleanup_cuda()

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "study_trial": trial.number,
            "metric": args.metric,
            "auc_mean": float(np.mean(auc_values)),
            "auc_std": float(np.std(auc_values)),
            "auprc_mean": float(np.mean(auprc_values)),
            "auprc_std": float(np.std(auprc_values)),
            "epochs_mean": float(np.mean(epoch_values)),
            "epochs_max": int(np.max(epoch_values)),
            **params,
        }
        append_trial_csv(args.trials_csv, row)

        return row[f"{args.metric}_mean"]

    return objective


def save_best_params(args: argparse.Namespace, study: "optuna.Study"):
    path = Path(os.path.expanduser(args.best_params_json))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "metric": args.metric,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    args = build_tune_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_bundle = load_mat(args.dataset, data_root=args.data_root)
    os.makedirs(args.save_dir, exist_ok=True)

    study_name = args.study_name or f"slgad_{args.dataset}_{args.metric}"
    args.best_params_json = os.path.join(args.save_dir, study_name+".json")
    args.trials_csv = os.path.join(args.save_dir, study_name+".csv")

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, args.jobs), n_warmup_steps=0)
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    study.optimize(
        objective_factory(args, data_bundle, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.jobs,
        gc_after_trial=True,
        show_progress_bar=args.tqdm,
    )

    best_path = save_best_params(args, study)
    print("==============================")
    print(f"Study: {study.study_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Optimized metric: {args.metric}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {json.dumps(study.best_params, ensure_ascii=False)}")
    print(f"Best params JSON: {best_path}")
    print(f"Trials CSV: {os.path.expanduser(args.trials_csv)}")
    print("==============================")


if __name__ == "__main__":
    main()
