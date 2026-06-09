import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from model import Model
from utils import (
    adj_to_pyg_data,
    build_neighbor_lists,
    generate_rwr_subgraph_from_neighbors,
    load_mat,
    normalize_adj,
    pr_auc_score,
    preprocess_features,
    set_seed,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def build_parser():
    parser = argparse.ArgumentParser(description="SL-GAD PyG resource-optimized refactor")
    parser.add_argument("--expid", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="BlogCatalog")
    parser.add_argument("--data_root", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--result_csv", type=str, default="results/slgad_results.csv")
    parser.add_argument("--cache_dir", type=str, default="tmp", help="Kept for CLI compatibility; no checkpoint is written by default.")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=5, help="Number of independent trials.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed. Trial i uses seed+i.")
    parser.add_argument("--tqdm", action="store_true")

    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--num_epoch", type=int, default=400)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--subgraph_size", type=int, default=4)
    parser.add_argument("--readout", type=str, default="avg")
    parser.add_argument("--auc_test_rounds", type=int, default=256)
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--rwr_restart_prob", type=float, default=0.9)
    return parser


def format_metric(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64) * 100.0
    return f"{arr.mean():.2f}±{arr.std(ddof=0):.2f}({arr.max():.2f})"


def append_result_csv(path: str, row: Dict[str, str]):
    path_obj = Path(os.path.expanduser(path))
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path_obj.exists()
    with path_obj.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _local_adj_batch(adj_csr: sp.csr_matrix, subgraphs, device) -> torch.Tensor:
    """Build only small B×S×S local adjacency tensors for the current batch."""
    mats = [adj_csr[sg, :][:, sg].toarray() for sg in subgraphs]
    return torch.as_tensor(np.stack(mats, axis=0), dtype=torch.float32, device=device)


def _insert_anchor_gap(x: torch.Tensor) -> torch.Tensor:
    zero = x.new_zeros((x.shape[0], 1, x.shape[2]))
    return torch.cat((x[:, :-1, :], zero, x[:, -1:, :]), dim=1)


def _append_adj_anchor(adj_batch: torch.Tensor) -> torch.Tensor:
    batch_size, subgraph_size, _ = adj_batch.shape
    zero_row = adj_batch.new_zeros((batch_size, 1, subgraph_size))
    adj_batch = torch.cat((adj_batch, zero_row), dim=1)
    zero_col = adj_batch.new_zeros((batch_size, subgraph_size + 1, 1))
    zero_col[:, -1, :] = 1.0
    return torch.cat((adj_batch, zero_col), dim=2)


def make_batch_tensors(idx, subgraphs_1, subgraphs_2, adj_csr, features_t, raw_features_t, device):
    sg1 = [subgraphs_1[i] for i in idx]
    sg2 = [subgraphs_2[i] for i in idx]
    sg1_t = torch.as_tensor(sg1, dtype=torch.long, device=device)
    sg2_t = torch.as_tensor(sg2, dtype=torch.long, device=device)

    ba1 = _append_adj_anchor(_local_adj_batch(adj_csr, sg1, device))
    ba2 = _append_adj_anchor(_local_adj_batch(adj_csr, sg2, device))
    bf1 = _insert_anchor_gap(features_t[sg1_t])
    bf2 = _insert_anchor_gap(features_t[sg2_t])
    raw_bf1 = _insert_anchor_gap(raw_features_t[sg1_t])
    raw_bf2 = _insert_anchor_gap(raw_features_t[sg2_t])
    return ba1, ba2, bf1, bf2, raw_bf1, raw_bf2


def _best_state_dict_on_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_one_trial(args, seed, data_bundle, device) -> Tuple[float, float, int]:
    adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = data_bundle
    set_seed(seed)

    raw_features = np.asarray(features.todense(), dtype=np.float32)
    features, _ = preprocess_features(features)
    pyg_graph = adj_to_pyg_data(adj)
    neighbors = build_neighbor_lists(pyg_graph.edge_index, int(pyg_graph.num_nodes))

    nb_nodes, ft_size = features.shape
    batch_size = args.batch_size
    batch_num = (nb_nodes + batch_size - 1) // batch_size

    # Keep normalized adjacency on CPU in sparse CSR format.  The previous code created
    # a dense N×N tensor on GPU, which is the largest avoidable memory cost.
    adj_norm = (normalize_adj(adj) + sp.eye(adj.shape[0], dtype=np.float32)).tocsr().astype(np.float32)

    features_t = torch.as_tensor(features, dtype=torch.float32, device=device)
    raw_features_t = torch.as_tensor(raw_features, dtype=torch.float32, device=device)

    model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([args.negsamp_ratio], device=device)
    )
    mse_loss = nn.MSELoss(reduction="mean")

    cnt_wait = 0
    best = float("inf")
    best_t = 0
    best_state = _best_state_dict_on_cpu(model)

    epoch_loop = range(args.num_epoch)
    if args.tqdm:
        epoch_loop = tqdm(epoch_loop, desc="Epoch", position=1, leave=False)

    for epoch in epoch_loop:
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.0
        seen_nodes = 0

        subgraphs_1 = generate_rwr_subgraph_from_neighbors(
            neighbors, args.subgraph_size, restart_prob=args.rwr_restart_prob
        )
        subgraphs_2 = generate_rwr_subgraph_from_neighbors(
            neighbors, args.subgraph_size, restart_prob=args.rwr_restart_prob
        )

        for batch_idx in range(batch_num):
            idx = all_idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            if not idx:
                continue
            cur_batch_size = len(idx)
            seen_nodes += cur_batch_size

            lbl = torch.cat(
                (
                    torch.ones(cur_batch_size, 1, device=device),
                    torch.zeros(cur_batch_size * args.negsamp_ratio, 1, device=device),
                ),
                dim=0,
            )

            ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = make_batch_tensors(
                idx, subgraphs_1, subgraphs_2, adj_norm, features_t, raw_features_t, device
            )

            optimiser.zero_grad(set_to_none=True)
            logits, f_1, f_2 = model(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
            loss1 = torch.mean(b_xent(logits, lbl))
            loss2 = 0.5 * (
                mse_loss(f_1[:, -2, :], raw_bf1[:, -1, :])
                + mse_loss(f_2[:, -2, :], raw_bf2[:, -1, :])
            )
            loss = args.alpha * loss1 + args.beta * loss2
            loss.backward()
            optimiser.step()

            total_loss += float(loss.detach()) * cur_batch_size

        mean_loss = total_loss / max(seen_nodes, 1)
        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            best_state = _best_state_dict_on_cpu(model)
        else:
            cnt_wait += 1
            if cnt_wait >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float32)

    test_loop = range(args.auc_test_rounds)
    if args.tqdm:
        test_loop = tqdm(test_loop, desc="Test", position=1, leave=False)

    with torch.inference_mode():
        for round_idx in test_loop:
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            subgraphs_1 = generate_rwr_subgraph_from_neighbors(
                neighbors, args.subgraph_size, restart_prob=args.rwr_restart_prob
            )
            subgraphs_2 = generate_rwr_subgraph_from_neighbors(
                neighbors, args.subgraph_size, restart_prob=args.rwr_restart_prob
            )

            for batch_idx in range(batch_num):
                idx = all_idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                if not idx:
                    continue
                cur_batch_size = len(idx)

                ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = make_batch_tensors(
                    idx, subgraphs_1, subgraphs_2, adj_norm, features_t, raw_features_t, device
                )
                logits, dist = model.inference(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
                logits = torch.sigmoid(torch.squeeze(logits))

                if args.alpha != 0.0 and args.beta != 0.0:
                    if args.negsamp_ratio == 1:
                        ano_score_1 = -(logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                    else:
                        pos_ano_score = logits[:cur_batch_size]
                        neg_ano_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
                        ano_score_1 = -(pos_ano_score - neg_ano_score).cpu().numpy()
                    ano_score_2 = dist.cpu().numpy()
                    ano_score_1 = MinMaxScaler().fit_transform(ano_score_1.reshape(-1, 1)).reshape(-1)
                    ano_score_2 = MinMaxScaler().fit_transform(ano_score_2.reshape(-1, 1)).reshape(-1)
                    ano_score = args.alpha * ano_score_1 + args.beta * ano_score_2
                elif args.alpha != 0.0:
                    if args.negsamp_ratio == 1:
                        ano_score = -(logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                    else:
                        pos_ano_score = logits[:cur_batch_size]
                        neg_ano_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
                        ano_score = -(pos_ano_score - neg_ano_score).cpu().numpy()
                elif args.beta != 0.0:
                    ano_score = dist.cpu().numpy()
                else:
                    raise ValueError("alpha and beta cannot be zero at the same time.")

                multi_round_ano_score[round_idx, idx] = ano_score

    ano_score_final = np.mean(multi_round_ano_score, axis=0)
    roc_auc = float(roc_auc_score(ano_label, ano_score_final))
    auprc = pr_auc_score(ano_label, ano_score_final)
    return roc_auc, auprc, best_t + 1


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_bundle = load_mat(args.dataset, data_root=args.data_root)

    auc_values, auprc_values, trained_epochs = [], [], []
    trial_loop = range(args.trials)
    if args.tqdm:
        trial_loop = tqdm(trial_loop, desc="Trial", position=0, leave=True)
    for trial in trial_loop:
        roc_auc, auprc, best_epoch = train_one_trial(args, args.seed + trial, data_bundle, device)
        auc_values.append(roc_auc)
        auprc_values.append(auprc)
        trained_epochs.append(best_epoch)

    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = {
        "time": finished_at,
        "dataset": args.dataset,
        "trials": args.trials,
        "epochs": f"{np.mean(trained_epochs):.1f}({max(trained_epochs)})",
        "auc": format_metric(auc_values),
        "auprc": format_metric(auprc_values),
    }
    append_result_csv(args.result_csv, row)

    print("==============================")
    print(f"Dataset: {args.dataset}")
    print(f"Trials: {args.trials}")
    print(f"Epochs: {row['epochs']}")
    print(f"AUC: {row['auc']}")
    print(f"AUPRC: {row['auprc']}")
    print(f"CSV: {os.path.expanduser(args.result_csv)}")
    print("==============================")


if __name__ == "__main__":
    main()
