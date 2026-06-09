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
from utils import adj_to_pyg_data, generate_rwr_subgraph, load_mat, normalize_adj, preprocess_features, set_seed, pr_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def build_parser():
    parser = argparse.ArgumentParser(description="SL-GAD PyG refactor")
    parser.add_argument("--expid", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="BlogCatalog")
    parser.add_argument("--data_root", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--result_csv", type=str, default="results/slgad_results.csv")
    parser.add_argument("--cache_dir", type=str, default="tmp")

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


def make_batch_tensors(
    idx,
    subgraphs_1,
    subgraphs_2,
    adj,
    features,
    raw_features,
    subgraph_size,
    ft_size,
    device,
):
    cur_batch_size = len(idx)
    ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = [], [], [], [], [], []

    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size), device=device)
    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1), device=device)
    added_adj_zero_col[:, -1, :] = 1.0
    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size), device=device)

    for i in idx:
        sg1 = subgraphs_1[i]
        sg2 = subgraphs_2[i]
        cur_adj_1 = adj[:, sg1, :][:, :, sg1]
        cur_adj_2 = adj[:, sg2, :][:, :, sg2]
        cur_feat_1 = features[:, sg1, :]
        cur_feat_2 = features[:, sg2, :]
        raw_cur_feat_1 = raw_features[:, sg1, :]
        raw_cur_feat_2 = raw_features[:, sg2, :]

        ba1.append(cur_adj_1)
        ba2.append(cur_adj_2)
        bf1.append(cur_feat_1)
        bf2.append(cur_feat_2)
        raw_bf1.append(raw_cur_feat_1)
        raw_bf2.append(raw_cur_feat_2)

    ba1 = torch.cat(ba1)
    ba1 = torch.cat((ba1, added_adj_zero_row), dim=1)
    ba1 = torch.cat((ba1, added_adj_zero_col), dim=2)
    ba2 = torch.cat(ba2)
    ba2 = torch.cat((ba2, added_adj_zero_row), dim=1)
    ba2 = torch.cat((ba2, added_adj_zero_col), dim=2)

    bf1 = torch.cat(bf1)
    bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
    bf2 = torch.cat(bf2)
    bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

    raw_bf1 = torch.cat(raw_bf1)
    raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)
    raw_bf2 = torch.cat(raw_bf2)
    raw_bf2 = torch.cat((raw_bf2[:, :-1, :], added_feat_zero_row, raw_bf2[:, -1:, :]), dim=1)
    return ba1, ba2, bf1, bf2, raw_bf1, raw_bf2


def train_one_trial(args, seed, data_bundle, device) -> Tuple[float, float, int]:
    (
        adj,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        ano_label,
        str_ano_label,
        attr_ano_label,
    ) = data_bundle

    set_seed(seed)

    raw_features = features.todense()
    features, _ = preprocess_features(features)
    pyg_graph = adj_to_pyg_data(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    batch_size = args.batch_size
    subgraph_size = args.subgraph_size
    batch_num = nb_nodes // batch_size + 1

    adj_norm = normalize_adj(adj)
    adj_norm = (adj_norm + sp.eye(adj_norm.shape[0])).todense()

    features_t = torch.FloatTensor(features[np.newaxis]).to(device)
    raw_features_t = torch.FloatTensor(raw_features[np.newaxis]).to(device)
    adj_t = torch.FloatTensor(adj_norm[np.newaxis]).to(device)

    model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([args.negsamp_ratio], device=device)
    )
    mse_loss = nn.MSELoss(reduction="mean")

    cnt_wait = 0
    best = 1e9
    best_t = 0

    loop = range(args.num_epoch)
    if args.tqdm:
        loop = tqdm(loop, desc="Epoch", position=1, leave=False)
    for epoch in loop:
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.0

        subgraphs_1 = generate_rwr_subgraph(
            pyg_graph, subgraph_size, restart_prob=args.rwr_restart_prob
        )
        subgraphs_2 = generate_rwr_subgraph(
            pyg_graph, subgraph_size, restart_prob=args.rwr_restart_prob
        )

        for batch_idx in range(batch_num):
            optimiser.zero_grad()
            is_final_batch = batch_idx == (batch_num - 1)
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]
            if len(idx) == 0:
                continue
            cur_batch_size = len(idx)
            lbl = torch.unsqueeze(
                torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))),
                1,
            ).to(device)

            ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = make_batch_tensors(
                idx, subgraphs_1, subgraphs_2, adj_t, features_t, raw_features_t,
                subgraph_size, ft_size, device
            )

            logits, f_1, f_2 = model(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
            loss_all = b_xent(logits, lbl)
            loss1 = torch.mean(loss_all)
            loss2 = 0.5 * (
                mse_loss(f_1[:, -2, :], raw_bf1[:, -1, :])
                + mse_loss(f_2[:, -2, :], raw_bf2[:, -1, :])
            )
            loss = args.alpha * loss1 + args.beta * loss2
            loss.backward()
            optimiser.step()

            loss_value = float(loss.detach().cpu().item())
            if not is_final_batch:
                total_loss += loss_value

        mean_loss = (total_loss * batch_size + loss_value * cur_batch_size) / nb_nodes
        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), args.checkpoint_path)
        else:
            cnt_wait += 1
            if cnt_wait == args.patience:
                break

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float64)

    loop = range(args.auc_test_rounds)
    if args.tqdm:
        loop = tqdm(loop, desc="Test", position=1, leave=False)
    for round_idx in loop:
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        subgraphs_1 = generate_rwr_subgraph(
            pyg_graph, subgraph_size, restart_prob=args.rwr_restart_prob
        )
        subgraphs_2 = generate_rwr_subgraph(
            pyg_graph, subgraph_size, restart_prob=args.rwr_restart_prob
        )

        for batch_idx in range(batch_num):
            is_final_batch = batch_idx == (batch_num - 1)
            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]
            if len(idx) == 0:
                continue
            cur_batch_size = len(idx)

            ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = make_batch_tensors(
                idx, subgraphs_1, subgraphs_2, adj_t, features_t, raw_features_t,
                subgraph_size, ft_size, device
            )

            with torch.no_grad():
                logits, dist = model.inference(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
            logits = torch.sigmoid(torch.squeeze(logits))

            if args.alpha != 0.0 and args.beta != 0.0:
                scaler1 = MinMaxScaler()
                scaler2 = MinMaxScaler()
                if args.negsamp_ratio == 1:
                    ano_score_1 = -(
                        logits[:cur_batch_size] - logits[cur_batch_size:]
                    ).cpu().numpy()
                else:
                    pos_ano_score = logits[:cur_batch_size]
                    neg_ano_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
                    ano_score_1 = -(pos_ano_score - neg_ano_score).cpu().numpy()
                ano_score_2 = dist.cpu().numpy()
                ano_score_1 = scaler1.fit_transform(ano_score_1.reshape(-1, 1)).reshape(-1)
                ano_score_2 = scaler2.fit_transform(ano_score_2.reshape(-1, 1)).reshape(-1)
                ano_score = args.alpha * ano_score_1 + args.beta * ano_score_2
            elif args.alpha != 0.0 and args.beta == 0.0:
                if args.negsamp_ratio == 1:
                    ano_score = -(
                        logits[:cur_batch_size] - logits[cur_batch_size:]
                    ).cpu().numpy()
                else:
                    pos_ano_score = logits[:cur_batch_size]
                    neg_ano_score = logits[cur_batch_size:].view(-1, cur_batch_size).mean(dim=0)
                    ano_score = -(pos_ano_score - neg_ano_score).cpu().numpy()
            elif args.alpha == 0.0 and args.beta != 0.0:
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
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path = cache_dir / "checkpoint.pkl"

    data_bundle = load_mat(args.dataset, data_root=args.data_root)

    auc_values, auprc_values, trained_epochs = [], [], []
    loop = range(args.trials)
    if args.tqdm:
        loop = tqdm(loop, desc="Trial", position=0, leave=True)
    for trial in loop:
        roc_auc, auprc, best_epoch = train_one_trial(args, args.seed + trial, data_bundle, device)
        auc_values.append(roc_auc)
        auprc_values.append(auprc)
        trained_epochs.append(best_epoch)

    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = {
        "time": finished_at,
        "dataset": args.dataset,
        "trials": args.trials,
        "auc": format_metric(auc_values),
        "auprc": format_metric(auprc_values),
    }
    append_result_csv(args.result_csv, row)

    print("==============================")
    print(f"Dataset: {args.dataset}")
    print(f"Trials: {args.trials}")
    print(f"AUC: {row['auc']}")
    print(f"AUPRC: {row['auprc']}")
    print(f"CSV: {os.path.expanduser(args.result_csv)}")
    print("==============================")


if __name__ == "__main__":
    main()
