import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, remove_self_loops, to_undirected


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for _ in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            ret[cur_nd][j] = float(toks[it])
            it += 1
    return ret


def micro_f1(logits, labels):
    preds = torch.round(nn.Sigmoid()(logits)).long()
    labels = labels.long()
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return (2 * prec * rec) / (prec + rec)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def sparse_to_tuple(sparse_mx, insert_batch=False):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            shape = mx.shape
        return coords, mx.data, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def _first_existing_key(data, keys):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of keys {keys} found in .mat file. Available keys: {list(data.keys())}")


def load_mat(dataset, data_root="~/datasets/GAD/mat", train_rate=0.3, val_rate=0.1):
    """Load GAD .mat dataset from ~/datasets/GAD/mat by default.

    Expected common keys:
    - adjacency: Network or A
    - attributes: Attributes or X
    - anomaly labels: Label or gnd
    - optional class labels: Class
    """
    path = os.path.join(os.path.expanduser(data_root), f"{dataset}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = sio.loadmat(path)
    label = _first_existing_key(data, ["Label", "gnd", "label", "y"])
    attr = _first_existing_key(data, ["Attributes", "X", "features", "attr"])
    network = _first_existing_key(data, ["Network", "A", "adj", "graph"])

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    if "Class" in data:
        labels_dense = np.squeeze(np.array(data["Class"], dtype=np.int64) - 1)
        num_classes = int(np.max(labels_dense)) + 1
        labels = dense_to_one_hot(labels_dense, num_classes)
    else:
        # Some GAD datasets only provide anomaly labels. The training code does not use class labels.
        labels = np.zeros((adj.shape[0], 1), dtype=np.float32)

    ano_labels = np.squeeze(np.array(label)).astype(np.int64)
    str_ano_labels = np.squeeze(np.array(data["str_anomaly_label"])) if "str_anomaly_label" in data else None
    attr_ano_labels = np.squeeze(np.array(data["attr_anomaly_label"])) if "attr_anomaly_label" in data else None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_pyg_data(adj: sp.spmatrix, make_undirected: bool = True) -> Data:
    """Convert scipy adjacency to a PyG Data object.

    This replaces the old DGLGraph construction. The current SL-GAD training loop
    still uses dense local subgraph adjacencies, while PyG is used for graph storage
    and RWR-style subgraph sampling.
    """
    edge_index, edge_weight = from_scipy_sparse_matrix(adj.tocoo())
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=adj.shape[0])
        edge_weight = None
    return Data(edge_index=edge_index.long(), edge_weight=edge_weight, num_nodes=adj.shape[0])


def build_neighbor_lists(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    edge_index = edge_index.detach().cpu()
    neighbors = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        if u != v:
            neighbors[u].append(v)
    # Deduplicate to avoid overweighting duplicate edges after to_undirected/coalescing differences.
    return [list(dict.fromkeys(nbrs)) for nbrs in neighbors]


def _sample_rwr_unique_nodes(
    seed: int,
    neighbors: Sequence[Sequence[int]],
    target_size: int,
    restart_prob: float,
    max_steps: int,
) -> List[int]:
    """Collect unique nodes using random walk with restart.

    The returned list excludes the seed when possible. The caller appends the seed as
    the final/main-vertex position, matching the original SL-GAD layout.
    """
    if target_size <= 0:
        return []

    current = seed
    visited = []
    seen = set()
    for _ in range(max_steps):
        if random.random() < restart_prob:
            current = seed

        nbrs = neighbors[current]
        if not nbrs:
            current = seed
            continue
        current = random.choice(nbrs)

        if current != seed and current not in seen:
            seen.add(current)
            visited.append(current)
            if len(visited) >= target_size:
                break
    return visited


def generate_rwr_subgraph(
    pyg_data: Data,
    subgraph_size: int,
    restart_prob: float = 0.9,
    max_steps_factor: int = 5,
) -> List[List[int]]:
    """Generate per-node RWR subgraphs without DGL.

    Output format matches the original function:
    [neighbor_1, ..., neighbor_{subgraph_size-1}, center_node]
    """
    num_nodes = int(pyg_data.num_nodes)
    reduced_size = subgraph_size - 1
    neighbors = build_neighbor_lists(pyg_data.edge_index, num_nodes)
    subgraphs = []

    for seed in range(num_nodes):
        nodes = _sample_rwr_unique_nodes(
            seed=seed,
            neighbors=neighbors,
            target_size=reduced_size,
            restart_prob=restart_prob,
            max_steps=max(subgraph_size * max_steps_factor, 20),
        )

        retry_time = 0
        while len(nodes) < reduced_size and retry_time < 10:
            more = _sample_rwr_unique_nodes(
                seed=seed,
                neighbors=neighbors,
                target_size=reduced_size,
                restart_prob=restart_prob,
                max_steps=max(subgraph_size * max_steps_factor * 2, 40),
            )
            for node in more:
                if node not in nodes:
                    nodes.append(node)
                if len(nodes) >= reduced_size:
                    break
            retry_time += 1

        if len(nodes) < reduced_size:
            # Isolated or very low-degree node: pad deterministically with the seed.
            pad_value = nodes[-1] if nodes else seed
            nodes = (nodes + [pad_value] * reduced_size)[:reduced_size]
        else:
            nodes = nodes[:reduced_size]

        nodes.append(seed)
        subgraphs.append(nodes)
    return subgraphs
