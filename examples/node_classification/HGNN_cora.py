import time
from copy import deepcopy
from typing import List, Union

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, cooking_200, Cooking200
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

def create_e_list(incidence_matrix: np.ndarray) -> List[Union[List[int], int]]:
    e_list = []
    num_hyperedges = incidence_matrix.shape[1]

    for hyperedge_index in range(num_hyperedges):
        nodes_in_hyperedge = np.where(incidence_matrix[:, hyperedge_index] == 1)[0]
        if len(nodes_in_hyperedge) == 1:
            e_list.append(nodes_in_hyperedge[0])
        else:
            e_list.append(nodes_in_hyperedge.tolist())

    return e_list


def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    # print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":
    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = Cora()
    X, lbl = data["features"], data["labels"]
    G = Graph(data["num_vertices"], data["edge_list"])
    HG = Hypergraph.from_graph_kHop(G, k=1)
    print(f"Hypergraph: {HG}")

    best_homogeneity_threshold = 0.0
    best_similarity_threshold = 0.0
    best_res = {"accuracy": 0.0, "f1_score": 0.0}

    homogeneity_threshold = 0.0
    similarity_threshold = 0.0
    low_homogeneity_indices = HG.save_low_homogeneity_indices_per(homogeneity_threshold)
    similiar = HG.calculate_intra_edge_similarity(low_homogeneity_indices, X, HG.v2e_aggregation(X))
    v_removed = HG.save_low_similarity_indices_from_intra(similiar, similarity_threshold)
    H_dense = HG.H.to_dense().to(device)
    H_new = HG.update_edge_node_matrix(v_removed, H_dense)
    H_cpu = H_new.cpu()
    H = H_cpu.numpy()
    e_v_list = create_e_list(H)
    e_v = [e if isinstance(e, list) else [e] for e in e_v_list]
    hg = Hypergraph(num_v=2708, e_list=e_v, e_weight=[1.0] * 2590,
                            v_weight=[1.0] * 2708,
                            merge_op="mean", device=torch.device("cuda"))

    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]
    net = HGNN(data["dim_features"], 16, data["num_classes"])
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    hg = hg.to(X.device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, hg, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, hg, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, hg, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
