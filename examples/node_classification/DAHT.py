import time
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg.data import Cora, Pubmed
from dhg.models import HGNN
from dhg.models.hypergraphs.HGNNTransformer import HGNNTransformer
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.structure.hypergraphs import Hypergraph
from dhg.structure.graphs import Graph


def train(net, X, hg, lbls, train_idx, optimizer, epoch):
    net.train()
    st = time.time()
    optimizer.zero_grad()

    outs = net(X, hg)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, hg, lbls, idx, test=False):
    net.eval()
    outs = net(X, hg)
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
    # data = Pubmed()
    X, lbl = data["features"], data["labels"]
    G = Graph(data["num_vertices"], data["edge_list"])
    HG = Hypergraph.from_graph_kHop(G, k=1)  # 使用Graph对象生成Hypergraph

    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]


    net = HGNNTransformer(data["dim_features"], 512, data["num_classes"], num_heads=8, drop_rate=0.9)

    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-05)

    X, lbl = X.to(device), lbl.to(device)
    HG = HG.to(X.device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        train(net, X, HG, lbl, train_mask, optimizer, epoch)
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, HG, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, HG, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)