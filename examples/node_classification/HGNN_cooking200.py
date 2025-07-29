import time
from copy import deepcopy
from typing import List, Union

import torch
import torch.optim as optim
import torch.nn.functional as F
from flatbuffers.builder import np

from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, cooking_200, Cooking200
from dhg.models import HGNN, HGNNP, HNHN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

# 遍历稠密图，找到超边节点关系
def create_e_list(incidence_matrix: np.ndarray) -> List[Union[List[int], int]]:
    e_list = []
    num_hyperedges = incidence_matrix.shape[1]

    for hyperedge_index in range(num_hyperedges):
        # 找到属于当前超边的所有节点的索引
        nodes_in_hyperedge = np.where(incidence_matrix[:, hyperedge_index] == 1)[0]

        # 如果只有一个节点，直接添加节点索引；如果有多个节点，添加节点索引列表
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
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
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
    # data = Cora()
    # data = Pubmed()
    # data = Citeseer()
    data = Cooking200()
    X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    # X, lbl = data["features"], data["labels"]
    # G = Graph(data["num_vertices"], data["edge_list"])
    HG = Hypergraph(data["num_vertices"], data["edge_list"])
    # HG = Hypergraph.from_graph_kHop(G, k=1)

    # HG = Hypergraph.from_feature_kNN(X,k=1)
    print(f"Hypergraph: {HG}")
    # print("节点特征")
    # print(X)
    # # 打印返回的索引
    # print("低同质性超边")
    # print(low_homogeneity_indices)
    # print(HG.homogeneity)
    # print(hg.homogeneities())
    # print("超边特征")
    # print(hg.v2e_aggregation(X))
    # Xe = hg.v2e_aggregation(X)
    # intra_edge_similarity = hg.calculate_intra_edge_similarity(low_homogeneity_indices, X , Xe)
    # print(intra_edge_similarity)
    # hg = Hypergraph(num_v=5, e_list=[[0, 1, 2], [2, 3], [1, 3, 4], [1, 2]], e_weight=[1.0, 1.0, 1.0, 1.0],
    #                 v_weight=[1.0] * 5,
    #                 merge_op="mean", device=torch.device("cpu"))
    # X = torch.randn(5, 3)
    # print("初始e_list")
    # print(HG.e_con)
    # print("用create_e_list方法构造的e_list")
    # print(create_e_list(HG.H.to_dense()))
    # print("初始关联矩阵")
    # print(HG.H)
    # low_homogeneity_indices = HG.save_low_homogeneity_indices_data()
    low_homogeneity_indices = HG.save_low_homogeneity_indices_per(0.3)
    # print("低同质性超边_阈值")
    # print(low_homogeneity_indices)
    print("低同质性超边_比例")
    print(HG.save_low_homogeneity_indices_per(0.3))
    current_device = torch.cuda.current_device()
    # print(f"Current device: {current_device}")


    # H_dense[0][0] = 0
    # H_cpu = H_dense.cpu()
    # H = H_cpu.numpy()
    #
    # e_list = create_e_list(H)
    # hg = Hypergraph(num_v=2708, e_list=e_list, e_weight=[1.0] * 2590,
    #                                 v_weight=[1.0] * 2708,

    #                                 merge_op="mean", device=torch.device("cuda"))
    # print("========================================")
    # print(e_list)
    # print("计算超边和节点余弦相似度")
    similiar = HG.calculate_intra_edge_similarity(low_homogeneity_indices,X,HG.v2e_aggregation(X))
    # print(HG.calculate_intra_edge_similarity(low_homogeneity_indices,X,HG.v2e_aggregation(X)))
    print("余弦相似度")
    print(similiar)
    print("保存需删除节点超边信息_阈值")
    v_removed = HG.save_low_similarity_indices_from_intra(similiar,1)
    print(HG.save_low_similarity_indices_from_intra(similiar,1))

    H_dense = HG.H.to_dense().to(device)
    # 将关联矩阵转换为稠密矩阵
    # print("原关联矩阵")
    # print(HG.H.to_dense())
    # print("更新关联矩阵")
    H_new = HG.update_edge_node_matrix(v_removed,H_dense)
    # print(H_new)
    # print("判断两个矩阵是否相等")
    # are_equal = np.array_equal(HG.H.to_dense(), H_new)
    # print(are_equal)


    H_cpu = H_new.cpu()
    H = H_cpu.numpy()
    e_v_list = create_e_list(H)
    # print("新超边节点结构")
    # print(e_v_list)
    e_v  = [e if isinstance(e, list) else [e] for e in e_v_list]

    hg = Hypergraph(num_v=7403, e_list=e_v, e_weight=[1.0] * 2750,
                                                    v_weight=[1.0] * 7403,
                                                    merge_op="mean", device=torch.device("cuda"))
    # hg = Hypergraph(num_v=2708, e_list=e_v, e_weight=[1.0] * 2590,
    #                                                 v_weight=[1.0] * 2708,
    #                                                 merge_op="mean", device=torch.device("cuda"))
    print("更新后超图关联矩阵")
    print(hg.H)
    # print("保存需删除节点超边信息_比例")
    # print(HG.save_lowest_percentage_similarity_indices_from_intra(similiar,percentage=0.05))

    # H_dense[0][0] = 0
    # print(H_dense)
    # H = H_dense.to_sparse_coo().to(device)
    # print(H)



    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=True)
    # net = HNHN(data["dim_features"], 16, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    HG = HG.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, HG, lbl, train_mask, optimizer, epoch)
        # validation
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
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, HG, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
