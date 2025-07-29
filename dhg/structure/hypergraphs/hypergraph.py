import itertools
import random
import pickle
from pathlib import Path
from copy import deepcopy
from typing import Optional, Union, List, Tuple, Dict, Any, TYPE_CHECKING

import torch
import scipy.spatial
from networkx.drawing.tests.test_pylab import plt

from dhg.structure import BaseHypergraph
from dhg.visualization.structure.draw import draw_hypergraph
from dhg.utils.sparse import sparse_dropout

import matplotlib.pyplot as plt
import tkinter
from tkinter import *

from ...visualization.feature.utils import normalize

if TYPE_CHECKING:
    from ..graphs import Graph, BiGraph


class Hypergraph(BaseHypergraph):
    r"""The ``Hypergraph`` class is developed for hypergraph structures.

    Args:
        ``num_v`` (``int``): The number of vertices in the hypergraph.
        ``e_list`` (``Union[List[int], List[List[int]]]``, optional): A list of hyperedges describes how the vertices point to the hyperedges. Defaults to ``None``.
        ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
        ``v_weight`` (``Union[List[float]]``, optional): A list of weights for vertices. If set to ``None``, the value ``1`` is used for all vertices. Defaults to ``None``.
        ``merge_op`` (``str``): The operation to merge those conflicting hyperedges in the same hyperedge group, which can be ``'mean'``, ``'sum'`` or ``'max'``. Defaults to ``'mean'``.
        ``device`` (``torch.device``, optional): The deivce to store the hypergraph. Defaults to ``torch.device('cpu')``.
    """

    # 初始化超图对象
    def __init__(
        self,
        num_v: int,
        e_list: Optional[Union[List[int], List[List[int]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        v_weight: Optional[List[float]] = None,
        merge_op: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(num_v, device=device)
        # init vertex weight
        if v_weight is None:
            self._v_weight = [1.0] * self.num_v
        else:
            assert len(v_weight) == self.num_v, "The length of vertex weight is not equal to the number of vertices."
            self._v_weight = v_weight
        # init hyperedges
        if e_list is not None:
            self.add_hyperedges(e_list, e_weight, merge_op=merge_op)
            # self.e_list = e_list

    def __repr__(self) -> str:
        r"""打印超边信息（节点数和超边数）.
        """
        return f"Hypergraph(num_v={self.num_v}, num_e={self.num_e})"

    # @property：这是一个装饰器，用于将方法转换为属性。
    @property
    def state_dict(self) -> Dict[str, Any]:
        # 返回的字典中包含了超图的顶点数和每个超边及其权重
        r"""Get the state dict of the hypergraph.
        """
        return {"num_v": self.num_v, "raw_groups": self._raw_groups}


    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's hypergraph structure a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to store the DHG's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.parent.exists(), "The directory does not exist."
        data = {
            "class": "Hypergraph",
            "state_dict": self.state_dict,
        }
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(file_path: Union[str, Path]):
        # 从文件中加载超图的结构
        r"""Load the DHG's hypergraph structure from a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to load the DHG's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.exists(), "The file does not exist."
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        assert data["class"] == "Hypergraph", "The file is not a DHG's hypergraph file."
        return Hypergraph.from_state_dict(data["state_dict"])

    def draw(
        self,
        e_style: str = "circle",
        v_label: Optional[List[str]] = None,
        v_size: Union[float, list] = 1.0,
        v_color: Union[str, list] = "r",
        v_line_width: Union[str, list] = 1.0,
        e_color: Union[str, list] = "gray",
        e_fill_color: Union[str, list] = "whitesmoke",
        e_line_width: Union[str, list] = 1.0,
        font_size: float = 1.0,
        font_family: str = "sans-serif",
        push_v_strength: float = 1.0,
        push_e_strength: float = 1.0,
        pull_e_strength: float = 1.0,
        pull_center_strength: float = 1.0,
    ):
        r"""Draw the hypergraph structure.

        Args:
            ``e_style`` (``str``): The style of hyperedges. The available styles are only ``'circle'``. Defaults to ``'circle'``.
            ``v_label`` (``list``): The labels of vertices. Defaults to ``None``.
            ``v_size`` (``float`` or ``list``): The size of vertices. Defaults to ``1.0``.
            ``v_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices. Defaults to ``'r'``.
            ``v_line_width`` (``float`` or ``list``): The line width of vertices. Defaults to ``1.0``.
            ``e_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'gray'``.
            ``e_fill_color`` (``str`` or ``list``): The fill `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'whitesmoke'``.
            ``e_line_width`` (``float`` or ``list``): The line width of hyperedges. Defaults to ``1.0``.
            ``font_size`` (``float``): The font size of labels. Defaults to ``1.0``.
            ``font_family`` (``str``): The font family of labels. Defaults to ``'sans-serif'``.
            ``push_v_strength`` (``float``): The strength of pushing vertices. Defaults to ``1.0``.
            ``push_e_strength`` (``float``): The strength of pushing hyperedges. Defaults to ``1.0``.
            ``pull_e_strength`` (``float``): The strength of pulling hyperedges. Defaults to ``1.0``.
            ``pull_center_strength`` (``float``): The strength of pulling vertices to the center. Defaults to ``1.0``.
        """
        draw_hypergraph(
            self,
            e_style,
            v_label,
            v_size,
            v_color,
            v_line_width,
            e_color,
            e_fill_color,
            e_line_width,
            font_size,
            font_family,
            push_v_strength,
            push_e_strength,
            pull_e_strength,
            pull_center_strength,
        )

    def clear(self):
        r"""Clear all hyperedges and caches from the hypergraph.
        # 创建超图实例
            hg = Hypergraph(num_v=5, e_list=[[0, 1, 2], [2, 3], [1, 3, 4]], e_weight=[1.0, 1.0, 1.0], v_weight=[1.0] * 5, merge_op="mean", device=torch.device("cpu"))

        # 清除超图
            hg.clear()
        """
        return super().clear()

    def clone(self) -> "Hypergraph":
        r"""Return a copy of the hypergraph.
        """
        hg = Hypergraph(self.num_v, device=self.device)
        hg._raw_groups = deepcopy(self._raw_groups)
        hg.cache = deepcopy(self.cache)
        hg.group_cache = deepcopy(self.group_cache)
        return hg

    def to(self, device: torch.device):
        r"""Move the hypergraph to the specified device.
        将超图从一个设备移动到另一个设备，例如从 CPU 移动到 GPU，以加速计算。
        Args:
            ``device`` (``torch.device``): The target device.
        """
        return super().to(device)

    # def remove_node_from_hyperedge(raw_groups, edge_tuple, node_to_remove):
    #     """
    #     Remove a node from a specified hyperedge in the raw_groups dictionary.
    #
    #     Args:
    #         raw_groups (dict): The raw_groups dictionary containing hyperedges.
    #         edge_tuple (tuple): The hyperedge as a tuple of node indices.
    #         node_to_remove (int): The index of the node to be removed.
    #     """
    #     # 遍历过每个超边组
    #     for group_name, edges in raw_groups.items():
    #         # 遍过每个超边
    #         for edge, edge_info in edges.items():
    #             # 检查要删除的节点是否在超边中
    #             if node_to_remove in edge:
    #                 # 从超边中移除节点
    #                 new_edge = tuple([node for node in edge if node != node_to_remove])
    #                 # 更新超边组字典
    #                 raw_groups[group_name][new_edge] = edge_info
    #                 # 如果超边为空，则删除超边
    #                 if not new_edge:
    #                     del raw_groups[group_name][edge]
    #     return raw_groups



    # =====================================================================================
    # some construction functions
    def generate_point(self,x_range, y_range, num_points):
        """
        生成指定范围内不重复的二维坐标点

        参数:
            x_range (tuple): x坐标的范围，格式为(min_x, max_x)
            y_range (tuple): y坐标的范围，格式为(min_y, max_y)
            num_points (int): 需要生成的坐标点数量

        返回:
            list: 包含生成的不重复二维坐标点的列表，每个坐标点是一个元组(x, y)
        """
        coordinates = set()  # 使用集合存储坐标点，自动去重
        max_attempts = num_points * 10  # 设置最大尝试次数，避免死循环
        attempts = 0

        while len(coordinates) < num_points and attempts < max_attempts:
            x = random.randint(x_range[0], x_range[1])  # 随机生成x坐标
            y = random.randint(y_range[0], y_range[1])  # 随机生成y坐标
            coordinates.add((x, y))  # 将生成的坐标点添加到集合中
            attempts += 1

        if len(coordinates) < num_points:
            return list(coordinates)  # 将集合转换为列表返回

        return list(coordinates)  # 将集合转换为列表返回

    def invert_matrix(self,matrix, coordinates):
        """
        根据给定的坐标列表，将矩阵中对应位置的元素进行反转（1变0，0变1）

        参数:
            matrix (numpy.ndarray): 输入的矩阵
            coordinates (list): 包含二维坐标点的列表，每个坐标点是一个元组(x, y)

        返回:
            numpy.ndarray: 修改后的矩阵
        """
        for x, y in coordinates:
                # 反转矩阵中对应位置的元素
                matrix[x, y] = 1 - matrix[x, y]
        return matrix

    def homogeneities(self) -> torch.Tensor:
        # 计算所有超边的同质性
        homogeneities = []
        for e_idx in range(self.num_e):
            homogeneity = self.homogeneity(e_idx)
            homogeneities.append(homogeneity)
        return torch.tensor(homogeneities, dtype=torch.float32)

    def homogeneity(self, e_idx: int) -> float:
        # 获取超边 e_idx 中的顶点
        e = self.N_v(e_idx).tolist()
        if len(e) <= 1:
            return 0.0

        # 计算超边内部顶点对的数量
        num_pairs = len(list(itertools.combinations(e, 2)))

        # 计算每对顶点之间的边数
        total_edges = 0
        for u, v in itertools.combinations(e, 2):
            total_edges += self.count_edges_between(u, v)

        # 计算同质性
        homogeneity = total_edges / num_pairs
        return homogeneity

    def count_edges_between(hg, node1, node2):
        # 获取每个节点的超边列表
        hyperedges_node1 = hg.N_e(node1).tolist()
        hyperedges_node2 = hg.N_e(node2).tolist()

        # 找到共同的超边
        common_hyperedges = set(hyperedges_node1) & set(hyperedges_node2)

        # 返回共同超边的数量
        return len(common_hyperedges)

        # 保存同质性低的节点

    def save_low_homogeneity_indices_data(self, threshold: float = 1.6):
        # 获取所有超边的同质性
        homogeneities = self.homogeneities()

        # 获取同质性小于阈值的超边索引
        low_homogeneity_indices = [idx for idx, homogeneity in enumerate(homogeneities) if homogeneity < threshold]

        # 保存这些索引，可以在类的属性中保存或者返回它们
        self.low_homogeneity_indices = low_homogeneity_indices
        return low_homogeneity_indices

    def save_low_homogeneity_indices_per(self, top_percentage: float = 0.2):
        # 获取所有超边的同质性
        homogeneities = self.homogeneities()

        # 检查homogeneities是否为空
        if torch.numel(homogeneities) == 0:
            print("没有超边的同质性信息可供处理。")
            self.low_homogeneity_indices = []
            return []

        # 将同质性与索引一起排序，确保最低的同质性排在前面
        sorted_homogeneities = sorted(enumerate(homogeneities), key=lambda x: x[1])

        # 计算20%的超边数量
        num_hyperedges = len(homogeneities)
        num_low_homogeneity = int(num_hyperedges * top_percentage)

        # 选择同质性最低的20%的超边索引
        low_homogeneity_indices = [idx for idx, _ in sorted_homogeneities[:num_low_homogeneity]]

        # 保存这些索引，可以在类的属性中保存或者返回它们
        self.low_homogeneity_indices = low_homogeneity_indices
        return low_homogeneity_indices

    def cosine_similarity(self, node_features, edge_features):
        # 计算两个向量的余弦相似度
        norm_node_features = torch.nn.functional.normalize(node_features, p=2, dim=0)
        norm_edge_features = torch.nn.functional.normalize(edge_features, p=2, dim=0)
        return torch.dot(norm_node_features, norm_edge_features)

    # def cosine_similarity(self, node_features, edge_features):
    #     """
    #     Calculate the cosine similarity between node features and edge features.
    #
    #     Args:
    #         node_features (torch.Tensor): Node feature matrix. Size (num_v, num_features).
    #         edge_features (torch.Tensor): Edge feature matrix. Size (num_e, num_features).
    #
    #     Returns:
    #         torch.Tensor: Cosine similarity matrix. Size (num_v, num_e).
    #     """
    #     # Normalize the node features
    #     norm_node_features = torch.nn.functional.normalize(node_features, p=2, dim=1)
    #
    #     # Normalize the edge features
    #     norm_edge_features = torch.nn.functional.normalize(edge_features, p=2, dim=1)
    #
    #     # Compute the cosine similarity matrix
    #     similarity_matrix = torch.mm(norm_node_features, norm_edge_features.t())
    #
    #     return similarity_matrix

    # def calculate_intra_edge_similarity(self, non_homogeneity_indices, X):
    #     intra_edge_similarity = {e_idx: [] for e_idx in non_homogeneity_indices}
    #
    #     for e_idx in non_homogeneity_indices:
    #         # 获取超边e_idx中的节点列表
    #         edge_nodes = self.N_v(e_idx)  # 假设 N_v是一个方法，返回给定超边索引的节点列表
    #
    #         # 检查 edge_nodes 是否为空
    #         if not edge_nodes:
    #             edge_features = torch.mean(X[edge_nodes], dim=0)
    #         else:
    #             # 如果超边为空，使用第一个节点特征作为超边特征
    #             edge_features = X[0]  # 或者其他适当的默认值
    #
    #         # 计算每个节点与超边的余弦相似度
    #         similarities = [self.cosine_similarity(X[node], edge_features) for node in edge_nodes]
    #
    #         # 记录每个节点与超边的余弦相似度
    #         intra_edge_similarity[e_idx] = similarities
    #
    #     return intra_edge_similarity


    # def calculate_intra_edge_similarity(self, non_homogeneity_indices, X, Xe):
    #     intra_edge_similarity = {}
    #     for e_idx in non_homogeneity_indices:
    #         # 获取超边e_idx中的节点列表
    #         edge_nodes = self.N_v(e_idx)
    #         # 获取对应序号超边的特征
    #         edge_feature = Xe[e_idx]
    #         # 计算每个节点与超边的余弦相似度
    #         similarities = [self.cosine_similarity(X[node], edge_feature) for node in edge_nodes]
    #         # 记录每个节点与超边的余弦相似度
    #         intra_edge_similarity[e_idx] = similarities
    #
    #     return intra_edge_similarity

    def calculate_intra_edge_similarity(self, non_homogeneity_indices, X, Xe):
        intra_edge_similarity = {}
        for e_idx in non_homogeneity_indices:
            # 获取超边e_idx中的节点列表
            edge_nodes = self.N_v(e_idx)
            # 获取对应序号超边的特征
            edge_feature = Xe[e_idx]
            # 计算每个节点与超边的余弦相似度，并保存节点的全局索引
            similarities_with_indices = [
                (node, self.cosine_similarity(X[node], edge_feature))
                for node in edge_nodes
            ]
            # 记录每个节点的全局索引与超边的余弦相似度
            intra_edge_similarity[e_idx] = similarities_with_indices

        return intra_edge_similarity

    def update_edge_node_matrix(self, low_similarity_indices, edge_node_matrix):
        for e_idx_node_pair in low_similarity_indices:
            # 检查 e_idx_node_pair 是否为元组，并且包含两个元素
            if not isinstance(e_idx_node_pair, tuple) or len(e_idx_node_pair) != 2:
                raise ValueError("Each element of low_similarity_indices must be a tuple of (e_idx, node_index)")

            # 解包 e_idx_node_pair 为 e_idx 和 node_index
            e_idx, node_index = e_idx_node_pair
            # 确保 node_index 是一个张量，并且将其转换为 Python 标量
            node_index = node_index.item() if isinstance(node_index, torch.Tensor) else node_index
            # 将矩阵中对应的节点和超边关系从1改为0
            edge_node_matrix[node_index, e_idx] = 0

        return edge_node_matrix

    def save_low_similarity_indices_from_intra(self, intra_edge_similarity, threshold):
        low_similarity_indices = []

        for e_idx, similarities_with_indices in intra_edge_similarity.items():
            for node_with_similarity in similarities_with_indices:
                # node_with_similarity 是一个元组，包含节点索引和相似度值
                node_index, similarity = node_with_similarity
                if similarity < threshold:
                    # 这里我们使用节点的全局索引和超边索引
                    low_similarity_indices.append((e_idx, node_index))

        return low_similarity_indices

    # def save_lowest_percentage_similarity_indices_from_intra(self, intra_edge_similarity, percentage=0.2):
    #     low_similarity_indices = []
    #
    #     for e_idx, similarities_with_indices in intra_edge_similarity.items():
    #         # similarities_with_indices 是一个包含 (节点全局索引, 相似度) 元组的列表
    #         # 将节点索引和相似度值配对，并排序
    #         sorted_similarities = sorted(similarities_with_indices, key=lambda x: x[1])
    #         num_nodes = len(similarities_with_indices)
    #         num_lowest = max(1, int(percentage * num_nodes))  # 根据百分比计算节点数量
    #
    #         # 获取相似度最低的指定百分比的节点索引和对应的相似度
    #         lowest_percentage = sorted_similarities[:num_lowest]
    #         low_similarity_indices.extend([(e_idx, node_index) for node_index, _ in lowest_percentage])
    #
    #     return low_similarity_indices



    # def remove_hyperedges_below_homogeneity(self, threshold: float = 1.25):
    #     # 获取所有超边的同质性
    #     homogeneities = self.homogeneities()
    #
    #     # 获取同质性小于阈值的超边索引
    #     low_homogeneity_indices = [idx for idx, homogeneity in enumerate(homogeneities) if homogeneity < threshold]
    #
    #     # 根据索引获取对应的超边信息
    #     edges_to_remove = [self.N_v(idx) for idx in low_homogeneity_indices]
    #     # print("=============================")
    #     # print("删除超边信息",edges_to_remove)
    #
    #     # 使用列表推导式将 tensor 对象转换为列表
    #     edges_to_remove_list = [edge.tolist() for edge in edges_to_remove]
    #
    #     # 删除这些超边
    #     self.remove_hyperedges(edges_to_remove_list)
    #
    #
    # def remove_bottom_20_percent_hyperedges(self):
    #     # 获取所有超边的同质性
    #     homogeneities = self.homogeneities()
    #
    #     # 计算要移除的超边数量，至少为1
    #     num_hyperedges_to_remove = max(1 , int(len(homogeneities) * 0.1))
    #
    #     # 按同质性值对超边进行排序，并获取最低20%的超边索引
    #     sorted_indices = torch.argsort(homogeneities)
    #     bottom_20_percent_indices = sorted_indices[:num_hyperedges_to_remove]
    #
    #     # 确保不会尝试移除不存在的超边
    #     if len(bottom_20_percent_indices) > 0:
    #         # 根据索引获取对应的超边信息
    #         edges_to_remove = [self.N_v(idx.item()) for idx in bottom_20_percent_indices]
    #
    #         # 使用列表推导式将 tensor 对象转换为列表
    #         edges_to_remove_list = [edge.tolist() for edge in edges_to_remove]
    #
    #         # 删除这些超边
    #         self.remove_hyperedges(edges_to_remove_list)

    # def cosine_similarity(self, node_features, edge_features):
    #     """
    #     Calculate the cosine similarity between node features and edge features.
    #
    #     Args:
    #         node_features (torch.Tensor): Node feature matrix. Size (num_v, num_features).
    #         edge_features (torch.Tensor): Edge feature matrix. Size (num_e, num_features).
    #
    #     Returns:
    #         torch.Tensor: Cosine similarity matrix. Size (num_v, num_e).
    #     """
    #     # Normalize the node features
    #     norm_node_features = torch.nn.functional.normalize(node_features, p=2, dim=1)
    #
    #     # Normalize the edge features
    #     norm_edge_features = torch.nn.functional.normalize(edge_features, p=2, dim=1)
    #
    #     # Compute the cosine similarity matrix
    #     similarity_matrix = torch.mm(norm_node_features, norm_edge_features.t())
    #
    #     return similarity_matrix

    @staticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the hypergraph from the state dict.
            从一个字典中恢复超图，使你能够重新加载之前保存的超图状态。
        Args:
            ``state_dict`` (``dict``): The state dict to load the hypergraph.
        """
        _hg = Hypergraph(state_dict["num_v"])
        _hg._raw_groups = deepcopy(state_dict["raw_groups"])
        return _hg

    @staticmethod
    def _e_list_from_feature_kNN(features: torch.Tensor, k: int):
        r"""Construct hyperedges from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.
        从特征矩阵构建超边列表
        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
        """
        features = features.cpu().numpy()
        # torch.Tensor是PyTorch库中的一种数据结构，用于存储多维数组（如矩阵）  NumPy 数组是 NumPy 库中的一种数据结构，也用于存储多维数组。
        # cKDTree只支持NumPy数组
        assert features.ndim == 2, "The feature matrix should be 2-D."
        assert (
            k <= features.shape[0]
        ), "The number of nearest neighbors should be less than or equal to the number of vertices."
        tree = scipy.spatial.cKDTree(features)
        # k-d树是一种用于高效查询最近邻的数据结构
        _, nbr_array = tree.query(features, k=k)
        # 使用k - d树查询每个顶点的k个最近邻。tree.query返回两个数组：第一个数组是距离，第二个数组是最近邻的索引。我们只关心最近邻的索引，所以用_忽略第一个数组
        return nbr_array.tolist()
    # 将最近邻数组转换为列表形式并返回。每个子列表表示一个超边，包含一个中心顶点及其k - 1个最近邻顶点

    @staticmethod
    def from_feature_kNN(features: torch.Tensor, k: int, device: torch.device = torch.device("cpu")):
        r"""Construct the hypergraph from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

        .. note::
            The constructed hypergraph is a k-uniform hypergraph. If the feature matrix has the size :math:`N \times C`, the number of vertices and hyperedges of the constructed hypergraph are both :math:`N`.

        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_feature_kNN(features, k)
        hg = Hypergraph(features.shape[0], e_list, device=device)
        # 用于获取features这个多维数组（通常是torch.Tensor或NumPy数组）的第一个维度的大小。具体来说，它表示features矩阵的行数，也就是顶点的数量。
        return hg

    @staticmethod
    def from_graph(graph: "Graph", device: torch.device = torch.device("cpu")) -> "Hypergraph":
        r"""Construct the hypergraph from the graph. Each edge in the graph is treated as a hyperedge in the constructed hypergraph.
        用于从图（Graph）构建超图（Hypergraph）。这个方法的核心思想是将图中的每条边视为超图中的一个超边。
        .. note::
            The construsted hypergraph is a 2-uniform hypergraph, and has the same number of vertices and edges/hyperedges as the graph.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list, e_weight = graph.e
        # graph.e是一个包含两个元素的元组，第一个元素是边列表e，第二个元素是边权重列表e_weight  这行代码将 graph.e 中的两个元素分别赋值给 e_list 和 e_weight
        hg = Hypergraph(graph.num_v, e_list, e_weight=e_weight, device=device)
        # 位置参数：必须按顺序提供，并且必须在所有关键字参数之前。关键字参数：可以按任意顺序提供，通过参数名来指定参数值。默认值参数：可以选择不提供，使用默认值。
        return hg


    @staticmethod
    def _e_list_from_graph_kHop(graph: "Graph", k: int, only_kHop: bool = False,) -> List[tuple]:
        r"""Construct the hyperedge list from the graph by k-Hop neighbors. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.
        根据k近邻结构构建超边
        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``, optional): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
        """
        assert k >= 1, "The number of hop neighbors should be larger than or equal to 1."
        A_1, A_k = graph.A.clone(), graph.A.clone()
        A_history = []
        # A_1和A_k都是图的邻接矩阵的副本。A_history用于存储中间结果。
        for _ in range(k - 1):
            A_k = torch.sparse.mm(A_k, A_1)
            if not only_kHop:
                A_history.append(A_k.clone())
        if not only_kHop:
            A_k = A_1
            for A_ in A_history:
                A_k = A_k + A_
        e_list = [
            tuple(set([v_idx] + A_k[v_idx]._indices().cpu().squeeze(0).tolist())) for v_idx in range(graph.num_v)
        ]
        return e_list


    @staticmethod
    def from_graph_kHop(
        graph: "Graph", k: int, only_kHop: bool = False, device: torch.device = torch.device("cpu"),
    ) -> "Hypergraph":
        r"""Construct the hypergraph from the graph by k-Hop neighbors. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.
        根据k近邻构建超图
        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_graph_kHop(graph, k, only_kHop)
        hg = Hypergraph(graph.num_v, e_list, device=device)
        return hg

    @staticmethod
    def _e_list_from_bigraph(bigraph: "BiGraph", U_as_vertex: bool = True) -> List[tuple]:
        r"""Construct hyperedges from the bipartite graph.
        根据二部图构建超边

        Args:
            ``bigraph`` (``BiGraph``): The bipartite graph to construct the hypergraph.
            ``U_as_vertex`` (``bool``, optional): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
        """
        e_list = []
        if U_as_vertex:
            for v in range(bigraph.num_v):
                u_list = bigraph.nbr_u(v)
                if len(u_list) > 0:
                    e_list.append(u_list)
        else:
            for u in range(bigraph.num_u):
                v_list = bigraph.nbr_v(u)
                if len(v_list) > 0:
                    e_list.append(v_list)
        return e_list

    @staticmethod
    def from_bigraph(
        bigraph: "BiGraph", U_as_vertex: bool = True, device: torch.device = torch.device("cpu")
    ) -> "Hypergraph":
        r"""Construct the hypergraph from the bipartite graph.
        根据二部图构建超图
        Args:
            ``bigraph`` (``BiGraph``): The bipartite graph to construct the hypergraph.
            ``U_as_vertex`` (``bool``, optional): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_bigraph(bigraph, U_as_vertex)
        if U_as_vertex:
            hg = Hypergraph(bigraph.num_u, e_list, device=device)
        else:
            hg = Hypergraph(bigraph.num_v, e_list, device=device)
        return hg

    # =====================================================================================
    # some structure modification functions
    def add_hyperedges(
        self,
        e_list: Union[List[int], List[List[int]]],
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
        group_name: str = "main",
    ):
        r"""Add hyperedges to the hypergraph. If the ``group_name`` is not specified, the hyperedges will be added to the default ``main`` hyperedge group.

        Args:
            向已存在的超图中添加超边，而不是生成一个新的超图。具体来说，这个方法允许你在现有的超图结构中增加新的超边，并且可以为这些超边指定权重和处理冲突的策略。
            ``num_v`` (``int``): The number of vertices in the hypergraph.
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges. The possible values are ``"mean"``, ``"sum"``, and ``"max"``. Defaults to ``"mean"``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        e_list = self._format_e_list(e_list)
        # _format_e_list方法确保每个超边都是一个顶点列表。例如，如果e_list是[1, [2, 3], 4]，则格式化后变为[[1], [2, 3], [4]]
        if e_weight is None:
            e_weight = [1.0] * len(e_list)
            # 如果e_weight为None，则所有超边的权重默认为1.0
        elif type(e_weight) in (int, float):
            e_weight = [e_weight]
            # 如果e_weight是一个int或float，则将其转换为一个包含相同值的列表
        elif type(e_weight) is list:
            pass
        # 如果e_weight已经是一个列表，则不做任何处理。
        else:
            raise TypeError(f"The type of e_weight should be float or list, but got {type(e_weight)}")
        # 如果e_weight的类型既不是int、float也不是list，则抛出TypeError
        assert len(e_list) == len(e_weight), "The number of hyperedges and the number of weights are not equal."

        for _idx in range(len(e_list)):
            self._add_hyperedge(
                self._hyperedge_code(e_list[_idx], e_list[_idx]), {"w_e": float(e_weight[_idx])}, merge_op, group_name,
            )
            # 调用_add_hyperedge方法添加超边。
            # _hyperedge_code方法用于生成超边的唯一标识码。
        self._clear_cache(group_name)

    def add_hyperedges_from_feature_kNN(self, feature: torch.Tensor, k: int, group_name: str = "main"):
        r"""Add hyperedges from the feature matrix by k-NN. Each hyperedge is constructed by the central vertex and its :math:`k`-Nearest Neighbor vertices.
            用于从特征矩阵中通过 k-最近邻 (k-NN) 算法构建超边，并将这些超边添加到超图中。
        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert (
            feature.shape[0] == self.num_v
        ), "The number of vertices in the feature matrix is not equal to the number of vertices in the hypergraph."
        e_list = Hypergraph._e_list_from_feature_kNN(feature, k)
        self.add_hyperedges(e_list, group_name=group_name)

    def add_hyperedges_from_graph(self, graph: "Graph", group_name: str = "main"):
        r"""Add hyperedges from edges in the graph. Each edge in the graph is treated as a hyperedge.

        Args:
            ``graph`` (``Graph``): The graph to join the hypergraph.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == graph.num_v, "The number of vertices in the hypergraph and the graph are not equal."
        e_list, e_weight = graph.e
        self.add_hyperedges(e_list, e_weight=e_weight, group_name=group_name)

    def add_hyperedges_from_graph_kHop(
        self, graph: "Graph", k: int, only_kHop: bool = False, group_name: str = "main"
    ):
        r"""Add hyperedges from vertices and its k-Hop neighbors in the graph. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to join the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == graph.num_v, "The number of vertices in the hypergraph and the graph are not equal."
        e_list = Hypergraph._e_list_from_graph_kHop(graph, k, only_kHop=only_kHop)
        self.add_hyperedges(e_list, group_name=group_name)

    def add_hyperedges_from_bigraph(self, bigraph: "BiGraph", U_as_vertex: bool = False, group_name: str = "main"):
        r"""Add hyperedges from the bipartite graph.

        Args:
            ``bigraph`` (``BiGraph``): The bigraph to join the hypergraph.
            ``U_as_vertex`` (``bool``): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        if U_as_vertex:
            assert (
                self.num_v == bigraph.num_u
            ), "The number of vertices in the hypergraph and the number of vertices in set U of the bipartite graph are not equal."
        else:
            assert (
                self.num_v == bigraph.num_v
            ), "The number of vertices in the hypergraph and the number of vertices in set V of the bipartite graph are not equal."
        e_list = Hypergraph._e_list_from_bigraph(bigraph, U_as_vertex=U_as_vertex)
        self.add_hyperedges(e_list, group_name=group_name)

    def remove_hyperedges(
        self, e_list: Union[List[int], List[List[int]]], group_name: Optional[str] = None,
    ):
        r"""Remove the specified hyperedges from the hypergraph.
        用于从超图中移除指定的超边。这个方法可以移除单个或多个超边，并且可以选择性地从特定的超边组中移除，或者从所有超边组中移除。
        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``group_name`` (``str``, optional): Remove these hyperedges from the specified hyperedge group. If not specified, the function will
                remove those hyperedges from all hyperedge groups. Defaults to the ``None``.
        """
        assert (
            group_name is None or group_name in self.group_names
        ), "The specified group_name is not in existing hyperedge groups."
        # 如果指定了group_name，则验证该组名是否存在于现有的超边组中。如果没有指定group_name，则默认从所有超边组中移除超边。
        e_list = self._format_e_list(e_list)
        # 格式化超边列表：_format_e_list：确保每个超边都是一个排序后的元组列表，以便于后续的比较和操作。
        if group_name is None:
            for _idx in range(len(e_list)):
                len(e_list)
                # 返回e_list中的元素数量，即超边的数量。
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                for name in self.group_names:
                    # self.group_names是一个集合，包含了所有已定义的超边组的名称。
                    self._raw_groups[name].pop(e_code, None)
                    # _raw_groups[name]是Hypergraph类中的一个属性，用于存储超图中各个超边组的超边信息。具体来说，_raw_groups是一个字典，其中每个键是一个超边组的名称（字符串），每个值是一个字典，存储了该超边组中的超边及其相关信息。
                    # pop方法会移除字典中的键值对，并返回被移除的值。
        else:
            for _idx in range(len(e_list)):
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                self._raw_groups[group_name].pop(e_code, None)
        self._clear_cache(group_name)


    def remove_group(self, group_name: str):
        r"""Remove the specified hyperedge group from the hypergraph.
            从超图中移除指定的超边组
        Args:
            ``group_name`` (``str``): The name of the hyperedge group to remove.
        """
        self._raw_groups.pop(group_name, None)
        # pop方法用于从字典中移除指定的键值对.如果键存在，则返回被移除的值；如果键不存在，则返回None，这样可以避免抛出异常。
        self._clear_cache(group_name)

    def drop_hyperedges(self, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the hypergraph. This function will return a new hypergraph with non-dropped hyperedges.
            用于从超图中随机删除一定比例的超边，并返回一个新的超图。
            drop_rate (float): 超边的删除率，范围在 [0, 1] 之间。
            ord (str): 删除超边的顺序。目前只支持 'uniform'，表示均匀随机删除。
        Args:
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            # 创建一个新的字典_raw_groups，用于存储新超图的超边组。
            for name in self.group_names:
                _raw_groups[name] = {k: v for k, v in self._raw_groups[name].items() if random.random() > drop_rate}
                # 遍历每个超边组name，对于每个组中的每个超边k，以概率1 - drop_rate保留该超边。c
                # self._raw_groups[name].items()返回一个迭代器，生成字典中的键值对(k, v)。
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unkonwn drop order: {ord}.")
        return _hg

    def drop_hyperedges_of_group(self, group_name: str, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the specified hyperedge group. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group.
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                if name == group_name:
                    _raw_groups[name] = {
                        k: v for k, v in self._raw_groups[name].items() if random.random() > drop_rate
                    }
                else:
                    _raw_groups[name] = self._raw_groups[name]
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unkonwn drop order: {ord}.")
        return _hg

    # =====================================================================================
    # properties for representation
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices.
        """
        return super().v
    
    @property
    def v_weight(self) -> List[float]:
        r"""Return the list of vertex weights.
        """
        return self._v_weight

    @property
    def e_con(self) -> Optional[Union[List[int], List[List[int]]]]:
        r"""Return all hyperedges in the hypergraph.
        返回超图中所有超边。
        """
        if self.cache.get("e_con", None) is None:
            # 检查缓存中是否已经有结果，如果没有则进行计算。
            e_list = []
            for name in self.group_names:
                _e_con = self.e_con_of_group(name)
                e_list.extend(_e_con)
            self.cache["e_con"] = (e_list)
        return self.cache["e_con"]

    def e_con_of_group(self, group_name: str) -> Optional[Union[List[int], List[List[int]]]]:
        r"""Return all hyperedges of the specified hyperedge group.
            返回超边组中所有超边
        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("e_con", None) is None:
            e_list = [e_code[0] for e_code in self._raw_groups[group_name].keys()]
            self.group_cache[group_name]["e_con"] = (e_list)
        return self.group_cache[group_name]["e_con"]

    @property
    def e_list(self) -> Optional[Union[List[int], List[List[int]]]]:
        r"""Return the number of vertices in the hypergraph.
        """
        return self.e_list

    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the hypergraph.
        返回超图中所有超边及其权重。
        """
        if self.cache.get("e", None) is None:
            # 检查缓存中是否已经有结果，如果没有则进行计算。
            e_list, e_weight = [], []
            for name in self.group_names:
                _e = self.e_of_group(name)
                e_list.extend(_e[0])
                e_weight.extend(_e[1])
            self.cache["e"] = (e_list, e_weight)
        return self.cache["e"]

    def e_of_group(self, group_name: str) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights of the specified hyperedge group.
            返回超边组中所有超边及其权重
        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("e", None) is None:
            e_list = [e_code[0] for e_code in self._raw_groups[group_name].keys()]
            e_weight = [e_content["w_e"] for e_content in self._raw_groups[group_name].values()]
            self.group_cache[group_name]["e"] = (e_list, e_weight)
        return self.group_cache[group_name]["e"]

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the hypergraph.
        """
        return super().num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of hyperedges in the hypergraph.
        """
        return super().num_e

    def num_e_of_group(self, group_name: str) -> int:
        r"""Return the number of hyperedges of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        return super().num_e_of_group(group_name)

    @property
    def deg_v(self) -> List[int]:
        r"""Return the degree list of each vertex.
        返回超图中每个顶点的度
        """
        return self.D_v._values().cpu().view(-1).numpy().tolist()

    # self.D_v：假设D_v是一个稀疏矩阵，存储了每个顶点的度信息。
    # _values()：提取稀疏矩阵中的非零值。这些值表示每个顶点的度。
    # cpu()：将数据移动到CPU上，确保可以在CPU上进行操作。
    # view(-1)：将数据展平成一维张量。
    # numpy()：将张量转换为NumPy数组。
    # tolist()：将NumPy数组转换为Python列表。

    def deg_v_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each vertex of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_v_of_group(group_name)._values().cpu().view(-1).numpy().tolist()

    @property
    def deg_e(self) -> List[int]:
        r"""Return the degree list of each hyperedge.
        """
        return self.D_e._values().cpu().view(-1).numpy().tolist()

    def deg_e_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each hyperedge of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_e_of_group(group_name)._values().cpu().view(-1).numpy().tolist()

    def nbr_e(self, v_idx: int) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex.
            获取特定顶点的邻居超边(得到的结果是超边的索引值）
        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        return self.N_e(v_idx).cpu().numpy().tolist()

    def nbr_e_of_group(self, v_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex of the specified hyperedge group.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_e_of_group(v_idx, group_name).cpu().numpy().tolist()

    def nbr_v(self, e_idx: int) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge.

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        return self.N_v(e_idx).cpu().numpy().tolist()

    def nbr_v_of_group(self, e_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge of the specified hyperedge group.

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_v_of_group(e_idx, group_name).cpu().numpy().tolist()

    @property
    def num_groups(self) -> int:
        r"""Return the number of hyperedge groups in the hypergraph.
        返回超图中不同超边组的数量
        """
        return super().num_groups

    @property
    def group_names(self) -> List[str]:
        r"""Return the names of all hyperedge groups in the hypergraph.
        """
        return super().group_names

    # =====================================================================================
    # properties for deep learning
    @property
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in the hypergraph including
            返回列表的方法，该列表包含了超图中可用于深度学习的各种变量名称。
        Sparse Matrices:

        .. math::
            \mathbf{H}, \mathbf{H}^\top, \mathcal{L}_{sym}, \mathcal{L}_{rw} \mathcal{L}_{HGNN},

        Sparse Diagnal Matrices:

        .. math::
            \mathbf{W}_v, \mathbf{W}_e, \mathbf{D}_v, \mathbf{D}_v^{-1}, \mathbf{D}_v^{-\frac{1}{2}}, \mathbf{D}_e, \mathbf{D}_e^{-1},

        Vectors:

        .. math::
            \overrightarrow{v2e}_{src}, \overrightarrow{v2e}_{dst}, \overrightarrow{v2e}_{weight},\\
            \overrightarrow{e2v}_{src}, \overrightarrow{e2v}_{dst}, \overrightarrow{e2v}_{weight}
        稀疏矩阵：
H: 超图的邻接矩阵，表示顶点到超边的关联关系。
H_T: H 的转置矩阵，表示超边到顶点的关联关系。
L_sym: 对称拉普拉斯矩阵。
L_rw: 随机游走拉普拉斯矩阵。
L_HGNN: HGNN（HyperGraph Neural Network）的拉普拉斯矩阵。

稀疏对角矩阵：
W_v: 顶点权重矩阵。
W_e: 超边权重矩阵。
D_v: 顶点度矩阵。
D_v_neg_1: 顶点度矩阵的逆矩阵。
D_v_neg_1_2: 顶点度矩阵的负半次幂矩阵。
D_e: 超边度矩阵。
D_e_neg_1: 超边度矩阵的逆矩阵。

v2e_src: 从顶点到超边的源顶点索引。
v2e_dst: 从顶点到超边的目标超边索引。
v2e_weight: 从顶点到超边的权重。
e2v_src: 从超边到顶点的源超边索引。
e2v_dst: 从超边到顶点的目标顶点索引。
e2v_weight: 从超边到顶点的权重
        """
        return [
            "H",
            "H_T",
            "L_sym",
            "L_rw",
            "L_HGNN",
            "W_v",
            "W_e",
            "D_v",
            "D_v_neg_1",
            "D_v_neg_1_2",
            "D_e",
            "D_e_neg_1",
            "v2e_src",
            "v2e_dst",
            "v2e_weight" "e2v_src",
            "e2v_dst",
            "e2v_weight",
        ]

    @property
    def v2e_src(self) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the hypergraph.
        返回的是超图中从顶点指向超边的连接的源顶点索引向量。
        """
        return self.H_T._indices()[1].clone()

    def v2e_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[1].clone()

    @property
    def v2e_dst(self) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the hypergraph.
        返回的是超图中从顶点指向超边的连接的目标超边索引向量
        """
        return self.H_T._indices()[0].clone()

    def v2e_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[0].clone()

    @property
    def v2e_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._values().clone()

    def v2e_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._values().clone()

    @property
    def e2v_src(self) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[1].clone()

    def e2v_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[1].clone()

    @property
    def e2v_dst(self) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[0].clone()

    def e2v_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[0].clone()

    @property
    def e2v_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._values().clone()

    def e2v_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._values().clone()

    def new_H(sparse_matrix):
        # 这个函数接收一个稀疏矩阵作为参数，并直接返回它
        return sparse_matrix

    @property
    def H(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("H") is None:
            self.cache["H"] = self.H_v2e.to(self.device)
        return self.cache["H"]

    def H_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H") is None:
            self.group_cache[group_name]["H"] = self.H_v2e_of_group(group_name)
        return self.group_cache[group_name]["H"]

    @property
    def H_T(self) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("H_T") is None:
            self.cache["H_T"] = self.H.t().to(self.device)
        return self.cache["H_T"]

    def H_T_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_T") is None:
            self.group_cache[group_name]["H_T"] = self.H_of_group(group_name).t()
        return self.group_cache[group_name]["H_T"]
    
    @property
    def W_v(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_v` of vertices with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("W_v") is None:
            _tmp = torch.Tensor(self.v_weight)
            # 转换为一个PyTorch张量
            _num_v = _tmp.size(0)
            self.cache["W_v"] = torch.sparse_coo_tensor(
                # 创建一个稀疏COO张量
                torch.arange(0, _num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                device=self.device,
            ).coalesce()
            torch.arange(0, _num_v, device=self.device)
            # 生成从0到_num_v - 1的一维张量
            # .view(1, -1)将张量reshape成形状为(1, _num_v)的二维张量。
            # .repeat(2, 1)将张量在第一个维度上重复两次，得到形状为(2, _num_v)的索引张量。
            # _tmp是非零元素的值，即顶点的权重
            # 指定张量的形状为 _num_vx_num_v
            # .coalesce()确保稀疏张量的索引是唯一的，合并重复的索引并求和相应的值。
            torch.Size([_num_v, _num_v])
        return self.cache["W_v"]

    @property
    def W_e(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("W_e") is None:
            _tmp = [self.W_e_of_group(name)._values().clone() for name in self.group_names]
            # 使用列表推导式获取每个超边组的权重向量，并将其克隆到一个新的张量中。
            _tmp = torch.cat(_tmp, dim=0).view(-1)
            # 将所有超边组的权重向量合并成一个一维张量。
            # torch.cat(_tmp, dim=0)沿着第一个维度（行）连接所有张量。
            # .view(-1)将结果张量转换为一维张量。
            _num_e = _tmp.size(0)
            self.cache["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.cache["W_e"]

    def W_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("W_e") is None:
            _tmp = self._fetch_W_of_group(group_name).view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["W_e"]

    @property
    def D_v(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v") is None:
            _tmp = [self.D_v_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.vstack(_tmp).sum(dim=0).view(-1)
            # torch.vstack(_tmp)将所有张量垂直堆叠成一个二维张量。
            # .sum(dim=0)沿着第一个维度（行）求和，得到每个顶点的总度。
            self.cache["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v"]

    def D_v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v") is None:
            H = self.H_of_group(group_name).clone()
            w_e = self.W_e_of_group(group_name)._values().clone()
            val = w_e[H._indices()[1]] * H._values().to(device="cuda")
            H_ = torch.sparse_coo_tensor(H._indices(), val, size=H.shape, device=self.device).coalesce()
            _tmp = torch.sparse.sum(H_, dim=1).to_dense().clone().view(-1)
            _num_v = _tmp.size(0)
            self.group_cache[group_name]["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_v"]

    @property
    def D_v_neg_1(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -1
            # 计算对角线元素的倒数。
            _val[torch.isinf(_val)] = 0
            # 将计算结果中的无穷大值替换为0。
            self.cache["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1"]

    def D_v_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1"]

    @property
    def D_v_neg_1_2(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1_2") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1_2"]

    def D_v_neg_1_2_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1_2") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1_2"]

    @property
    def D_e(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e") is None:
            _tmp = [self.D_e_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.cat(_tmp, dim=0).view(-1)
            # 将所有超边组的度拼接成一个一维张量。
            _num_e = _tmp.size(0)
            # 返回_tmp张量在第一个维度（即行）上的大小，也就是超边的数量。
            self.cache["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.cache["D_e"]

    def D_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e") is None:
            _tmp = torch.sparse.sum(self.H_T_of_group(group_name), dim=1).to_dense().clone().view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_e"]

    @property
    def D_e_neg_1(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e_neg_1") is None:
            _mat = self.D_e.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            torch.isinf(_val)
            # 返回一个布尔张量，表示哪些元素是无穷大
            self.cache["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_e_neg_1"]

    def D_e_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e_neg_1") is None:
            _mat = self.D_e_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_e_neg_1"]

    def N_e(self, v_idx: int) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex with ``torch.Tensor`` format.
            返回指定顶点的邻居超边
        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        assert v_idx < self.num_v
        _tmp, e_bias = [], 0
        # _tmp = []初始化一个空列表，用于存储每个超边组的邻居超边索引
        # e_bias = 0初始化一个偏移量，用于记录当前超边组的起始索引
        for name in self.group_names:
            _tmp.append(self.N_e_of_group(v_idx, name) + e_bias)
            # e_bias的作用就是记录当前超边组的起始索引，以便在拼接所有超边组的邻居超边索引时，确保索引的唯一性和连续性。
            e_bias += self.num_e_of_group(name)
        return torch.cat(_tmp, dim=0)

    def N_e_of_group(self, v_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert v_idx < self.num_v
        e_indices = self.H_of_group(group_name)[v_idx]._indices()[0]
        # v_idx行，表示与顶点v_idx相连的超边。
        # ._indices()[0]提取稀疏张量中非零元素的列索引，这些索引即为邻居超边的索引。
        return e_indices.clone()

    def N_v(self, e_idx: int) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :attr:`num_e`).

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        assert e_idx < self.num_e
        for name in self.group_names:
            if e_idx < self.num_e_of_group(name):
                return self.N_v_of_group(e_idx, name)
            else:
                e_idx -= self.num_e_of_group(name)

    def N_v_of_group(self, e_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :func:`num_e_of_group`).

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert e_idx < self.num_e_of_group(group_name)
        v_indices = self.H_T_of_group(group_name)[e_idx]._indices()[0]
        return v_indices.clone()

    # =====================================================================================
    # spectral-based convolution/smoothing
    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        return super().smoothing(X, L, lamb)

    @property
    def L_sym(self) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_sym") is None:
            L_HGNN = self.L_HGNN.clone()
            self.cache["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), L_HGNN._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["L_sym"]

    def L_sym_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_sym") is None:
            L_HGNN = self.L_HGNN_of_group(group_name).clone()
            self.group_cache[group_name]["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), L_HGNN._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["L_sym"]

    @property
    def L_rw(self) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top
        """
        if self.cache.get("L_rw") is None:
            _tmp = self.D_v_neg_1.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T)
            self.cache["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _tmp._indices(),]),
                    torch.hstack([torch.ones(self.num_v, device=self.device), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.cache["L_rw"]

    def L_rw_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_rw") is None:
            _tmp = (
                self.D_v_neg_1_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name),)
                .mm(self.D_e_neg_1_of_group(group_name),)
                .mm(self.H_T_of_group(group_name),)
            )
            self.group_cache[group_name]["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _tmp._indices(),]),
                    torch.hstack([torch.ones(self.num_v, device=self.device), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.group_cache[group_name]["L_rw"]

    ## HGNN Laplacian smoothing
    @property
    def L_HGNN(self) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_HGNN") is None:
            # _tmp = self.D_v_neg_1_2.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T,).mm(self.D_v_neg_1_2)

            # 第一步
            tmp1 = self.D_v_neg_1_2.mm(self.H)

            # 第二步
            tmp2 = tmp1.mm(self.W_e)

            # 第三步
            tmp3 = tmp2.mm(self.D_e_neg_1)

            # 第四步
            tmp4 = tmp3.mm(self.H_T)

            # 第五步
            _tmp = tmp4.mm(self.D_v_neg_1_2)

            self.cache["L_HGNN"] = _tmp.coalesce()
        return self.cache["L_HGNN"]


    def L_HGNN_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_HGNN") is None:
            _tmp = (
                self.D_v_neg_1_2_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name))
                .mm(self.D_e_neg_1_of_group(group_name),)
                .mm(self.H_T_of_group(group_name),)
                .mm(self.D_v_neg_1_2_of_group(group_name),)
            )
            self.group_cache[group_name]["L_HGNN"] = _tmp.coalesce()
        return self.group_cache[group_name]["L_HGNN"]

    def smoothing_with_HGNN(self, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
    """
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN, drop_rate)
        else:
            L_HGNN = self.L_HGNN
        return L_HGNN.mm(X)

    def smoothing_with_HGNN_of_group(self, group_name: str, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN_of_group(group_name), drop_rate)
        else:
            L_HGNN = self.L_HGNN_of_group(group_name)
        return L_HGNN.mm(X)

    # =====================================================================================
    # spatial-based convolution/message-passing
    ## general message passing functions
    def v2e_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggretation step of ``vertices to hyperedges``.
        执行从顶点到超边的消息聚合
        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T, drop_rate)
            else:
                P = self.H_T
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight.shape[0]
            ), "The size of v2e_weight must be equal to the size of self.v2e_weight."
            P = torch.sparse_coo_tensor(self.H_T._indices(), v2e_weight, self.H_T.shape, device=self.device)
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X

    def v2e_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T_of_group(group_name), drop_rate)
            else:
                P = self.H_T_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight_of_group(group_name).shape[0]
            ), f"The size of v2e_weight must be equal to the size of self.v2e_weight_of_group('{group_name}')."
            P = torch.sparse_coo_tensor(
                self.H_T_of_group(group_name)._indices(),
                v2e_weight,
                self.H_T_of_group(group_name).shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X

    def v2e_update(self, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e, X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e, "The size of e_weight must be equal to the size of self.num_e."
            X = e_weight * X
        return X

    def v2e_update_of_group(self, group_name: str, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e_of_group(group_name), X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e_of_group(
                group_name
            ), f"The size of e_weight must be equal to the size of self.num_e_of_group('{group_name}')."
            X = e_weight * X
        return X

    def v2e(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.v2e_aggregation(X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update(X, e_weight)
        return X

    def v2e_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        X = self.v2e_aggregation_of_group(group_name, X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update_of_group(group_name, X, e_weight)
        return X

    def e2v_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggregation step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H, drop_rate)
            else:
                P = self.H
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight.shape[0]
            ), "The size of e2v_weight must be equal to the size of self.e2v_weight."
            P = torch.sparse_coo_tensor(self.H._indices(), e2v_weight, self.H.shape, device=self.device)
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X

    def e2v_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_of_group(group_name), drop_rate)
            else:
                P = self.H_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight_of_group(group_name).shape[0]
            ), f"The size of e2v_weight must be equal to the size of self.e2v_weight_of_group('{group_name}')."
            P = torch.sparse_coo_tensor(
                self.H_of_group(group_name)._indices(),
                e2v_weight,
                self.H_of_group(group_name).shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X

    def e2v_update(self, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        if self.device != X.device:
            self.to(X.device)
        return X

    def e2v_update_of_group(self, group_name: str, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            self.to(X.device)
        return X

    def e2v(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.e2v_aggregation(X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update(X)
        return X

    def e2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        X = self.e2v_aggregation_of_group(group_name, X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update_of_group(group_name, X)
        return X

    def v2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices``. The combination of ``v2e`` and ``e2v``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e`` and ``e2v``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e``. Default: ``None``.
        """
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e(X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v(X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)
        return X

    def v2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices`` in specified hyperedge group. The combination of ``v2e_of_group`` and ``e2v_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e_of_group`` and ``e2v_of_group``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v_of_group``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v_of_group``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e_of_group``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e_of_group``. Default: ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e_of_group(group_name, X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v_of_group(group_name, X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)
        return X

    def modify_H(self, i, j, device):
        """
        修改稀疏矩阵中的特定位置的值。

        Args:
            i (int): 行索引
            j (int): 列索引
            device (torch.device): 目标设备

        Returns:
            torch.Tensor: 修改后的稀疏矩阵
        """
        # 将稀疏矩阵转换为密集矩阵并移动到目标设备
        H_dense = self.H.to_dense().to(device)

        # 修改密集矩阵中的特定位置的值
        if H_dense[i, j].item() == 1:
            H_dense[i, j] = 0

        # 打印修改后的密集矩阵
        # print("Modified Dense Matrix:")
        # print(H_dense)

        # 将密集矩阵转换回稀疏矩阵并移动到目标设备
        H_sparse = H_dense.to_sparse_coo().to(device)

        # 打印修改后的稀疏矩阵
        # print("Modified Sparse Matrix:")
        # print(H_sparse)

        return H_sparse
