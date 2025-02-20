from __future__ import annotations  # 启用新的类型提示解析
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import heapq


def get_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])

    return X, y


class KNNNode:
    """ 使用堆结构来存储KNN的数据
    """

    def __init__(self, label: int, dist: int):
        self.label = label
        self.dist = dist

    def __lt__(self, other: KNNNode):
        return self.dist < other.dist

    def __repr__(self):
        return f"KNNNode({self.label}, {self.dist})"

    def __str__(self):
        return f"KNNNode({self.label}, {self.dist})"


class KNN:
    def __init__(self, k, p=2):
        self.heap: list[KNNNode] = []
        self.k = k
        self.p = p

    def train(self, data, label):
        self.data = data
        self.label = label

    def predict(self, x):
        for (x_train, label) in zip(self.data, self.label):
            dist = -np.linalg.norm(x_train-x, ord=self.p)
            if (len(self.heap) < self.k):
                heapq.heappush(self.heap, KNNNode(label, dist))
            else:
                max_dist = self.heap[0]
                if (dist > max_dist.dist):
                    heapq.heappop(self.heap)
                    heapq.heappush(self.heap, KNNNode(label, dist))

        pos = 0
        neg = 0
        for node in self.heap:
            if node.label == 1:
                pos += 1
            else:
                neg += 1

        if pos >= neg:
            return 1
        else:
            return -1


class KdNode:
    def __init__(self, point, label, left: Optional[KdNode] = None, right: Optional[KdNode] = None) -> None:
        self.point = point
        self.label = label
        self.left = left
        self.right = right


class KdTree:
    @staticmethod
    def create_node(data: np.ndarray, depth: int) -> KdNode:
        if (len(data) == 0):
            return None

        k = data.shape[1]   # data.shape: [batch, n_feats]
        axis = depth % k

        sorted_indices = np.argsort(data[:, axis])  # 对维度进行排序
        data = data[sorted_indices]
        median_idx = len(data) // 2
        median_point = data[median_idx]

        left = KdTree.create_node(data[:median_idx], depth+1)
        right = KdTree.create_node(data[median_idx+1:], depth+1)

        return KdNode(median_point, left, right)

    @staticmethod
    def print_tree(root: KdNode, depth=0):
        if root is not None:
            print(" " * depth*2, root.point)
            KdTree.print_tree(root.left, depth+1)
            KdTree.print_tree(root.right, depth+1)

    def __init__(self, data: np.ndarray, k: int = 3, p: int = 2):
        self.root = KdTree.create_node(data, 0)
        self.k = k
        self.p = p
        self.heap = []

    def find_nearst(self, root: KdNode, target: np.array, depth: int) -> KdNode:
        self.heap = []
        dist = -np.linalg.norm(root.point-target, ord=self.p)
        if (len(self.heap) < self.k):
            heapq.heappush(self.heap, )


def test_kdtree():
    data = np.random.randn(10, 2)
    kd_tree = KdTree(data)
    KdTree.print_tree(kd_tree.root)


def test_KNNNode():
    heap = []
    heapq.heappush(heap, KNNNode(-1, 5))
    heapq.heappush(heap, KNNNode(-1, 1))
    heapq.heappush(heap, KNNNode(-1, 2))
    heapq.heappush(heap, KNNNode(-1, 6))
    heapq.heappush(heap, KNNNode(-1, 2))

    min_node = heap[0]
    print(min_node)
    print(heap)


def test_knn():
    data, label = get_dataset()
    model = KNN(k=3, p=2)
    model.train(data, label)

    test_x = np.array([5.5, 3.3])
    y_pred = model.predict(test_x)

    plt.figure(figsize=(8, 6))
    for i in np.unique(label):
        plt.scatter(data[label == i, 0], data[label == i, 1], label=i)

    plt.scatter(test_x[0], test_x[1], label=y_pred, marker='*')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_KNNNode()
    # test_knn()
    test_kdtree()
    pass
