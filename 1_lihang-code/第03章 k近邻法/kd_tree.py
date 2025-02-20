from __future__ import annotations  # 启用新的类型提示解析
from typing import Optional, List

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import heapq


def distance(x: np.ndarray, y: np.ndarray, p: int = 2):
    return np.linalg.norm(x-y, p)


class HeapNode:
    def __init__(self, label: int, dist: int):
        self.label = label
        self.dist = dist

    def __lt__(self, other: HeapNode):
        return self.dist < other.dist

    def __repr__(self):
        return f"HeapNode({self.label}, {self.dist})"

    def __str__(self):
        return f"HeapNode({self.label}, {self.dist})"


class KdNode:
    def __init__(self, x: np.ndarray, y: int, left: Optional[KdNode] = None, right: Optional[KdNode] = None) -> None:
        self.x = x
        self.y = y
        self.left = left
        self.right = right

    def __repr__(self):
        return f"KdNode({self.y}, {self.x})"


class KdTree:
    @staticmethod
    def build_kdtree(data: np.ndarray, label: np.ndarray, depth: int = 0) -> KdNode:
        """
        Build a k-d tree from the given data and labels.

        Parameters
        ----------
        data : np.ndarray, shape [batch, n_feats]
            The data to build the k-d tree from.
        label : np.ndarray, shape [batch, ]
            The labels of the data.
        depth : int, optional
            The current depth of the tree. Defaults to 0.

        Returns
        -------
        KdNode
            The root of the k-d tree.
        """
        if (len(data) == 0):
            return None

        k = data.shape[1]
        axis = depth % k

        sorted_idx = np.argsort(data[:, axis])
        data = data[sorted_idx]
        label = label[sorted_idx]

        median_idx = len(data) // 2
        x, y = data[median_idx], label[median_idx]

        left = KdTree.build_kdtree(data[:median_idx], label[:median_idx], depth+1)
        right = KdTree.build_kdtree(data[median_idx+1:], label[median_idx+1:], depth+1)

        return KdNode(x, y, left, right)

    @staticmethod
    def print_tree(root: KdNode):
        """ 层序打印 """
        tmp = [root]
        while (len(tmp) > 0):
            node = tmp.pop(0)
            print(node)
            if node.left is not None:
                tmp.append(node.left)
            if node.right is not None:
                tmp.append(node.right)

    def __init__(self, data: np.ndarray, label: np.ndarray, k: int = 3, p: int = 2) -> None:
        self.root = KdTree.build_kdtree(data, label, 0)
        self.k = k
        self.p = p
        self.heap = []

    def find_nearst(self, node: KdNode, x: np.ndarray, depth: int = 0) -> None:
        if node is None:
            return

        dist = -distance(node.x, x, self.p)  # 处理根节点
        if (len(self.heap) < self.k):
            heapq.heappush(self.heap, HeapNode(node.y, dist))
        elif (self.heap[0].dist < -dist):
            heapq.heapreplace(self.heap, HeapNode(node.y, dist))

        k = x.shape[0]
        axis = depth % k

        if x[axis] < node.x[axis]:
            self.find_nearst(node.left, x, depth+1)

            # abs(x[axis] - node.x[axis]) 是搜索点x到当前超平面的距离
            # -self.heap[0].dist 是搜索点x到最近邻的距离
            # abs(x[axis] - node.x[axis]) < -self.heap[0].dist 表示与超平面相交
            # 可以对比 https://www.cnblogs.com/maxlpy/p/4297254.html 的图4进行理解
            if len(self.heap) < self.k or abs(x[axis] - node.x[axis]) < -self.heap[0].dist:
                self.find_nearst(node.right, x, depth+1)
        else:
            self.find_nearst(node.right, x, depth+1)
            if len(self.heap) < self.k or abs(x[axis] - node.x[axis]) < -self.heap[0].dist:
                self.find_nearst(node.left, x, depth+1)

    def predict(self, x: np.ndarray) -> int:
        self.find_nearst(self.root, x, 0)
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


def test_kdtree():
    data = np.array([[2, 3],
                     [5, 4],
                     [9, 6],
                     [4, 7],
                     [8, 1],
                     [7, 2]])
    label = np.array([-1, 1, -1, -1,  1,  1])

    print(data.shape)
    model = KdTree(data, label, k=1)
    # KdTree.print_tree(model.root)

    x = np.array([2, 4.5])
    print(x.shape)
    model.find_nearst(model.root, x, 0)
    print(model.heap)


if __name__ == "__main__":
    test_kdtree()
