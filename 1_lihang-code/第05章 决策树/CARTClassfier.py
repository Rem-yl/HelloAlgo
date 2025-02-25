from __future__ import annotations  # 启用新的类型提示解析
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, feature: str = None, thr: float = None, left: Node = None, right: Node = None, value: float = None):
        self.feature = feature
        self.thr = thr
        self.value = value
        self.left: Node = left
        self.right: Node = right

    def is_leaf(self):
        return self.value is not None


def gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)  # 计算每个类别的概率
    return 1.0 - np.sum(probs ** 2)  # 计算基尼指数


class CARTClassfier:
    def __init__(self, min_sample: int = 2, max_depth: int = 3, alpha: float = 5):
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.root: Node = None
        self.alpha = alpha

    def best_split(self, x: np.ndarray, y: np.ndarray):
        """ 返回最佳分割的特征和特征值 """
        best_gini = np.inf
        best_feature, best_thr = None, None
        best_left, best_right = None, None

        n_samples, n_feats = x.shape
        for feat in range(n_feats):
            thrs = np.unique(x[:, feat])    # 对特征值进行去重操作
            for thr in thrs:
                left_mask = x[:, feat] <= thr
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gini = gini(y[left_mask])
                right_gini = gini(y[right_mask])
                final_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / n_samples

                if final_gini < best_gini:
                    best_gini = final_gini
                    best_feature, best_thr = feat, thr
                    best_left, best_right = left_mask, right_mask

        return (best_feature, best_thr, best_left, best_right)

    def build_tree(self, x: np.ndarray, y: np.ndarray, depth=0):
        if len(set(y)) == 1 or len(y) < self.min_sample or (self.max_depth and depth >= self.max_depth):
            return Node(value=np.argmax(np.bincount(y)))    # 返回多数类别

        feature, thr, left_mask, right_mask = self.best_split(x, y)
        if feature == None:
            return Node(value=np.argmax(np.bincount(y)))

        left_tree = self.build_tree(x[left_mask], y[left_mask], depth+1)
        right_tree = self.build_tree(x[right_mask], y[right_mask], depth+1)

        return Node(feature=feature, thr=thr, left=left_tree, right=right_tree)

    def train(self, x: np.ndarray, y: np.ndarray):
        self.root = self.build_tree(x, y)
        # self.prune(self.root, x, y)

    def predict(self, x: np.ndarray):
        node = self.root

        while not node.is_leaf():
            if x[node.feature] <= node.thr:
                node = node.left
            else:
                node = node.right

        return node.value

    def get_cost(self, node: Node, x: np.ndarray, y: np.ndarray):
        if node.is_leaf():
            return gini(y) + self.alpha     # 见《统计学习方法第二版》公式(5.26)

        left_mask = x[:, node.feature] <= node.thr
        right_mask = ~left_mask

        left_cost = self.get_cost(node.left, x[left_mask], y[left_mask])
        right_cost = self.get_cost(node.right, x[right_mask], y[right_mask])

        return left_cost + right_cost

    def prune(self, node: Node, X, y):
        """后剪枝"""
        if node.is_leaf():
            return

        left_mask = X[:, node.feature] <= node.thr
        right_mask = ~left_mask

        self.prune(node.left, X[left_mask], y[left_mask])
        self.prune(node.right, X[right_mask], y[right_mask])

        # 计算剪枝前后的损失
        before_pruning_cost = self.get_cost(node, X, y)
        leaf_value = np.argmax(np.bincount(y))
        after_pruning_cost = gini(y) + self.alpha

        # 如果剪枝后损失变小，则剪枝
        if after_pruning_cost <= before_pruning_cost:
            node.left = None
            node.right = None
            node.value = leaf_value


def main():
    # 测试代码
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target  # 只取前两个特征
    y = np.where(y == 2, 1, y)  # 只分类 0 和 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = CARTClassfier(max_depth=3)
    model.train(X_train, y_train)
    y = []
    for x in X_train:
        y.append(model.predict(x))
    train_auc = accuracy_score(y_train, y)

    y_pred = []
    for x in X_test:
        y_pred.append(model.predict(x))

    y_pred = np.array(y_pred)
    test_auc = accuracy_score(y_test, y_pred)

    print(f"train auc: {train_auc:.4f}, test auc: {test_auc:.4f}")


if __name__ == "__main__":
    main()
