from __future__ import annotations  # 启用新的类型提示解析
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


def mse(y: np.ndarray):
    return np.var(y) if len(y) > 0 else 0.0


class Node:
    def __init__(self, feat_dim: int = None, thr: float = None, value: float = None, left: Node = None, right: Node = None):
        self.feat_dim = feat_dim
        self.thr = thr
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.value is not None


class CARTRegresion:
    def __init__(self, min_sample_split: int = 2, min_mse_decrease: float = 0.0, max_depth: int = 3):
        self.min_sample_split = min_sample_split
        self.min_mse_decrease = min_mse_decrease
        self.max_split = max_depth
        self.root = None

    def best_split(self, x: np.ndarray, y: np.ndarray):
        best_mse = np.inf
        best_feat_dim, best_thr = None, None
        best_left, best_right = None, None

        n_sample, n_dim = x.shape

        for feat_dim in range(n_dim):
            thrs = np.unique(x[:, feat_dim])

            for thr in thrs:
                left_mask = x[:, feat_dim] <= thr
                right_mask = ~left_mask

                # 决策树不能划分空节点
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mse = mse(y[left_mask])
                right_mse = mse(y[right_mask])

                final_mse = (len(left_mask) * left_mse + len(right_mask) * right_mse) / n_sample

                if best_mse - final_mse >= self.min_mse_decrease:
                    best_mse = final_mse
                    best_feat_dim, best_thr = feat_dim, thr
                    best_left, best_right = left_mask, right_mask

        return (best_feat_dim, best_thr, best_left, best_right)

    def build_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0):
        if len(set(y)) == 1 or len(y) < self.min_sample_split or (self.max_split and depth >= self.max_split):
            return Node(value=np.mean(y))

        feat_dim, thr, left_mask, right_mask = self.best_split(x, y)
        if feat_dim is None:
            return Node(value=np.mean(y))

        left_tree = self.build_tree(x[left_mask], y[left_mask])
        right_tree = self.build_tree(x[right_mask], y[right_mask])

        return Node(feat_dim=feat_dim, thr=thr, left=left_tree, right=right_tree)

    def train(self, x: np.ndarray, y: np.ndarray):
        self.root = self.build_tree(x, y)

    def predict(self, X: np.ndarray):
        y_pred = []
        for x in X:
            node = self.root

            while not node.is_leaf():
                if x[node.feat_dim] <= node.thr:
                    node = node.left
                else:
                    node = node.right

            y_pred.append(node.value)

        return np.array(y_pred)


if __name__ == "__main__":
    data = fetch_california_housing()
    X, y = data.data[:1000], data.target[:1000]

    # 只选取两个特征，方便可视化
    X = X[:, :2]
    y = y / np.max(y)  # 归一化，方便训练

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练CART回归树
    model = CARTRegresion(max_depth=5)
    model.train(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_loss = mean_squared_error(y_train, y_train_pred)
    test_loss = mean_squared_error(y_test, y_test_pred)

    print(f"训练集损失 (MSE): {train_loss:.4f}")
    print(f"测试集损失 (MSE): {test_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_train, y_train_pred, alpha=0.5, label="Predicted vs. Actual")
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.plot([0, 1], [0, 1], "r--")  # 参考线
    plt.legend()
    plt.title("CART Regression Tree Performance")
    plt.show()
