"""
使用合页损失+梯度下降的线性SVM模型
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])

    return X, y


class SVM:
    # SVM对 lr和l 取值敏感
    def __init__(self, in_dim: int, out_dim: int = 1, lr: float = 0.001, l: float = 0.0001):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.l = l
        self.w = np.random.randn(in_dim, out_dim) * 0.01  # 初始化小值
        self.b = np.zeros((1, out_dim))

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        return np.sign(x @ self.w + self.b)

    def train(self, X: np.ndarray, y: np.ndarray, epochs=2000):
        for _ in range(epochs):
            indices = np.random.permutation(X.shape[0])  # 打乱数据
            for i in indices:
                xi = X[i].reshape(1, -1)
                yi = y[i]
                margin = yi * (xi @ self.w + self.b)

                if margin < 1:
                    # 误分类
                    self.w -= self.lr * (-yi * xi.T + 2 * self.l * self.w)
                    self.b -= self.lr * yi * 0.0001  # 缩小偏置更新
                else:
                    # 正确分类
                    self.w -= self.lr * (2 * self.l * self.w)


if __name__ == "__main__":
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 读标准化敏感
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVM(in_dim=X_train.shape[1])
    model.train(X_train, y_train)

    y_train_pred = model(X_train).flatten()
    y_train_pred = np.where(y_train_pred > 0, 1, -1)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = model(X_test).flatten()
    y_test_pred = np.where(y_test_pred > 0, 1, -1)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print("Final Weights:", model.w.flatten())
    print("Final Bias:", model.b)

    # 画决策边界
    def plot_decision_boundary(model, X, y):
        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}")

        x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        x2_vals = np.squeeze(-(model.w[0, 0] * x1_vals + model.b) / model.w[1, 0])

        plt.plot(x1_vals, x2_vals, "r--", label="Decision Boundary")
        plt.legend()
        plt.show()

    plot_decision_boundary(model, X_train, y_train)
