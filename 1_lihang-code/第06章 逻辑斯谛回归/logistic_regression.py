import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]

    data = np.array(df.iloc[:10000, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else 0 for i in y])

    return X, y


def sigmoid(x: np.ndarray):
    eps = 1e-8
    return 1.0 / (1 + np.exp(-x) + eps)


class LogisticResgression:
    def __init__(self, in_dim: int, out_dim: int, lr: float = 0.1):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self._init_param()

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def _init_param(self):
        self.w = np.random.randn(self.in_dim, self.out_dim)
        self.b = np.zeros((1, self.out_dim))

    def forward(self, x: np.ndarray):
        return x @ self.w + self.b

    def train(self, X: np.ndarray, y: np.ndarray):
        for _ in range(1000):  # 迭代次数增加，通常需要更多的迭代
            # 使用批量数据进行更新
            y_pred = self.predict(X)
            loss = self.compute_loss(y, y_pred)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            if _ % 100 == 0:  # 打印每100步的损失
                print(f"Loss at step {_}: {loss}")

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 防止log(0)错误
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def compute_gradients(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        error = y_pred - y
        error = np.expand_dims(error, axis=1)
        grad_w = X.T @ error / X.shape[0]  # 计算梯度
        grad_b = np.sum(error) / X.shape[0]
        return grad_w, grad_b

    def predict(self, X: np.ndarray):
        output = self.forward(X)
        y_pred = sigmoid(output)
        return y_pred.squeeze(axis=1)


def main():
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = LogisticResgression(X_train.shape[1], 1)
    model.train(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_train_pred = (y_train_pred >= 0.5).astype(int)
    train_auc = accuracy_score(y_train, y_train_pred)
    print(f"Train accuracy: {train_auc}")

    y_pred = model.predict(X_test)
    y_pred = (y_pred >= 0.5).astype(int)
    test_auc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_auc}")


if __name__ == "__main__":
    main()
