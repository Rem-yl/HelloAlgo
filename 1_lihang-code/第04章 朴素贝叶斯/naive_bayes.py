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

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])

    return X, y


class GassianNB:
    def __init__(self):
        self.classes = None
        self.class_prior = {}   # 计算训练集中类别的先验概率 P(y), 使用极大似然估计
        self.class_mean = {}
        self.class_var = {}

    def train(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.class_prior[c] = len(X_c) / X.shape[0]
            self.class_mean[c] = np.mean(X_c, axis=0)   # 对高斯分布均值的极大似然估计
            self.class_var[c] = np.var(X_c, axis=0)  # 对方差的极大似然估计

    @staticmethod
    def gaussian_pdf(x, mean, var):
        # 高斯分布的公式
        eps = 1e-9
        coff = 1 / np.sqrt(2 * np.pi * var + eps)
        res = np.exp(-(x-mean)**2 / (2*var + eps))

        return coff * res

    def predict(self, x: np.ndarray):
        y_pred = {}

        for c in self.classes:
            p_yc = np.log(self.class_prior[c])
            likelihood = np.sum(np.log(GassianNB.gaussian_pdf(
                x, self.class_mean[c], self.class_var[c])))   # 将每个特征的likelihood加起来
            y_pred[c] = p_yc + likelihood

        return max(y_pred, key=y_pred.get)


if __name__ == "__main__":
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = GassianNB()
    model.train(X_train, y_train)

    y_pred = []
    for data in X_test:
        y_pred.append(model.predict(data))

    auc = accuracy_score(y_test, np.array(y_pred))
    print(f"AUC: {auc}")
