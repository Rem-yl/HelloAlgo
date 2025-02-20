import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
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


def plot_dataset(data, label):
    plt.figure(figsize=(8, 6))
    for i in np.unique(label):
        plt.scatter(data[label == i, 0], data[label == i, 1], label=i)
    plt.legend()
    plt.show()


class Perceptron:
    def __init__(self, n_feats, lr=0.001):
        self.n_feats = n_feats
        self.lr = lr
        self.w = np.random.randn(1, n_feats)
        self.b = np.zeros((1,))

    def __call__(self, x):
        self.forward(x)

    def forward(self, x):
        x = self.w @ x + self.b
        return x

    def train(self, data, label):
        is_wrong = False
        epoch = 0
        while not is_wrong:
            wrong_count = 0
            train_loss = 0.0

            for (x, y) in zip(data, label):
                o = self.forward(x)
                if (-y * o > 0):    # 误分类点
                    self.w += self.lr * y * x.T
                    self.b += self.lr * y
                    wrong_count += 1

            for (x, y) in zip(data, label):
                o = self.forward(x)
                if (-y * o > 0):
                    train_loss += y*o

            train_loss /= np.linalg.norm(self.w)

            if (epoch % 10 == 0):
                print(f"Epoch: {epoch}, Train Loss: {train_loss[0]}")

            epoch += 1
            if wrong_count == 0:
                is_wrong = True

        print(f"Final Train loss: {train_loss}")
        print(f"Train Epoch: {epoch}")

    def predict(self, x):
        x = self.forward(x)
        output = np.sign(x)

        return output


if __name__ == "__main__":
    data, label = get_dataset()
    # plot_dataset(data, label)
    model = Perceptron(n_feats=2)

    model.train(data, label)

    # 绘制最终更新完成后的曲线
    plt.figure(figsize=(8, 6))
    for i in np.unique(label):
        plt.scatter(data[label == i, 0], data[label == i, 1], label=i)
    plt.legend()

    x_points = np.linspace(4, 7, 10)
    y_ = -(model.w[0, 0] * x_points + model.b) / model.w[0, 1]
    plt.plot(x_points, y_)
    plt.show()
