import numpy as np
import matplotlib.pyplot as plt
from gmm import GMM
from sklearn.metrics import accuracy_score


def generate_gmm_data(n_samples=300, means=[[-2, 2], [2, -2], [7, 7]], covs=[[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[2, 0], [0, 2]]], weights=[0.33, 0.33, 0.34]):
    """
    生成 2D 高斯混合数据（三类样本），修正其中一类分布
    :param n_samples: 总样本数
    :param means: 每个高斯分布的均值
    :param covs: 每个高斯分布的协方差矩阵
    :param weights: 每个高斯分布的权重
    :return: 生成的数据点 (X) 和对应的类别标签 (y)
    """
    n_clusters = len(means)
    assert len(covs) == n_clusters and len(weights) == n_clusters, "means, covs, weights 需要长度一致"

    X = []
    y = []

    for i in range(n_clusters):
        num_samples = int(n_samples * weights[i])  # 按照权重分配样本
        points = np.random.multivariate_normal(means[i], covs[i], num_samples)
        X.append(points)
        y.append(np.full(num_samples, i))  # 记录类别标签

    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


def plot_data():
    # 生成三类数据
    X, y = generate_gmm_data()

    # 绘制数据
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Generated GMM Data (Three Classes with Adjusted Class)")
    plt.show()


def main():
    pass


if __name__ == "__main__":
    X, y = generate_gmm_data(n_samples=10000)
    model = GMM()
    model.fit(X, verbose=True)
    y_pred = model.predict(X, soft_labels=False)
    acc = accuracy_score(y, y_pred)
    print(y)
    print(y_pred)
    print(f"acc: {acc}")
