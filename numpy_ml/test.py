from nn.optimizer import (
    OptimizerBase,
    SGD,
)
from nn.layers import (
    LayerBase,
    Linear,
)
from nn.activations import (
    ActivationBase,
    Sigmoid,
    ReLU,
)
from nn.initializers import ActivationInitializer, OptimizerInitializer
from nn.utils import miniBatch, sigmoid
from copy import deepcopy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 1. 载入Iris数据集，并将其转换为二分类问题


def load_data():
    X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, flip_y=0, class_sep=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def plot_dataset(X_train, y_train):
    plt.figure(figsize=(8, 6))

# 画出类别为0的点
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0', alpha=0.6)

    # 画出类别为1的点
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1', alpha=0.6)

    # 添加标签
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Plot of the Binary Classification Dataset')

    # 添加图例
    plt.legend()

    # 显示图像
    plt.show()


class FCModel:
    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int, optimizer: OptimizerBase):
        self.model_list = []
        self.linear1 = Linear(input_size, hidden_size1, optimizer=deepcopy(optimizer))  # 第一层全连接
        self.linear2 = Linear(hidden_size1, hidden_size2, optimizer=deepcopy(optimizer))  # 第二层全连接
        self.linear3 = Linear(hidden_size2, output_size, optimizer=deepcopy(optimizer))  # 输出层

        self.act = ReLU()
        self.model_list.extend([self.linear1, self.act, self.linear2, self.act, self.linear3])

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

    def unfreeze(self):
        for model in self.model_list:
            if isinstance(model, LayerBase):
                model.unfreeze()

    def freeze(self):
        for model in self.model_list:
            if isinstance(model, LayerBase):
                model.freeze()

    def forward(self, X):
        for model in self.model_list:
            X = model(X)

        return X

    def backward(self, dldy):
        for model in reversed(self.model_list):
            if isinstance(model, ActivationBase):
                dldy = model.backward(dldy)
            elif isinstance(model, LayerBase):
                dldy = model.backward(dldy, retain_grads=True)
            else:
                raise ValueError(f"Unknown model: {model}")

    def update(self):
        for model in self.model_list:
            if isinstance(model, LayerBase):
                model.update()


def sigmoid_cross_entropy(predictions, labels):
    """
    计算sigmoid交叉熵损失函数
    """
    epsilon = 1e-5  # 避免log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))


def train(model: FCModel, X_train, y_train, epochs=10, batch_size=32, shuffle=True):
    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        # 打乱数据顺序
        if shuffle:
            indices = np.random.permutation(num_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

        epoch_loss = 0
        num_batches = int(np.ceil(num_samples / batch_size))

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)

            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # 前向传播
            out = model.forward(X_batch)
            y_pred = sigmoid(out)

            # 计算损失
            loss = sigmoid_cross_entropy(y_pred, y_batch)
            epoch_loss += loss

            # 反向传播
            dldy = (y_pred - y_batch) / y_batch.shape[0]  # sigmoid cross entropy 的导数
            model.backward(dldy)

            # 更新参数
            model.update()

        # 平均损失
        epoch_loss /= num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


def test_fc_model():
    # 载入数据
    X_train, X_test, y_train, y_test = load_data()
    plot_dataset(X_train, y_train)

    # 将标签转换为一维二分类（0, 1）
    y_train = y_train.reshape(-1, 1)
    # y_test = y_test.reshape(-1, 1)

    # 选择优化器（SGD）
    optimizer = OptimizerInitializer(init="SGD(lr=0.1, clip_norm=2.0)")()

    # 初始化模型
    model = FCModel(input_size=X_train.shape[1], hidden_size1=10, hidden_size2=4, output_size=1, optimizer=optimizer)

    # 训练模型
    train(model, X_train, y_train, epochs=100)

    # 简单的预测
    out = model.forward(X_train)
    predictions = sigmoid(out)

    # 计算准确率
    auc = roc_auc_score(y_train, predictions)
    print(f"Test Accuracy: {auc:.4f}")

    # 断言预测的准确率大于一定值（根据需要可以调整）
    assert auc > 0.85, f"auc {auc} is below the threshold"

    print("Test passed!")


if __name__ == "__main__":
    # 运行测试
    test_fc_model()
