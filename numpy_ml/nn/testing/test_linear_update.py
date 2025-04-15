import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from copy import deepcopy
import numpy as np
from nn.layers import LayerBase, Linear
from nn.activations import ReLU
from nn.optimizer import SGD
from nn.utils import sigmoid

# 数据加载函数


def load_data():
    X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, flip_y=0, class_sep=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, X_test, y_train, y_test


class Model:
    def __init__(self, input_dim):
        self.model_list = []
        self.linear1 = Linear(input_dim, 10)
        self.relu = ReLU()
        self.linear2 = Linear(10, 1)

        self.model_list.extend([self.linear1, self.relu, self.linear2])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.model_list:
            x = layer(x)

        return x

    def backward(self, grad_in):
        for layer in reversed(self.model_list):
            grad_in = layer.backward(grad_in)

        return grad_in

    def parameters(self):
        params = []
        for layer in self.model_list:
            if isinstance(layer, LayerBase):
                params.append(layer.parameters)

        return params


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def bce_loss_with_logit(logits: np.ndarray, y: np.ndarray):
    y_pred = sigmoid(logits)
    epsilon = 1e-10  # 避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def test_linear_model():
    X_train, X_test, y_train, y_test = load_data()
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    y_test_torch = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    torch_model = TwoLayerNet(input_dim=2)
    torch_loss_fn = nn.BCEWithLogitsLoss()  # 用这个代替 Sigmoid + BCELoss
    torch_optimizer = optim.SGD(torch_model.parameters(), lr=0.1)

    my_model = Model(input_dim=2)
    with torch.no_grad():
        my_model.linear1.parameters["W"].data = deepcopy(torch_model.linear1.weight.detach().numpy())
        my_model.linear1.parameters["b"].data = deepcopy(
            np.expand_dims(torch_model.linear1.bias.detach().numpy(), axis=0))
        my_model.linear2.parameters["W"].data = deepcopy(torch_model.linear2.weight.detach().numpy())
        my_model.linear2.parameters["b"].data = deepcopy(
            np.expand_dims(torch_model.linear2.bias.detach().numpy(), axis=0))

    my_loss_fn = bce_loss_with_logit
    my_optimizer = SGD(my_model.parameters(), lr=0.1)

    for epoch in range(20):
        torch_model.train()
        torch_out = torch_model(X_train_torch)
        my_out = my_model(X_train)

        torch_loss = torch_loss_fn(torch_out, y_train_torch)
        my_loss = my_loss_fn(my_out, y_train)
        np.testing.assert_allclose(torch_loss.item(), my_loss, atol=1e-5, rtol=1e-5)

        torch_optimizer.zero_grad()
        my_optimizer.zero_grad()
        torch_loss.backward()
        dldy = (sigmoid(my_out) - y_train) / y_train.shape[0]
        my_model.backward(dldy)

        torch_optimizer.step()
        my_optimizer.step()

        # print(f"Epoch {epoch+1}, torch loss: {torch_loss.item():.4f}, my loss: {my_loss:.4f}")

    # 测试
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(X_test_torch)
        my_out = my_model(X_test)
        np.testing.assert_allclose(torch_out.detach().numpy(), my_out, atol=1e-5, rtol=1e-5)

        torch_probs = torch.sigmoid(torch_out)  # 推理时加 Sigmoid 得到概率
        torch_preds = (torch_probs >= 0.5).float()
        my_probs = sigmoid(my_out)
        my_preds = (my_probs >= 0.5).astype(float)

        torch_acc = accuracy_score(y_test, torch_preds.numpy())
        torch_auc = roc_auc_score(y_test, torch_preds.numpy())
        my_acc = accuracy_score(y_test, my_preds)
        my_auc = roc_auc_score(y_test, my_preds)
        # print(f"torch acc : {torch_acc:.4f}, torch auc: {torch_auc:.4f}")
        # print(f"my acc : {my_acc:.4f}, my auc: {my_auc:.4f}")


if __name__ == "__main__":
    test_linear_model()
