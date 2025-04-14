from typing import List, Dict, Optional
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn as nn

from nn.utils import sigmoid
from nn.param import Parameter


from nn.layers import (
    LayerBase,
    Linear,
)
from nn.activations import (
    ActivationBase,
    ReLU,
)
from nn.optimizer import (
    OptimizerBase,
    SGD
)


def load_data():
    X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, flip_y=0, class_sep=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test


class FCModel:
    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int):
        self.model_list = []
        self.linear1 = Linear(input_size, hidden_size1)  # 第一层全连接
        self.linear2 = Linear(hidden_size1, hidden_size2)  # 第二层全连接
        self.linear3 = Linear(hidden_size2, output_size)  # 输出层

        self.act1 = ReLU()
        self.act2 = ReLU()
        self.model_list.extend([self.linear1, self.act1, self.linear2, self.act2, self.linear3])

    def __call__(self, X):
        return self.forward(X)

    def parameters(self) -> List[Dict[str, Parameter]]:
        params = []
        for layer in self.model_list:
            if isinstance(layer, LayerBase):
                params.append(layer.parameters)
                # for param_name, param in layer.parameters.items():
                #     print(f"[Model] param_name: {param_name}, id: {id(param)}")

        return params

    def forward(self, X):
        for model in self.model_list:
            X = model(X)

        return X

    def backward(self, dldy):
        for model in reversed(self.model_list):
            if isinstance(model, ActivationBase) or isinstance(model, LayerBase):
                dldy = model.backward(dldy, retain_grads=True)
            else:
                raise ValueError(f"Unknown layer: {model}")


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.model_list = []
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.act = nn.ReLU()

        self.model_list.extend([self.linear1, self.act, self.linear2, self.act, self.linear3])

    def forward(self, X):
        self.z1 = self.linear1(X)
        self.z1.retain_grad()
        self.a1 = self.act(self.z1)
        self.a1.retain_grad()

        self.z2 = self.linear2(self.a1)
        self.z2.retain_grad()
        self.a2 = self.act(self.z2)
        self.a2.retain_grad()

        self.z3 = self.linear3(self.a2)
        self.z3.retain_grad()

        return self.z3


def bce_loss_with_logit(logits: np.ndarray, y: np.ndarray):
    y_pred = sigmoid(logits)
    epsilon = 1e-10  # 避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def test_model():
    X_train, X_test, y_train, y_test = load_data()
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)

    model = FCModel(input_size=X_train.shape[1], hidden_size1=10, hidden_size2=4, output_size=1)
    torch_model = TorchModel(input_size=X_train.shape[1], hidden_size1=10, hidden_size2=4, output_size=1)

    with torch.no_grad():
        torch_model.linear1.weight.copy_(torch.tensor(model.linear1.params["W"], dtype=torch.float32))
        torch_model.linear1.bias.copy_(torch.tensor(model.linear1.params["b"].squeeze(), dtype=torch.float32))
        torch_model.linear2.weight.copy_(torch.tensor(model.linear2.params["W"], dtype=torch.float32))
        torch_model.linear2.bias.copy_(torch.tensor(model.linear2.params["b"].squeeze(), dtype=torch.float32))
        torch_model.linear3.weight.copy_(torch.tensor(model.linear3.params["W"], dtype=torch.float32))
        torch_model.linear3.bias.copy_(torch.tensor(model.linear3.params["b"].squeeze(), dtype=torch.float32))

    np.testing.assert_allclose(model.linear1.params["W"], torch_model.linear1.weight.detach().numpy())
    np.testing.assert_allclose(model.linear2.params["W"], torch_model.linear2.weight.detach().numpy())
    np.testing.assert_allclose(model.linear3.params["W"], torch_model.linear3.weight.detach().numpy())
    np.testing.assert_allclose(model.linear1.params["b"].squeeze(), torch_model.linear1.bias.detach().numpy())
    np.testing.assert_allclose(model.linear2.params["b"].squeeze(), torch_model.linear2.bias.detach().numpy())
    np.testing.assert_allclose(model.linear3.params["b"].squeeze(), torch_model.linear3.bias.detach().numpy())

    num_samples = X_train.shape[0]
    batch_size = 32
    torch_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    my_loss_fn = bce_loss_with_logit

    my_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.0)
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01, momentum=0.0)
    my_epoch_loss = 0.0
    torch_epoch_loss = 0.0
    j = 0

    for epoch in range(10):
        num_batches = int(np.ceil(num_samples / batch_size))

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)

            X = X_train[start:end]
            y = y_train[start:end]
            X_torch = torch.tensor(X, dtype=torch.float32)
            y_torch = torch.tensor(y, dtype=torch.float32)

            my_out = model(X)
            torch_out = torch_model(X_torch)
            np.testing.assert_allclose(my_out, torch_out.detach().numpy(), atol=1e-5, rtol=1e-5)

            torch_loss = torch_loss_fn(torch_out, y_torch).sum()
            my_loss = my_loss_fn(my_out, y)
            torch_epoch_loss += torch_loss
            my_epoch_loss += my_loss
            j += 1

            np.testing.assert_allclose(my_loss, torch_loss.detach().numpy(), atol=1e-5, rtol=1e-5)

            my_pred = sigmoid(my_out)
            dldy = (my_pred - y) / y.shape[0]
            my_optimizer.zero_grad()
            torch_model.zero_grad()
            torch_loss.backward()
            model.backward(dldy)

            with torch.no_grad():
                np.testing.assert_allclose(dldy, torch_out.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(model.linear3.parameters["W"].grad,
                                           torch_model.linear3.weight.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(model.linear3.parameters["b"].grad[0, :],
                                           torch_model.linear3.bias.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(dldy, torch_model.z3.grad.detach().numpy(), atol=1e-6, rtol=1e-6)

                np.testing.assert_allclose(model.linear2.parameters["W"].grad,
                                           torch_model.linear2.weight.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(model.linear2.parameters["b"].grad[0, :],
                                           torch_model.linear2.bias.grad.detach().numpy(), atol=1e-6, rtol=1e-6)

                np.testing.assert_allclose(model.linear1.parameters["W"].grad,
                                           torch_model.linear1.weight.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(model.linear1.parameters["b"].grad[0, :],
                                           torch_model.linear1.bias.grad.detach().numpy(), atol=1e-6, rtol=1e-6)

            torch_optimizer.step()
            my_optimizer.step()
            np.testing.assert_allclose(model.linear3.parameters["W"].grad,
                                       torch_model.linear3.weight.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(model.linear3.parameters["b"].grad[0, :],
                                       torch_model.linear3.bias.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(model.linear2.parameters["W"].grad,
                                       torch_model.linear2.weight.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(model.linear2.parameters["b"].grad[0, :],
                                       torch_model.linear2.bias.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(model.linear1.parameters["W"].grad,
                                       torch_model.linear1.weight.grad.detach().numpy(), atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(model.linear1.parameters["b"].grad[0, :],
                                       torch_model.linear1.bias.grad.detach().numpy(), atol=1e-6, rtol=1e-6)

        np.testing.assert_allclose(torch_epoch_loss.detach().numpy(), my_epoch_loss, atol=1e-4, rtol=1e-5)

        torch_y_pred = torch_model(X_test_torch)
        my_y_pred = model(X_test)
        np.testing.assert_allclose(torch_y_pred.detach().numpy(), my_y_pred, atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
    test_model()
