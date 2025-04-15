import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from nn.layers import (
    LayerBase,
    Linear,
)
from nn.activations import ReLU
from nn.optimizer import SGD
from nn.utils import sigmoid
from nn.init import kaiming_normal_


def load_binary_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)


class TwoLayerClassifier:
    def __init__(self, input_dim: int, hidden_dim: int):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.linear2 = Linear(hidden_dim, 1)

        self.layer_list = [
            self.linear1, self.relu, self.linear2
        ]

        # self._init_param()

    def _init_param(self):
        kaiming_normal_(self.linear1.parameters["W"].data)
        kaiming_normal_(self.linear2.parameters["W"].data)

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def forward(self, x, retain_derived=True):
        for layer in self.layer_list:
            x = layer(x, retain_derived)

        return x

    def backward(self, grad_in, retain_grads=True):
        for layer in reversed(self.layer_list):
            grad_in = layer.backward(grad_in, retain_grads)

        return grad_in

    def parameters(self):
        params = []
        for layer in self.layer_list:
            if isinstance(layer, LayerBase):
                params.append(layer.parameters)

        return params


def bce_loss_with_logit(logits: np.ndarray, y: np.ndarray):
    y_pred = sigmoid(logits)
    epsilon = 1e-10  # 避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def train():
    X_train, X_test, y_train, y_test = load_binary_data()
    model = TwoLayerClassifier(input_dim=30, hidden_dim=16)
    optimizer = SGD(model.parameters(), lr=0.1)
    loss_fn = bce_loss_with_logit
    epoch_loss = 0.0
    for epoch in range(30):
        out = model(X_train)
        loss = loss_fn(out, y_train)
        epoch_loss += loss
        dldy = (1 / y_train.shape[0]) * (sigmoid(out) - y_train.reshape(-1, 1))

        model.backward(dldy)
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        epoch_loss = 0.0

    test_logits = model(X_test)
    preds = sigmoid(test_logits) > 0.5
    acc = np.mean(preds.flatten() == y_test)
    auc = roc_auc_score(y_test.reshape(-1, 1), sigmoid(test_logits))

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc}")


if __name__ == "__main__":
    train()
