from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np
from ..utils import relu, sigmoid
from ..param import Parameter

eps = np.finfo(float).eps


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.derived_variables: Dict[str, Optional[Parameter]] = {"x": None, "out": None}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """ 抽象的前向传播方法 """
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        """ 抽象的反向传播方法 """
        raise NotImplementedError


class ReLU(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def forward(self, x: np.ndarray, retain_derived=True):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        out = relu(x)

        if retain_derived:
            self.derived_variables["x"] = Parameter(x)
            self.derived_variables["out"] = Parameter(out)

        return out

    def backward(self, grad_in: np.ndarray, retain_grads=True):
        x = self.derived_variables["x"].data

        if x is None:
            raise ValueError("Must call forward before backward")

        mask = np.where(x > 0, 1.0, 0.0)
        grad_out = grad_in * mask

        if retain_grads:
            self.derived_variables["x"].grad = grad_out
            self.derived_variables["out"].grad = grad_in

        return grad_out


class Sigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid"

    def forward(self, x: np.ndarray, retain_derived=True):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        out = sigmoid(x)

        if retain_derived:
            self.derived_variables["x"] = Parameter(x)
            self.derived_variables["out"] = Parameter(out)

        return out

    def grad(self, x: np.ndarray):
        s = sigmoid(x)
        return s * (1 - s)

    def backward(self, grad_in: np.ndarray, retain_grads=True):
        x = self.derived_variables["x"].data
        out = self.derived_variables["out"].data

        if x is None or out is None:
            raise ValueError("Must call forward before backward")

        grad_out = grad_in * (out * (1 - out))

        if retain_grads:
            self.derived_variables["x"].grad = grad_out
            self.derived_variables["out"].grad = grad_in

        return grad_out


class LeakyReLU(ActivationBase):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def __str__(self):
        return f"LeakyReLU(alpha={self.alpha})"

    def forward(self, x: np.ndarray, retain_derived=True):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        out = np.where(x > 0, x, self.alpha*x)

        if retain_derived:
            self.derived_variables["x"] = Parameter(x)
            self.derived_variables["out"] = Parameter(out)

        return out

    def backward(self, grad_in: np.ndarray, retain_grads=True):
        x = self.derived_variables["x"].data

        if x is None:
            raise ValueError("Must call forward before backward")

        grad_mask = np.where(x > 0, 1.0, self.alpha)
        grad_out = grad_in * grad_mask

        if retain_grads:
            self.derived_variables["x"].grad = grad_out
            self.derived_variables["out"].grad = grad_in

        return grad_out


class SELU(ActivationBase):
    def __init__(self):
        super().__init__()
        self.alpha = 1.6732632423543772
        self.scale = 1.0507009873554805

    def __str__(self):
        return f"SELU(alpha={self.alpha}, scale={self.scale})"

    def forward(self, x: np.ndarray, retain_derived=True):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        out = self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

        if retain_derived:
            self.derived_variables["x"] = Parameter(x)
            self.derived_variables["out"] = Parameter(out)

        return out

    def backward(self, grad_in: np.ndarray, retain_grads=True):
        x = self.derived_variables["x"].data

        if x is None:
            raise ValueError("Must call forward before backward")

        # Derivative of SELU: scale * (1 if x>0 else alpha * exp(x))
        grad_mask = self.scale * np.where(x > 0, 1.0, self.alpha * np.exp(x))
        grad_out = grad_in * grad_mask

        if retain_grads:
            self.derived_variables["x"].grad = grad_out
            self.derived_variables["out"].grad = grad_in

        return grad_out
