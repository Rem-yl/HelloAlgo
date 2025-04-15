from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np
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

    def grad(self, x: np.ndarray):
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray, retain_derived=True):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        out = self.grad(x)

        if retain_derived:
            self.derived_variables["x"] = Parameter(x)
            self.derived_variables["out"] = Parameter(out)

        return out

    def backward(self, grad_in: np.ndarray, retain_grads=True):
        X = self.derived_variables["x"].data

        if X is None:
            raise ValueError("Must call forward before backward")

        mask = np.where(X > 0, 1.0, 0.0)
        grad_out = grad_in * mask

        if retain_grads:
            self.derived_variables["x"].grad = grad_out
            self.derived_variables["out"].grad = grad_in

        return grad_out
