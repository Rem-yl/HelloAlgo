from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional

from ..utils import sigmoid


class LossBase(ABC):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def hyperparameters(self):
        raise NotImplementedError


class BCEWithLogitLoss(LossBase):
    def __init__(self, weight=None, pos_weight=None, reduction='mean', eps=1e-6):
        super().__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.eps = eps
        self.cache = {"logits": None, "y": None}

    @property
    def hyperparameters(self):
        return {
            "id": self.__class__.__name__,
            "weight": self.weight,
            "pos_weight": self.pos_weight,
            "reduction": self.reduction,
            "eps": self.eps,
        }

    def _check_input(self, x: np.ndarray, y: np.ndarray):
        if x.shape != y.shape:
            raise ValueError(f"Input arrays x and y must have the same shape. Got shapes {x.shape} and {y.shape}.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("All values in y must be either 0 or 1.")

    def forward(self, logits: np.ndarray, y: np.ndarray, retain_derived=True):
        """ 损失函数变式的推导: http://giantpandacv.com/academic/%E7%AE%97%E6%B3%95%E7%A7%91%E6%99%AE/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/Pytorch%E4%B8%AD%E7%9A%84%E5%9B%9B%E7%A7%8D%E7%BB%8F%E5%85%B8Loss%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90/"""
        self._check_input(logits, y)

        if retain_derived:
            self.cache["logits"], self.cache["y"] = logits, y
        max_val = np.maximum(-logits, 0)
        neg_loss = (1 - y) * logits
        pos_loss = np.log(np.exp(-max_val)+np.exp(-logits-max_val)) + max_val

        if self.pos_weight is not None:
            log_weight = 1 + (self.pos_weight - 1) * y    # if y=1, log_weight=self.pos_weight, else 1
            loss = neg_loss + log_weight * pos_loss  # 广播机制, log_weight * pos_loss会将y=0处的值置0
        else:
            loss = neg_loss + pos_loss

        if self.weight is not None:
            loss *= self.weight

        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        else:
            return loss

    def backward(self):
        logits, targets = self.cache["logits"], self.cache["y"]
        if logits is None or targets is None:
            raise ValueError("Make sure forward is called and retain_derived is True before backward.")

        y_pred = sigmoid(logits)
        grad = y_pred - targets

        if self.weight is not None:
            grad *= self.weight

        if self.pos_weight is not None:
            grad = grad * (self.pos_weight * targets + (1 - targets))

        # reduction
        if self.reduction == "mean":
            grad = grad / logits.shape[0]
        elif self.reduction == "sum":
            pass
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        return grad


class BCELoss(LossBase):
    def __init__(self, weight: Optional[float] = None, reduction: str = 'mean', eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.eps = eps

        self.cache = {"y_pred": None, "y": None}

    @property
    def hyperparameters(self):
        return {
            "id": self.__class__.__name__,
            "weight": self.weight,
            "reduction": self.reduction,
            "eps": self.eps,
        }

    def _check_input(self, x: np.ndarray, y: np.ndarray):
        if x.shape != y.shape:
            raise ValueError(f"Input arrays x and y must have the same shape. Got shapes {x.shape} and {y.shape}.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("All values in y must be either 0 or 1.")
        if not np.all((x >= 0) & (x <= 1)):
            raise ValueError("All values in x must be probabilities in [0, 1] range.")

    def forward(self, y_pred: np.ndarray, y: np.ndarray, retain_derived=True):
        self._check_input(y_pred, y)

        y_pred = np.clip(y_pred, self.eps, 1-self.eps)
        if retain_derived:
            self.cache["y_pred"], self.cache["y"] = y_pred, y

        loss = -y * np.log(y_pred) - (1-y) * np.log(1-y_pred)

        if self.weight is not None:
            loss *= self.weight

        if self.reduction == "mean":
            return np.mean(loss)
        elif self.reduction == "sum":
            return np.sum(loss)
        else:
            return loss

    def backward(self):
        y_pred, y = self.cache["y_pred"], self.cache["y"]
        if y_pred is None or y is None:
            raise ValueError("Make sure forward is called and retain_derived is True before backward.")

        grad = (y_pred - y) / (y_pred * (1-y_pred))

        if self.weight is not None:
            grad *= self.weight

        if self.reduction == "mean":
            grad = grad / y_pred.shape[0]
        elif self.reduction == "sum":
            pass
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        return grad
