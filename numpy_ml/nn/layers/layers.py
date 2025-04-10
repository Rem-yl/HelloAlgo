from abc import ABC, abstractmethod

import numpy as np


class LayerBase(ABC):
    """ 所有的输入输出必须是np.ndarray, 不接受任何的列表 
    X.shape: [Batch_size, *]
    Y.shape: [Batch_size, *]
    """

    def __init__(self, optimizer=None):
        """An abstract base class inherited by all neural network layers"""
        self.X: np.ndarray = None
        self.trainable = True
        self.optimizer = None

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}

        super().__init__()

    @abstractmethod
    def _init_params(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Perform a forward pass through the layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        """Perform a backward pass through the layer"""
        raise NotImplementedError

    @property
    def hyperparameters(self):
        return {}

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        self.trainable = False

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.trainable = True

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        assert self.trainable, f"{self.__class__.__name__} is frozen"
        self.X = None
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self):
        # todo: 等跟 optimizer一起更新
        pass

    def summary(self):
        return {
            "layer": self.__class__.__name__,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }


class Linear(LayerBase):
    r"""
        https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear 
        和pytorch的api调用保持完全一致

        Applies an affine linear transformation to the incoming data: :math:`y = xW^T + b

        - x.shape: [*, in_feats]
        - W.shape: [out_feats, in_feats]
        - b.shape: [1, out_feats]
    """

    def __init__(self, in_feat: int, out_feat: int, init=None, optimizer=None):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.init = init
        self.parameters = {"W": None, "b": None}
        super().__init__(optimizer)

        self._init_params()

    def _init_params(self):
        W = np.random.randn(self.out_feat, self.in_feat)
        b = np.zeros((1, self.out_feat))

        self.parameters = {"W": W, "b": b}
        self.derived_variables = {"out": None}
        self.gradients = {
            "W": np.zeros_like(W),
            "b": np.zeros_like(b),
        }

    @property
    def hyperparameters(self):
        return {
            "layer": self.__class__.__name__,
            "param_init": self.init,
            "in_feat": self.in_feat,
            "out_feat": self.out_feat,
            "optimizer":
                {
                    "cache": self.optimizer.cache,
                    "hyperparameters": self.optimizer.hyperparameters,
                },
        }

    def _check_input(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input type must be numpy array")

        if x.ndim <= 1:
            raise ValueError("Input dim must > 1")

    def forward(self, X: np.ndarray, retain_derived=True) -> np.ndarray:
        self._check_input(X)

        out = self._fwd(X)

        if retain_derived:
            self.X = X
            self.derived_variables["out"] = out

        return out

    def _fwd(self, X: np.ndarray) -> np.ndarray:
        W = self.parameters["W"]
        b = self.parameters["b"]

        out = X @ W.T + b

        return out

    def _check_grad(self, dldy: np.ndarray, out: np.ndarray):
        self._check_input(dldy)
        if dldy.shape != out.shape:
            raise ValueError(f"grad.shape: {dldy.shape} != out.shape: {out.shape}")

    def backward(self, dldy: np.ndarray, retain_grads=True):
        assert self.trainable, f"{self.__class__.__name__} is frozen"

        out = self.derived_variables["out"]
        self._check_grad(dldy, out)

        x = self.X
        dx, dw, db = self._bwd(dldy, x)

        if retain_grads:
            self.gradients["W"] += dw
            self.gradients["b"] += db

        return dx

    def _bwd(self, dldy: np.ndarray, x: np.ndarray):
        W = self.parameters["W"]

        dx = dldy @ W
        dw = dldy.T @ x
        db = dldy.sum(axis=0, keepdims=True)

        return dx, dw, db
