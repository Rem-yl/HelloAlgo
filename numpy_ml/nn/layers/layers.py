from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional

from ..param import Parameter


class LayerBase(ABC):
    """ 所有的输入输出必须是np.ndarray, 不接受任何的列表 
    X.shape: [Batch_size, *]
    Y.shape: [Batch_size, *]
    """

    def __init__(self):
        """An abstract base class inherited by all neural network layers"""
        super().__init__()

        self.trainable = True
        self.parameters: Dict[str, Optional[Parameter]] = {}
        self.derived_variables: Dict[str, Optional[Parameter]] = {"x": None, "out": None}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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
    @abstractmethod
    def hyperparameters(self):
        raise NotImplementedError

    @property
    def params(self):
        np_param = {}
        for param_name, param in self.parameters.items():
            np_param[param_name] = np.asarray(param.data)

        return np_param

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        self.trainable = False
        for _, param in self.parameters.items():
            param.freeze()

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.trainable = True
        for _, param in self.parameters.items():
            param.unfreeze()

    def summary(self):
        return {
            "layer": self.__class__.__name__,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

    def _check_input(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input type must be numpy array")

        if x.ndim <= 1:
            raise ValueError("Input dim must > 1")

    def _check_grad(self, dldy: np.ndarray, out: np.ndarray):
        self._check_input(dldy)
        if dldy.shape != out.shape:
            raise ValueError(f"grad.shape: {dldy.shape} != out.shape: {out.shape}")


class Linear(LayerBase):
    r"""
        https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear 
        和pytorch的api调用保持完全一致

        Applies an affine linear transformation to the incoming data: :math:`y = xW^T + b

        - x.shape: [*, in_feats]
        - W.shape: [out_feats, in_feats]
        - b.shape: [1, out_feats]
    """

    def __init__(self, in_feat: int, out_feat: int, init=None):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.init = init
        self.parameters: Dict[str, Optional[Parameter]] = {"W": None, "b": None}

        self._init_params()

    def _init_params(self):
        W = np.random.randn(self.out_feat, self.in_feat) * 0.1
        b = np.zeros((1, self.out_feat))

        W = Parameter(W)
        b = Parameter(b)

        self.parameters: Dict[str, Optional[Parameter]] = {"W": W, "b": b}

    @property
    def hyperparameters(self):
        return {
            "layer": self.__class__.__name__,
            "param_init": self.init,
            "in_feat": self.in_feat,
            "out_feat": self.out_feat,
        }

    def forward(self, X: np.ndarray, retain_derived=True) -> np.ndarray:
        self._check_input(X)

        out = self._fwd(X)

        if retain_derived:
            self.derived_variables["out"] = Parameter(out)
            self.derived_variables["x"] = Parameter(X)

        return out

    def _fwd(self, X: np.ndarray) -> np.ndarray:
        W = np.asarray(self.parameters["W"].data)
        b = np.asarray(self.parameters["b"].data)

        out = X @ W.T + b

        return out

    def backward(self, dldy: np.ndarray, retain_grads=True):
        x = self.derived_variables["x"].data
        out = self.derived_variables["out"].data
        self._check_grad(dldy, out)

        dx, dw, db = self._bwd(dldy, x)

        if retain_grads:
            self.derived_variables["x"].grad = dx
            self.derived_variables["out"].grad = dldy
            self.parameters["W"].grad += dw
            self.parameters["b"].grad += db

        return dx

    def _bwd(self, dldy: np.ndarray, x: np.ndarray):
        W = np.asarray(self.parameters["W"].data)

        dx = dldy @ W
        dw = dldy.T @ x
        db = dldy.sum(axis=0, keepdims=True)

        return dx, dw, db


class Flatten(LayerBase):
    r"""
        https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html

        Flattens a contiguous range of dimensions into a single dimension.

        Example:
        - input.shape: (B, C, H, W)
        - output.shape: (B, C*H*W)  # if start_dim=1, end_dim=-1
        - output.shape: (B*C*H*W,) # if start_dim=0, end_dim=-1
    """

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.parameters = {}
        self.derived_variables = {"input_shape": None, "x": None, "out": None}

        self._init_params()

    def _init_params(self):
        self.parameters = {}

    @property
    def hyperparameters(self):
        return {
            "layer": self.__class__.__name__,
            "start_dim": self.start_dim,
            "end_dim": self.end_dim,
        }

    def forward(self, X: np.ndarray, retain_derived=True) -> np.ndarray:
        self._check_input(X)
        input_shape = X.shape

        start = self.start_dim if self.start_dim >= 0 else X.ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else X.ndim + self.end_dim

        flattened_dim = int(np.prod(X.shape[start:end + 1]))
        new_shape = X.shape[:start] + (flattened_dim,) + X.shape[end + 1:]

        out = X.reshape(new_shape)

        if retain_derived:
            self.derived_variables["x"] = Parameter(X)
            self.derived_variables["out"] = Parameter(out)
            self.derived_variables["input_shape"] = input_shape

        return out

    def backward(self, dldy: np.ndarray, retain_grads=True):
        input_shape = self.derived_variables["input_shape"]
        dx = dldy.reshape(input_shape)
        if retain_grads:
            self.derived_variables["x"].grad = dx
            self.derived_variables["out"].grad = dldy

        return dx


class BatchNorm2D(LayerBase):
    r"""
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

        Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
        Expected input shape: (N, C, H, W)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.parameters = {"gamma": None, "beta": None}
        self.derived_variables = {"x": None, "out": None, "x_centered": None, "std_inv": None, "x_norm": None}

        self._init_params()

    def _init_params(self):
        self.parameters: Dict[str, Optional[Parameter]] = {
            "gamma": Parameter(np.ones((1, self.num_features, 1, 1))) if self.affine else None,
            "beta": Parameter(np.zeros((1, self.num_features, 1, 1))) if self.affine else None,
        }

        self.running_mean = np.zeros((1, self.num_features, 1, 1)) if self.track_running_stats else None
        self.running_var = np.ones((1, self.num_features, 1, 1)) if self.track_running_stats else None

        self.derived_variables: Dict[str, Optional[Parameter]] = {
            "x_centered": None,
            "std_inv": None,
            "x_norm": None,
        }

    @property
    def hyperparameters(self):
        return {
            "layer": self.__class__.__name__,
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "track_running_stats": self.track_running_stats,
        }

    def _check_input(self, x: np.ndarray):
        if x.ndim != 4:
            raise ValueError(f"expected 4D input (got {x.ndim}D input)")

    def forward(self, X: np.ndarray, retain_derived=True):
        self._check_input(X)

        if self.trainable:
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = X.var(axis=(0, 2, 3), keepdims=True)
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                mean = X.mean(axis=(0, 2, 3), keepdims=True)
                var = X.var(axis=(0, 2, 3), keepdims=True)

        X_centered = X - mean
        std_inv = 1.0 / np.sqrt(var + self.eps)
        X_norm = X_centered * std_inv

        if self.affine:
            out = self.parameters["gamma"].data * X_norm + self.parameters["beta"].data
        else:
            out = X_norm

        if retain_derived:
            self.derived_variables["x"] = Parameter(X)
            self.derived_variables["out"] = Parameter(out)
            self.derived_variables["x_centered"] = Parameter(X_centered)
            self.derived_variables["std_inv"] = Parameter(std_inv)
            self.derived_variables["x_norm"] = Parameter(X_norm)

        return out

    def backward(self, dldy: np.ndarray, retain_grads=True):
        X_centered = self.derived_variables["x_centered"].data
        std_inv = self.derived_variables["std_inv"].data
        X_norm = self.derived_variables["x_norm"].data
        N, C, H, W = dldy.shape
        m = N * H * W

        dX_norm = dldy * self.parameters["gamma"].data if self.affine else dldy

        dvar = np.sum(dX_norm * X_centered, axis=(0, 2, 3), keepdims=True) * -0.5 * (std_inv**3)
        dmean = np.sum(dX_norm * -std_inv, axis=(0, 2, 3), keepdims=True) + dvar * \
            np.mean(-2. * X_centered, axis=(0, 2, 3), keepdims=True)
        dX = dX_norm * std_inv + dvar * 2 * X_centered / m + dmean / m

        if retain_grads:
            self.derived_variables["x"].grad = dX
            self.derived_variables["out"].grad = dldy

            if self.affine:
                self.parameters["gamma"].grad += np.sum(dldy * X_norm, axis=(0, 2, 3), keepdims=True)
                self.parameters["beta"].grad += np.sum(dldy, axis=(0, 2, 3), keepdims=True)

        return dX


class BatchNorm1D(LayerBase):
    r"""
    Applies Batch Normalization over a 2D or 3D input.
    Expected input shape: (N, C) or (N, C, L)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self._init_params()

    def _init_params(self):
        shape = (self.num_features,)
        if self.affine:
            self.parameters = {
                "gamma": Parameter(np.ones(shape, dtype=np.float32)),
                "beta": Parameter(np.zeros(shape, dtype=np.float32)),
            }
        else:
            self.parameters = {"gamma": None, "beta": None}

        if self.track_running_stats:
            self.running_mean = np.zeros(shape, dtype=np.float32)
            self.running_var = np.ones(shape, dtype=np.float32)
        else:
            self.running_mean = None
            self.running_var = None

        self.derived_variables = {
            "x": None,
            "out": None,
            "x_centered": None,
            "std_inv": None,
            "x_norm": None,
        }

    @property
    def hyperparameters(self):
        return {
            "layer": self.__class__.__name__,
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "track_running_stats": self.track_running_stats,
        }

    def _check_input(self, x: np.ndarray):
        if x.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D input (got {x.ndim}D input)")

    def forward(self, X: np.ndarray, retain_derived=True):
        self._check_input(X)

        is_2d = X.ndim == 2
        if is_2d:
            X = X[:, :, None]  # (N, C) -> (N, C, 1)

        mean = X.mean(axis=(0, 2)).astype(np.float32)   # (C, )
        var = X.var(axis=(0, 2)).astype(np.float32)     # (C, )

        if self.trainable:
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean if self.track_running_stats else mean
            var = self.running_var if self.track_running_stats else var

        mean = mean.reshape(1, -1, 1)
        std = np.sqrt(var + self.eps).reshape(1, -1, 1)

        x_centered = X - mean
        x_norm = x_centered / std

        if self.affine:
            gamma = self.parameters["gamma"].data.reshape(1, -1, 1)
            beta = self.parameters["beta"].data.reshape(1, -1, 1)
            out = gamma * x_norm + beta
        else:
            out = x_norm

        if retain_derived:
            self.derived_variables["x"] = Parameter(X)
            self.derived_variables["out"] = Parameter(out)
            self.derived_variables["x_centered"] = x_centered
            self.derived_variables["std_inv"] = 1.0 / std
            self.derived_variables["x_norm"] = x_norm

        return out.squeeze(-1) if is_2d else out

    def backward(self, dldy: np.ndarray, retain_grads=True):
        is_2d = dldy.ndim == 2
        if is_2d:
            dldy = dldy[:, :, None]

        X_centered = self.derived_variables["x_centered"]
        std_inv = self.derived_variables["std_inv"]
        x_norm = self.derived_variables["x_norm"]
        N, C, L = dldy.shape
        m = N * L

        if self.affine:
            gamma = self.parameters["gamma"].data.reshape(1, C, 1)
            dX_norm = dldy * gamma
        else:
            dX_norm = dldy

        dvar = np.sum(dX_norm * X_centered, axis=(0, 2), keepdims=True) * -0.5 * (std_inv**3)
        dmean = np.sum(dX_norm * -std_inv, axis=(0, 2), keepdims=True) + \
            dvar * np.mean(-2. * X_centered, axis=(0, 2), keepdims=True)

        dX = dX_norm * std_inv + dvar * 2 * X_centered / m + dmean / m

        if retain_grads:
            if self.affine:
                self.parameters["gamma"].grad = np.sum(dldy * x_norm, axis=(0, 2)).reshape(-1)
                self.parameters["beta"].grad = np.sum(dldy, axis=(0, 2)).reshape(-1)

            self.derived_variables["x"].grad = dX
            self.derived_variables["out"].grad = dldy

        return dX.squeeze(-1) if is_2d else dX
