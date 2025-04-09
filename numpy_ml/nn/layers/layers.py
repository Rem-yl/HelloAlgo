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
    pass
