""" 各个初始化器的工厂函数 """
from typing import Optional, Union
import re
import numpy as np

from ..activations import (
    ActivationBase,
    ReLU
)

from ..optimizer import (
    OptimizerBase,
    SGD,
)


class ActivationInitializer(object):
    def __init__(self, init: Union[ActivationBase, str]):
        """
            支持解析字符串来初始化activation
        """
        self.init = init

    def __call__(self):
        init = self.init
        if isinstance(init, ActivationBase):
            act = init
        elif isinstance(init, str):
            act = self._init_from_str(init)
        elif init is None:
            raise ValueError("activation init is None.")
        else:
            raise ValueError(f"Unknown activation: {init}")

        return act

    def _init_from_str(self, init: str):
        init = init.lower()
        if init == "relu":
            act = ReLU()
        else:
            raise NotImplementedError(f"activation: {init}")

        return act


class OptimizerInitializer(object):
    def __init__(self, init: Union[OptimizerBase, str] = None):
        self.init = init

    def __call__(self):
        init = self.init
        if isinstance(init, OptimizerBase):
            optimizer = init
        elif isinstance(init, str):
            optimizer = self._init_from_str(init)
        elif init is None:
            raise ValueError("optimizer init is None.")
        else:
            raise ValueError(f"Unknown optimizer: {init}")

        return optimizer

    def _init_from_str(self, init: str):
        r = r"([a-zA-Z]*)=([^,)]*)"
        init = init.lower()
        kwargs = {i: eval(j) for i, j in re.findall(r, init)}
        if "sgd" in init:
            optimizer = SGD(**kwargs)
        else:
            raise NotImplementedError(f"optimizer: {init}")

        return optimizer
