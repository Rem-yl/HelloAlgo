import numpy as np
import re
from ast import literal_eval as _eval

from ..activations import (
    Sigmoid,
    ReLU,
    LeakyReLU,
    GELU,
    Tanh,
    Affine,
    Identity,
    ELU,
    SELU,
    HardSigmoid,
    SoftPlus,
    ActivationBase,
)

from ..optimizer import (
    OptimizerBase,
    SGD
)


class OptimizerInitializer(object):
    """ 
    优化器工厂类 
    Valid `param` values are:
            (a) __str__ representations of `OptimizerBase` instances
            (b) `OptimizerBase` instances
            (c) Parameter dicts (e.g., as produced via the `summary` method in
                `LayerBase` instances)

        If `param` is `None`, return the SGD optimizer with default parameters.
    """

    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            opt = SGD()
        elif isinstance(param, OptimizerBase):
            opt = param
        elif isinstance(param, str):
            opt = self.init_from_str(param)
        else:
            raise ValueError(f"Unknown Optimizer: {param}")

        return opt

    def init_from_str(self, opt_str: str):
        r = r"([a-zA-Z]*)=([^,)]*)"
        opt_str = opt_str.lower()

        # _eval()是安全的eval()
        kwargs = {i: _eval(j) for i, j in re.findall(r, opt_str)}
        if "sgd" in opt_str:
            optimizer = SGD(**kwargs)
        else:
            raise NotImplementedError(f"{opt_str}")

        return optimizer


class ActivationInitializer(object):
    """ 激活函数工厂类 """

    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            act = Identity()
        elif isinstance(param, ActivationBase):
            act = param
        elif isinstance(param, str):
            act = self.init_from_str(param)
        else:
            raise ValueError(f"Unknown activation: {param}")

        return act

    def init_from_str(self, param: str):
        act_str = param.lower()
        if act_str == "sigmoid":
            act_fn = Sigmoid()
        elif act_str == "relu":
            act_fn = ReLU()
        elif "leakyrelu" in act_str:
            r = r"leakyrelu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            act_fn = LeakyReLU(alpha=float(alpha))
        elif "gelu" in act_str:
            r = r"gelu\(approximate=(.*)\)"
            approximate = re.match(r, act_str).groups()[0] == "true"
            act_fn = GELU(approximate=approximate)
        elif act_str == "tanh":
            act_fn = Tanh()
        elif "affine" in act_str:
            r = r"affine\(slope=(.*), intercept=(.*)\)"
            slope, intercept = re.match(r, act_str).groups()
            act_fn = Affine(slope=float(slope), intercept=float(intercept))
        elif act_str == "identity":
            act_fn = Identity()
        elif "elu" in act_str:
            r = r"elu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            act_fn = ELU(alpha=float(alpha))
        elif act_str == "selu":
            act_fn = SELU()
        elif act_str == "hardsigmoid":
            act_fn = HardSigmoid()
        elif act_str == "softplus":
            act_fn = SoftPlus()
        else:
            raise ValueError(f"Unknown activation: {act_str}")

        return act_fn
