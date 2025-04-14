from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Union
import numpy as np

from ..param import Parameter


class OptimizerBase(ABC):
    def __init__(self, parameters: List[Dict[str, Parameter]], lr: float, lr_scheduler=None):
        if not isinstance(parameters, List):
            raise TypeError(f"parameters must be list!")

        self.param_cache = {}
        self.cur_step = 0
        self.hyperparameters = {}
        self.parameters: List[Dict[str, Parameter]] = parameters
        self.lr_scheduler = lr_scheduler

    def reset(self):
        self.cur_step = 0
        self.parameters = []
        self.param_cache = {}

    def zero_grad(self):
        for param_dict in self.parameters:
            for _, param in param_dict.items():
                param.grad = np.zeros_like(param.grad)

    @abstractmethod
    def step(self):
        raise NotImplementedError

    def copy(self):
        return deepcopy(self)

    def add_parameter(self, parameter: Union[Dict[str, Parameter], List[Dict[str, Parameter]]]):
        if isinstance(parameter, Dict):
            self.parameters.append(parameter)
        elif isinstance(parameter, List):
            self.parameters.extend(parameter)
        else:
            raise TypeError("parameter must be Dict[str, Parameter] or List[Dict[str, Parameter]]")


class SGD(OptimizerBase):
    def __init__(
        self, parameters: List[Dict[str, Parameter]], lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None
    ):
        super().__init__(parameters, lr, lr_scheduler)

        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        hparam = self.hyperparameters
        lr, mm, cn, sc = hparam["lr"], hparam["momentum"], hparam["clip_norm"], hparam["lr_scheduler"]

        return f"SGD(lr={lr}, momentum={mm}, clip_norm={cn}, lr_scheduler={sc})"

    def step(self):
        hparam = self.hyperparameters
        momentum, clip_norm, lr = hparam["momentum"], hparam["clip_norm"], hparam["lr"]
        t = np.inf if clip_norm is None else clip_norm  # 梯度裁剪

        for param_dict in self.parameters:
            for param_name, param in param_dict.items():
                param_id = f"{param_name}_{id(param)}"
                # print(f"[SGD] param_name: {param_name}, id: {id(param)}")
                if param.grad is None:
                    continue

                if param_id not in self.param_cache:
                    self.param_cache[param_id] = np.zeros_like(param.grad)

                grad_norm = np.linalg.norm(param.grad)
                if grad_norm > t:
                    param.grad = param.grad * t / grad_norm

                update = momentum * self.param_cache[param_id] + lr * param.grad
                self.param_cache[param_id] = update

                param.data -= update

        self.cur_step += 1
