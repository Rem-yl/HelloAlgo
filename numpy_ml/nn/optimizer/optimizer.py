from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Union, Optional
import numpy as np

from ..param import Parameter


class OptimizerBase(ABC):
    def __init__(self, parameters: Optional[List[Dict[str, Parameter]]], lr: float, lr_scheduler=None):
        if parameters is None:
            self.parameters = []
        else:
            self.parameters = parameters

        self.param_cache = {}
        self.cur_step = 0
        self.lr = lr
        self.lr_scheduler = lr_scheduler

    def reset(self):
        self.cur_step = 0
        self.parameters = []
        self.param_cache = {}

    def zero_grad(self):
        for param_dict in self.parameters:
            for _, param in param_dict.items():
                if param.required_grads:
                    param.grad = np.zeros_like(param.grad)

    @property
    def hyperparameters(self):
        return {
            "id": self.__class__.__name__,
            "lr": self.lr,
            "lr_scheduler": str(self.lr_scheduler),
        }

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
        self,
        parameters: List[Dict[str, Parameter]] = None,
        lr=0.01,
        momentum=0.0,
        weight_decay=0,
        clip_norm=None,
        lr_scheduler=None
    ):
        super().__init__(parameters, lr, lr_scheduler)
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.weight_decay = weight_decay

    @property
    def hyperparameters(self):
        return {
            "id": self.__class__.__name__,
            "lr": self.lr,
            "momentum": self.momentum,
            "clip_norm": self.clip_norm,
            "weight_decay": self.weight_decay,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        hparam = self.hyperparameters
        lr, mm, cn, sc = hparam["lr"], hparam["momentum"], hparam["clip_norm"], hparam["lr_scheduler"]

        return f"SGD(lr={lr}, momentum={mm}, clip_norm={cn}, lr_scheduler={sc})"

    def step(self):
        hparam = self.hyperparameters
        momentum, clip_norm, lr, weight_decay = hparam["momentum"], hparam["clip_norm"], hparam["lr"], hparam["weight_decay"]
        t = np.inf if clip_norm is None else clip_norm  # 梯度裁剪

        self.cur_step += 1

        for param_dict in self.parameters:
            for param_name, param in param_dict.items():
                if param.grad is None or param.required_grads is False:
                    continue

                param_id = f"{param_name}_{id(param)}"
                param_grad = deepcopy(param.grad)

                if param_id not in self.param_cache:
                    self.param_cache[param_id] = np.zeros_like(param_grad)

                grad_norm = np.linalg.norm(param_grad)
                if grad_norm > t:
                    param_grad = param_grad * t / grad_norm

                if weight_decay != 0:
                    param_grad = weight_decay * param.data

                update = momentum * self.param_cache[param_id] + lr * param_grad
                self.param_cache[param_id] = update

                param.data -= update


class Adam(OptimizerBase):
    def __init__(
        self,
        parameters: List[Dict[str, Parameter]] = None,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        clip_norm=None,
        lr_scheduler=None
    ):
        super().__init__(parameters, lr, lr_scheduler)

        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm

    @property
    def hyperparameters(self):
        return {
            "id": self.__class__.__name__,
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "clip_norm": self.clip_norm,
            "weight_decay": self.weight_decay,
            "eps": self.eps,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        return f"Adam(lr={self.lr}, betas={(self.beta1, self.beta2)}, weight_decay={self.weight_decay}, clip_norm={self.clip_norm}, lr_scheduler={str(self.lr_scheduler)})"

    def step(self):
        hparam = self.hyperparameters
        lr, betas, clip_norm, weight_decay, eps = hparam["lr"], hparam["betas"], hparam["clip_norm"], hparam["weight_decay"], hparam["eps"]
        beta1, beta2 = betas
        t = np.inf if clip_norm is None else clip_norm

        self.cur_step += 1

        for param_dict in self.parameters:
            for param_name, param in param_dict.items():
                if param.grad is None or param.required_grads is False:
                    continue

                param_id = f"{param_name}_{id(param)}"
                param_grad = deepcopy(param.grad)

                if param_id not in self.param_cache:
                    self.param_cache[param_id] = {
                        "m": np.zeros_like(param_grad),
                        "v": np.zeros_like(param_grad)
                    }

                grad_norm = np.linalg.norm(param_grad)
                if grad_norm > t:
                    param_grad = param_grad * t / grad_norm

                if weight_decay != 0:
                    param_grad += weight_decay * param.data

                m = self.param_cache[param_id]["m"]
                v = self.param_cache[param_id]["v"]

                # 一阶矩估计
                m[:] = beta1 * m + (1 - beta1) * param_grad
                # 二阶矩估计
                v[:] = beta2 * v + (1 - beta2) * (param_grad ** 2)

                if self.cur_step > 10000:
                    m_hat, v_hat = m, v
                else:
                    m_hat = m / (1 - beta1 ** self.cur_step)
                    v_hat = v / (1 - beta2 ** self.cur_step)

                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                param.data -= update
