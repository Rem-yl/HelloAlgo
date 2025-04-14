from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict
import numpy as np
from numpy.linalg import norm


class OptimizerBase(ABC):
    def __init__(self, lr, scheduler=None):
        self.cache = {}     # 存储参数更新的迭代变量
        self.cur_step = 0
        self.hyperparameters = {}
        self.lr_scheduler = None    # todo: need to implement

    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        self.cur_step += 1

    def reset_step(self):
        self.cur_step = 0

    def copy(self):
        return deepcopy(self)

    def set_params(self, hparam_dict: Dict = None, cache_dict: Dict = None):
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
                    if k == "lr_scheduler":
                        raise ValueError("SchedulerInitializer NotImplemented")

        if cache_dict is not None:
            for k, v in cache_dict.items():
                if k in self.cache:
                    self.cache[k] = v

    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss=None):
        raise NotImplementedError


class SGD(OptimizerBase):
    def __init__(
        self, lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None, **kwargs
    ):
        super().__init__(lr, lr_scheduler)
        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        lr, mm, cn, sc = H["lr"], H["momentum"], H["clip_norm"], H["lr_scheduler"]
        return "SGD(lr={}, momentum={}, clip_norm={}, lr_scheduler={})".format(
            lr, mm, cn, sc
        )

    def update(self, param, param_grad, param_name, cur_loss=None):
        cache = self.cache
        hparam = self.hyperparameters
        momentum, clip_norm = hparam["momentum"], hparam["clip_norm"]
        # lr = self.lr_scheduler(self.cur_step, cur_loss)  # todo: need to implement
        lr = hparam["lr"]

        if param_name not in cache:
            cache[param_name] = np.zeros_like(param_grad)

        # 裁剪梯度
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update = momentum * cache[param_name] + lr * param_grad
        self.cache[param_name] = update

        return param - update
