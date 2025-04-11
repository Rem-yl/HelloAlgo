import pytest

from nn.initializers import ActivationInitializer, OptimizerInitializer
from nn.activations import Sigmoid
from nn.optimizer import (
    OptimizerBase,
    SGD,
)


class TestActivationInitializer:
    def test_init_with_class_instance(self):
        act = Sigmoid()
        initializer = ActivationInitializer(act)
        result = initializer()
        assert isinstance(result, Sigmoid)

    def test_init_with_string(self):
        initializer = ActivationInitializer("sigmoid")()
        assert isinstance(initializer, Sigmoid)


class TestOptimizerInitializer:
    def test_init_with_optimizer_instance(self):
        sgd = SGD(lr=0.01)
        optimizer = OptimizerInitializer(sgd)()

        assert optimizer is sgd
        assert isinstance(optimizer, OptimizerBase)

    def test_init_with_string(self):
        opt = OptimizerInitializer("SGD(lr=0.1)")()

        assert isinstance(opt, SGD)
        assert abs(opt.hyperparameters["lr"] - 0.1) < 1e-6
