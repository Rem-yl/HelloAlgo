import pytest
import numpy as np
from numpy.testing import assert_allclose

from nn.layers import *


class TestAddLayer:
    @classmethod
    def setup_class(cls):
        cls.layer = Add("identity", "sgd(lr=0.01, momentum=0.0)")

    def setup_method(self, method):
        self.layer.X.clear()
        self.layer.gradients.clear()
        self.layer.parameters.clear()
        self.layer.derived_variables["sum"].clear()

    def test_forward_single(self):
        x = [np.array([[1, 2, 3], [3, 4, 5]]) for _ in range(3)]
        out = self.layer(x)
        assert_allclose(out, np.array([[3, 6, 9], [9, 12, 15]]))

    def test_backward_single(self):
        x = [np.array([[1, 2, 3], [3, 4, 5]]) for _ in range(3)]
        _ = self.layer(x)
        dldy = np.array([[1, 2, 3], [1, 1, 1]])
        grads = self.layer.backward(dldy)
        assert_allclose(grads, [dldy] * len(x))


class TestMultiplyLayer:
    @classmethod
    def setup_class(cls):
        cls.layer = Multiply("identity", "sgd(lr=0.01, momentum=0.0)")

    def setup_method(self, method):
        self.layer.X.clear()
        self.layer.gradients.clear()
        self.layer.parameters.clear()
        self.layer.derived_variables["product"].clear()

    def test_forward_single(self):
        x = [np.array([[1, 2, 3], [3, 4, 5]]) for _ in range(3)]
        out = self.layer(x)
        assert_allclose(out, np.array([[1, 8, 27], [27, 64, 125]]))

    def test_backward_single(self):
        x = [np.array([[1, 2, 3], [3, 4, 5]]) for _ in range(3)]
        _ = self.layer(x)
        dldy = np.array([[1, 2, 3], [1, 1, 1]])
        grads = self.layer.backward(dldy)


class TestFlattenLayer:
    @classmethod
    def setup_class(cls):
        cls.layer = Flatten(keep_dim=-1, optimizer="sgd(lr=0.01, momentum=0.0)")

    def setup_method(self, method):
        self.layer.X.clear()
        self.layer.gradients.clear()
        self.layer.parameters.clear()
        self.layer.derived_variables["in_dims"].clear()

    def test_forward_single(self):
        x = np.array([[1, 2, 3], [3, 4, 5]])
        out = self.layer(x)
        print(out.shape)
