import pytest
import torch
import numpy as np

from nn.optimizer import SGD as MySGD
from nn.param import Parameter


class TestSGD:
    def setup_method(self):
        params = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        grads = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        parameters = Parameter(params)
        parameters.grad = grads

        self.parameters = {"w": parameters}
        self.my_optimizer = MySGD([self.parameters], lr=0.1, momentum=0.0)

    def test_step(self):
        self.my_optimizer.step()
        print(self.parameters["w"].grad)


if __name__ == "__main__":
    test = TestSGD()
    test.setup_method()
    test.test_step()
