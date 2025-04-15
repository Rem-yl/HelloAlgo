import pytest
import numpy as np
import torch


from nn.activations import ReLU


class ActivationTestCase:
    def __init__(self, name, my_activation, torch_activation):
        self.name = name
        self.my_act = my_activation
        self.torch_act = torch_activation

    def forward(self, x: np.ndarray):
        out_my = self.my_act.forward(x)
        x_torch = torch.tensor(x, dtype=torch.float32)
        out_torch = self.torch_act(x_torch)
        return out_my, out_torch.detach().numpy()

    def backward(self, x: np.ndarray, grad_in: np.ndarray):
        self.my_act.forward(x)
        grad_my = self.my_act.backward(grad_in)

        x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        grad_in_torch = torch.tensor(grad_in, dtype=torch.float32)
        out_torch = self.torch_act(x_torch)
        out_torch.backward(grad_in_torch)
        grad_torch = x_torch.grad.numpy()
        return grad_my, grad_torch


@pytest.mark.parametrize("case", [
    ActivationTestCase("ReLU", ReLU(), torch.nn.ReLU()),
])
def test_activation_forward(case):
    x = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    out_my, out_torch = case.forward(x)
    np.testing.assert_allclose(out_my, out_torch, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("case", [
    ActivationTestCase("ReLU", ReLU(), torch.nn.ReLU()),
])
def test_activation_backward(case):
    x = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    grad_in = np.ones_like(x)
    grad_my, grad_torch = case.backward(x, grad_in)
    np.testing.assert_allclose(grad_my, grad_torch, rtol=1e-5, atol=1e-6)
