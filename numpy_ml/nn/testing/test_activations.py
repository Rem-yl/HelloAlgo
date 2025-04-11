import pytest
import numpy as np
import torch
import torch.nn.functional as F


from nn.activations import Sigmoid


class TestSigmoid:
    def test_forward(self):
        np.random.seed(42)
        x_np = np.random.randn(4, 5).astype(np.float32)
        x_torch = torch.tensor(x_np)

        sigmoid = Sigmoid()
        out_np = sigmoid.forward(x_np)
        out_torch = torch.sigmoid(x_torch).numpy()

        assert np.allclose(out_np, out_torch, atol=1e-6), "Forward output not close to PyTorch"

    def test_backward(self):
        np.random.seed(42)
        x_np = np.random.randn(4, 5).astype(np.float32)
        x_torch = torch.tensor(x_np, requires_grad=True)

        sigmoid = Sigmoid()
        grad_np = sigmoid.backward(x_np)

        # PyTorch gradient computation
        out_torch = torch.sigmoid(x_torch)
        grad_torch = out_torch * (1 - out_torch)
        grad_torch_np = grad_torch.detach().numpy()

        assert np.allclose(grad_np, grad_torch_np, atol=1e-6), "Backward gradient not close to PyTorch"
