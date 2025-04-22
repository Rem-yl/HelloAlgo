import numpy as np
import torch
import torch.nn as nn
import pytest

from copy import deepcopy
from nn.layers import Linear, Flatten, BatchNorm2D, BatchNorm1D, AdaptiveAvgPool1d


@pytest.mark.parametrize("batch_size, in_feat, out_feat", [
    (4, 3, 2),
    (10, 5, 7),
    (1, 6, 4)
])
def test_linear(batch_size, in_feat, out_feat):
    np.random.seed(42)
    torch.manual_seed(42)

    X_np = np.random.randn(batch_size, in_feat).astype(np.float32)
    X_torch = torch.tensor(deepcopy(X_np), requires_grad=True)

    my_fc = Linear(in_feat, out_feat)

    pt_fc = nn.Linear(in_feat, out_feat, bias=True)
    pt_fc.weight.data = torch.tensor(my_fc.parameters["W"].data, dtype=torch.float32)
    pt_fc.bias.data = torch.tensor(my_fc.parameters["b"].data.squeeze(), dtype=torch.float32)
    np.testing.assert_allclose(my_fc.parameters["W"].data, pt_fc.weight.data.detach().numpy(),  rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(my_fc.parameters["b"].data.squeeze(),
                               pt_fc.bias.data.detach().numpy(),  rtol=1e-5, atol=1e-6)

    out_my = my_fc(X_np)
    out_pt = pt_fc(X_torch)

    np.testing.assert_allclose(out_my, out_pt.detach().numpy(), rtol=1e-5, atol=1e-6)

    dLdy_np = np.random.randn(*out_my.shape).astype(np.float32)
    dLdy_pt = torch.tensor(deepcopy(dLdy_np))

    dX_my = my_fc.backward(dLdy_np)
    out_pt.backward(dLdy_pt)

    dX_pt = X_torch.grad.detach().numpy()
    dW_pt = pt_fc.weight.grad.detach().numpy()
    dB_pt = pt_fc.bias.grad.detach().numpy()

    dW_my = my_fc.parameters["W"].grad
    dB_my = my_fc.parameters["b"].grad.squeeze()

    np.testing.assert_allclose(dX_my, dX_pt, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dW_my, dW_pt, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dB_my, dB_pt, rtol=1e-5, atol=1e-6)


class TestFlatten:
    @pytest.mark.parametrize("start_dim, end_dim, shape", [
        (1, -1, (2, 3, 4)),   # flatten [3, 4] to [12]
        (0, -1, (2, 3, 4)),   # flatten all
        (1, 2, (2, 3, 4)),    # flatten [3, 4] to [12]
        (0, 1, (3, 4)),       # flatten first two dims
    ])
    def test_forward_and_backward(self, shape, start_dim, end_dim):
        x_np = np.random.randn(*shape).astype(np.float32)
        x_torch = torch.tensor(deepcopy(x_np), requires_grad=True)

        my_layer = Flatten(start_dim=start_dim, end_dim=end_dim)
        out_np = my_layer.forward(x_np)

        torch_layer = torch.nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        out_torch = torch_layer(x_torch)

        np.testing.assert_allclose(out_np, out_torch.detach().numpy(), atol=1e-6, rtol=1e-5)

        grad_out = np.random.randn(*out_np.shape).astype(np.float32)
        grad_out_torch = torch.tensor(deepcopy(grad_out))

        grad_in_np = my_layer.backward(grad_out)
        out_torch.backward(grad_out_torch)
        grad_in_torch = x_torch.grad.detach().numpy()

        np.testing.assert_allclose(grad_in_np, grad_in_torch, atol=1e-6, rtol=1e-5)


class TestBatchNorm2D:
    @pytest.mark.parametrize("affine, track_running_stats, training", [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ])
    def test_batchnorm2d_vs_pytorch(self, affine, track_running_stats, training):
        np.random.seed(42)
        torch.manual_seed(42)

        N, C, H, W = 4, 3, 5, 5
        x_np = np.random.randn(N, C, H, W).astype(np.float32)
        x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        # PyTorch BN
        bn_torch = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1, affine=affine, track_running_stats=track_running_stats)
        bn_torch.train(training)
        out_torch = bn_torch(x_torch)

        # Our BN
        bn_np = BatchNorm2D(C, eps=1e-5, momentum=0.1, affine=affine, track_running_stats=track_running_stats)
        bn_np.trainable = training
        out_np = bn_np.forward(x_np.copy())

        # Compare outputs
        np.testing.assert_allclose(out_np, out_torch.detach().numpy(), rtol=1e-4, atol=1e-4)

        # Compare running stats
        if track_running_stats:
            np.testing.assert_allclose(bn_np.running_mean, bn_torch.running_mean.detach(
            ).numpy().reshape(1, C, 1, 1), rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(bn_np.running_var, bn_torch.running_var.detach(
            ).numpy().reshape(1, C, 1, 1), rtol=1e-3, atol=1e-3)

        # Compare affine params if used
        if affine:
            # Simulate param sync
            gamma = torch.nn.Parameter(torch.randn(1, C, 1, 1))
            beta = torch.nn.Parameter(torch.randn(1, C, 1, 1))
            bn_torch.weight.data = gamma.view(C)
            bn_torch.bias.data = beta.view(C)
            bn_np.parameters["gamma"].data = gamma.detach().numpy()
            bn_np.parameters["beta"].data = beta.detach().numpy()

            out_torch = bn_torch(x_torch)
            out_np = bn_np.forward(x_np.copy())

            np.testing.assert_allclose(out_np, out_torch.detach().numpy(), rtol=1e-4, atol=1e-4)

            # Backward check
            dout_np = np.random.randn(*out_np.shape).astype(np.float32)
            out_torch.backward(torch.tensor(dout_np))
            bn_np.backward(dout_np)

            np.testing.assert_allclose(bn_np.parameters["gamma"].grad,
                                       bn_torch.weight.grad.view(1, C, 1, 1).numpy(), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(bn_np.parameters["beta"].grad,
                                       bn_torch.bias.grad.view(1, C, 1, 1).numpy(), rtol=1e-4, atol=1e-4)


class TestBatchNorm1D:
    @pytest.mark.parametrize("affine, track_running_stats, training", [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ])
    @pytest.mark.parametrize("shape", [(8, 4), (8, 4, 10)])  # (N, C) and (N, C, L)
    def test_batchnorm1d_vs_pytorch(self, affine, track_running_stats, training, shape):
        np.random.seed(0)
        torch.manual_seed(0)

        C = shape[1]

        # 初始化 PyTorch BN
        bn_torch = nn.BatchNorm1d(C, eps=1e-5, momentum=0.1,
                                  affine=affine, track_running_stats=track_running_stats)
        bn_torch.train(training)

        # 初始化自定义 BN
        bn_np = BatchNorm1D(C, eps=1e-5, momentum=0.1,
                            affine=affine, track_running_stats=track_running_stats)
        bn_np.trainable = training

        # 设置相同的 affine 参数（如果需要）
        if affine:
            gamma = torch.randn(1, C, *([1] * (len(shape) - 2)))
            beta = torch.randn(1, C, *([1] * (len(shape) - 2)))
            bn_torch.weight.data = gamma.view(C)
            bn_torch.bias.data = beta.view(C)
            bn_np.parameters["gamma"].data = gamma.numpy()
            bn_np.parameters["beta"].data = beta.numpy()

        # forward
        for _ in range(10):
            x_np = np.random.randn(*shape).astype(np.float32)
            x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

            out_torch = bn_torch(x_torch)
            out_np = bn_np.forward(x_np.copy())

        np.testing.assert_allclose(out_np, out_torch.detach().numpy(), rtol=1e-4, atol=1e-4)

        # running stats 检查
        if track_running_stats:
            np.testing.assert_allclose(bn_np.running_mean, bn_torch.running_mean.detach().numpy(), rtol=5e-2, atol=5e-2)
            np.testing.assert_allclose(bn_np.running_var, bn_torch.running_var.detach().numpy(), rtol=5e-2, atol=5e-2)

        # backward（仅在 affine 时检查 gamma/beta 的梯度）
        if affine and training:
            dout_np = np.random.randn(*shape).astype(np.float32)
            dout_torch = torch.tensor(dout_np, dtype=torch.float32)

            out_torch.backward(dout_torch)
            bn_np.backward(dout_np)

            np.testing.assert_allclose(bn_np.parameters["gamma"].grad,
                                       bn_torch.weight.grad.detach().numpy(), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(bn_np.parameters["beta"].grad,
                                       bn_torch.bias.grad.detach().numpy(), rtol=1e-4, atol=1e-4)


class TestAdaptiveAvgPool1d:
    @pytest.mark.parametrize("input_shape, output_size", [
        ((4, 3, 25), 5),  # N, C, L
        ((3, 25), 5),     # C, L
        ((2, 4, 8), 4),
        ((4, 3, 16), 1),
    ])
    def test_adaptive_avg_pool1d_vs_pytorch(self, input_shape, output_size):
        np.random.seed(0)
        torch.manual_seed(0)

        # Create input
        x_np = np.random.randn(*input_shape).astype(np.float32)
        x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

        # Torch pool
        torch_pool = nn.AdaptiveAvgPool1d(output_size)
        if x_np.ndim == 2:
            out_torch = torch_pool(x_torch.unsqueeze(0)).squeeze(0)  # add batch dim
        else:
            out_torch = torch_pool(x_torch)

        # Our pool
        my_pool = AdaptiveAvgPool1d(output_size)
        out_np = my_pool.forward(x_np.copy())

        # Compare output
        np.testing.assert_allclose(out_np, out_torch.detach().numpy(), rtol=1e-4, atol=1e-4)

        # Backward test
        grad_out = np.random.randn(*out_np.shape).astype(np.float32)
        grad_out_torch = torch.tensor(grad_out, dtype=torch.float32)

        out_torch.backward(grad_out_torch)
        grad_np = my_pool.backward(grad_out)

        np.testing.assert_allclose(grad_np, x_torch.grad.numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
