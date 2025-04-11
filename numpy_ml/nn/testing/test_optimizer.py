import numpy as np
import pytest
import torch

from nn.optimizer import SGD as MySGD


class TestSGD:
    def test_sgd(self):
        # 初始化参数与梯度
        param_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad_np = np.array([[0.1, 0.1], [0.1, 0.1]])

        param_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        grad_torch = torch.tensor([[0.1, 0.1], [0.1, 0.1]])

        # 超参数
        lr = 0.01
        momentum = 0.9
        clip_norm = None

        # 自定义 SGD 更新
        my_sgd = MySGD(lr=lr, momentum=momentum, clip_norm=clip_norm)
        updated_np = my_sgd.update(param_np.copy(), grad_np, "param")

        # PyTorch SGD 更新
        optimizer = torch.optim.SGD([param_torch], lr=lr, momentum=momentum)
        param_torch.grad = grad_torch
        optimizer.step()
        updated_torch = param_torch.detach().numpy()

        # 断言二者相近
        np.testing.assert_allclose(updated_np, updated_torch, atol=1e-6)

    def test_sgd_multiple_params(self):
        # 超参数
        lr = 0.01
        momentum = 0.9
        clip_norm = None

        # 初始化参数与梯度（模拟两个参数，带 batch）
        param_dict_np = {
            "W": np.random.randn(32, 64),
            "b": np.random.randn(1, 64)
        }
        grad_dict_np = {
            "W": np.random.randn(32, 64),
            "b": np.random.randn(1, 64)
        }

        # 转换为 torch tensor
        param_dict_torch = {
            k: torch.tensor(v.copy(), requires_grad=True)
            for k, v in param_dict_np.items()
        }

        # 设置梯度
        for k in param_dict_torch:
            param_dict_torch[k].grad = torch.tensor(grad_dict_np[k].copy())

        # 初始化优化器
        my_sgd = MySGD(lr=lr, momentum=momentum, clip_norm=clip_norm)
        torch_optimizer = torch.optim.SGD(param_dict_torch.values(), lr=lr, momentum=momentum)

        # 自定义优化器更新
        updated_np = {}
        for k in param_dict_np:
            updated_np[k] = my_sgd.update(param_dict_np[k], grad_dict_np[k], k)

        # PyTorch 更新
        torch_optimizer.step()

        # 对比结果
        for k in param_dict_np:
            np.testing.assert_allclose(
                updated_np[k],
                param_dict_torch[k].detach().numpy(),
                atol=1e-6,
                err_msg=f"Mismatch in parameter: {k}"
            )
