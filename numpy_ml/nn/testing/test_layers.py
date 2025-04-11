import numpy as np
import torch
import torch.nn as nn
import pytest

from nn.layers import Linear  # 根据你的实际导入路径调整


@pytest.mark.parametrize("batch_size, in_feat, out_feat", [
    (4, 3, 2),
    (10, 5, 7),
    (1, 6, 4)
])
def test_linear(batch_size, in_feat, out_feat):
    # 设置种子以保证一致性
    np.random.seed(42)
    torch.manual_seed(42)

    # 输入张量
    X_np = np.random.randn(batch_size, in_feat).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)

    # 自定义 Linear
    my_fc = Linear(in_feat, out_feat)

    # PyTorch Linear
    pt_fc = nn.Linear(in_feat, out_feat, bias=True)
    pt_fc.weight.data = torch.tensor(my_fc.parameters["W"].data, dtype=torch.float32)
    pt_fc.bias.data = torch.tensor(my_fc.parameters["b"].data.squeeze(), dtype=torch.float32)
    np.testing.assert_allclose(my_fc.parameters["W"].data, pt_fc.weight.data.detach().numpy(),  rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(my_fc.parameters["b"].data.squeeze(),
                               pt_fc.bias.data.detach().numpy(),  rtol=1e-5, atol=1e-6)

    # 前向传播
    out_my = my_fc.forward(X_np)
    out_pt = pt_fc(X_torch)

    # 检查前向传播输出是否一致
    np.testing.assert_allclose(out_my, out_pt.detach().numpy(), rtol=1e-5, atol=1e-6)

    # 构造 dL/dy 模拟上游梯度
    dLdy_np = np.random.randn(*out_my.shape).astype(np.float32)
    dLdy_pt = torch.tensor(dLdy_np)

    # 后向传播
    dX_my = my_fc.backward(dLdy_np)
    out_pt.backward(dLdy_pt)

    # 取出 PyTorch 的梯度
    dX_pt = X_torch.grad.detach().numpy()
    dW_pt = pt_fc.weight.grad.detach().numpy()
    dB_pt = pt_fc.bias.grad.detach().numpy()

    # 自定义模型的梯度
    dW_my = my_fc.parameters["W"].grad
    dB_my = my_fc.parameters["b"].grad.squeeze()

    # 检查梯度是否一致
    np.testing.assert_allclose(dX_my, dX_pt, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dW_my, dW_pt, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dB_my, dB_pt, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_linear(32, 10, 2)
