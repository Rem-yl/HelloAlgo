from __future__ import annotations  # 启用新的类型提示解析
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(0.7 * torch.randn(self.out_dim, self.in_dim), requires_grad=True)
        self.b = nn.Parameter(0.3 * torch.randn(1, self.out_dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return x @ self.w.T + self.b


# 测试代码
def test_linear():
    torch.manual_seed(42)  # 设置随机种子以保证结果可复现

    # 定义输入
    batch_size = 4
    in_dim = 3
    out_dim = 2
    x = torch.randn(batch_size, in_dim)  # 形状 (4, 3)

    # 初始化 Linear 层
    model = Linear(in_dim, out_dim)

    out = model(x)


if __name__ == "__main__":
    test_linear()
