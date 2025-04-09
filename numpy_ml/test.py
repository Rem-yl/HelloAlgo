import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.layers import FullyConnected
# --- 你的 FullyConnected 层
# 假设你已经 import FullyConnected 进来了

# 1. 准备输入和梯度
np.random.seed(0)
torch.manual_seed(0)

X_np = np.random.randn(4, 5).astype(np.float32)  # batch_size=4, in_dim=5
dLdy_np = np.ones((4, 3), dtype=np.float32)      # assume output dim=3

# 2. 初始化你的层
my_fc = FullyConnected(n_out=3, act_fn="relu")
out_my = my_fc.forward(X_np)  # forward，内部自动初始化参数

# 3. 用 PyTorch 初始化线性层
pt_fc = nn.Linear(5, 3).float()
pt_fc.weight.data = torch.tensor(my_fc.parameters["W"].T, dtype=torch.float32)
pt_fc.bias.data = torch.tensor(my_fc.parameters["b"].reshape(-1), dtype=torch.float32)

# forward
X_pt = torch.tensor(X_np, requires_grad=True, dtype=torch.float32)
dLdy_pt = torch.tensor(dLdy_np, dtype=torch.float32)
out_pt = F.relu(pt_fc(X_pt))
out_pt.backward(torch.tensor(dLdy_np), retain_graph=True)

# compare forward
print("✅ Forward diff:", np.abs(out_my - out_pt.detach().numpy()).max())

# 4. backward
# PyTorch
out_pt.backward(torch.tensor(dLdy_np))
dX_pt = X_pt.grad.detach().numpy()
dW_pt = pt_fc.weight.grad.detach().numpy()
dB_pt = pt_fc.bias.grad.detach().numpy()

# your layer backward
dX_my = my_fc.backward(dLdy_np)
dW_my = my_fc.gradients["W"]
dB_my = my_fc.gradients["b"]

# 5. compare backward
print("✅ dX diff:", np.abs(dX_my - dX_pt).max())
print("✅ dW diff:", np.abs(dW_my - dW_pt.T).max())  # transpose for alignment
print("✅ dB diff:", np.abs(dB_my - dB_pt.reshape(1, -1)).max())
