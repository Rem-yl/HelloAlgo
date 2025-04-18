import numpy as np
import pytest
import torch
import torch.nn.functional as F


from nn.loss import BCEWithLogitLoss, BCELoss


class Test_BCEWithLogitLoss:
    @pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
    @pytest.mark.parametrize("use_pos_weight", [False, True])
    @pytest.mark.parametrize("use_weight", [False, True])
    def test_bcewithlogitsloss_match_pytorch(self, reduction, use_pos_weight, use_weight):
        np.random.seed(0)
        torch.manual_seed(0)

        logits_np = np.random.randn(8).astype(np.float32)
        y_np = np.random.randint(0, 2, size=8).astype(np.float32)
        weight_np = np.random.rand(8).astype(np.float32) if use_weight else None
        pos_weight_val = 2.0 if use_pos_weight else None

        # PyTorch tensors
        logits_torch = torch.tensor(logits_np, requires_grad=True)
        y_torch = torch.tensor(y_np)
        if use_weight:
            weight_torch = torch.tensor(weight_np)
        else:
            weight_torch = None
        if use_pos_weight:
            pos_weight_torch = torch.tensor([pos_weight_val])
        else:
            pos_weight_torch = None

        # 自定义类
        loss_fn = BCEWithLogitLoss(weight=weight_np, pos_weight=pos_weight_val, reduction=reduction)
        np_loss = loss_fn.forward(logits_np, y_np)
        np_grad = loss_fn.backward()

        # PyTorch 计算 loss 和 grad
        loss_torch = F.binary_cross_entropy_with_logits(
            logits_torch, y_torch,
            weight=weight_torch,
            pos_weight=pos_weight_torch,
            reduction=reduction
        )
        if reduction != "none":
            loss_val = loss_torch.item()
        else:
            loss_val = loss_torch.detach().numpy()

        # 反向传播
        logits_torch.grad = None
        loss_torch.backward(torch.ones_like(loss_torch) if reduction == "none" else None)
        torch_grad = logits_torch.grad.detach().numpy()

        # 断言 loss 相近
        if isinstance(np_loss, np.ndarray):
            assert np.allclose(np_loss, loss_val, atol=1e-5), f"Loss mismatch: {np_loss} vs {loss_val}"
        else:
            assert abs(np_loss - loss_val) < 1e-5, f"Loss mismatch: {np_loss} vs {loss_val}"

        # 断言 grad 相近
        assert np.allclose(np_grad, torch_grad, atol=1e-5), f"Grad mismatch\nCustom: {np_grad}\nTorch: {torch_grad}"


class TestBCELoss:
    @pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
    @pytest.mark.parametrize("use_weight", [False, True])
    def test_bceloss_match_pytorch(self, reduction, use_weight):
        np.random.seed(0)
        torch.manual_seed(0)

        probs_np = np.random.rand(8).astype(np.float32)
        y_np = np.random.randint(0, 2, size=8).astype(np.float32)
        weight_np = np.random.rand(8).astype(np.float32) if use_weight else None

        probs_torch = torch.tensor(probs_np, requires_grad=True)
        y_torch = torch.tensor(y_np)
        weight_torch = torch.tensor(weight_np) if use_weight else None

        loss_fn = BCELoss(weight=weight_np, reduction=reduction)
        np_loss = loss_fn.forward(probs_np, y_np)
        np_grad = loss_fn.backward()

        loss_torch = F.binary_cross_entropy(
            probs_torch, y_torch,
            weight=weight_torch,
            reduction=reduction
        )
        if reduction != "none":
            loss_val = loss_torch.item()
        else:
            loss_val = loss_torch.detach().numpy()

        probs_torch.grad = None
        loss_torch.backward(torch.ones_like(loss_torch) if reduction == "none" else None)
        torch_grad = probs_torch.grad.detach().numpy()

        if isinstance(np_loss, np.ndarray):
            assert np.allclose(np_loss, loss_val, atol=1e-5), f"Loss mismatch: {np_loss} vs {loss_val}"
        else:
            assert abs(np_loss - loss_val) < 1e-5, f"Loss mismatch: {np_loss} vs {loss_val}"

        assert np.allclose(np_grad, torch_grad, atol=1e-5), f"Grad mismatch\nCustom: {np_grad}\nTorch: {torch_grad}"
