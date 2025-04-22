import numpy as np
import pytest
import torch
import torch.nn.functional as F


from nn.loss import BCEWithLogitLoss, BCELoss, MSELoss, L1Loss, CrossEntropyLoss


class TestBCEWithLogitLoss:
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


class TestL1Loss:
    @pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
    def test_l1loss_match_pytorch(self, reduction):
        np.random.seed(0)
        torch.manual_seed(0)

        # 生成随机的预测值和标签
        preds_np = np.random.rand(8).astype(np.float32)
        y_np = np.random.rand(8).astype(np.float32)

        preds_torch = torch.tensor(preds_np, requires_grad=True)
        y_torch = torch.tensor(y_np)

        # 创建自定义 L1Loss
        loss_fn = L1Loss(reduction=reduction)

        # 计算 NumPy 和自定义 L1Loss 的损失
        np_loss = loss_fn.forward(preds_np, y_np)
        np_grad = loss_fn.backward()

        # 使用 PyTorch 的 L1Loss 计算损失
        loss_torch = F.l1_loss(preds_torch, y_torch, reduction=reduction)
        if reduction != "none":
            loss_val = loss_torch.item()
        else:
            loss_val = loss_torch.detach().numpy()

        # 反向传播
        preds_torch.grad = None
        loss_torch.backward(torch.ones_like(loss_torch) if reduction == "none" else None)
        torch_grad = preds_torch.grad.detach().numpy()

        # 比较损失
        if isinstance(np_loss, np.ndarray):
            assert np.allclose(np_loss, loss_val, atol=1e-5), f"Loss mismatch: {np_loss} vs {loss_val}"
        else:
            assert abs(np_loss - loss_val) < 1e-5, f"Loss mismatch: {np_loss} vs {loss_val}"

        # 比较梯度
        assert np.allclose(np_grad, torch_grad, atol=1e-5), f"Grad mismatch\nCustom: {np_grad}\nTorch: {torch_grad}"


class TestL2Loss:
    @pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
    def test_l2loss_match_pytorch(self, reduction):
        np.random.seed(0)
        torch.manual_seed(0)

        # 生成随机的预测值和标签
        preds_np = np.random.rand(8).astype(np.float32)
        y_np = np.random.rand(8).astype(np.float32)

        preds_torch = torch.tensor(preds_np, requires_grad=True)
        y_torch = torch.tensor(y_np)

        # 创建自定义 L2Loss
        loss_fn = MSELoss(reduction=reduction)

        # 计算 NumPy 和自定义 L2Loss 的损失
        np_loss = loss_fn.forward(preds_np, y_np)
        np_grad = loss_fn.backward()

        # 使用 PyTorch 的 MSELoss 计算损失
        loss_torch = F.mse_loss(preds_torch, y_torch, reduction=reduction)
        if reduction != "none":
            loss_val = loss_torch.item()
        else:
            loss_val = loss_torch.detach().numpy()

        # 反向传播
        preds_torch.grad = None
        loss_torch.backward(torch.ones_like(loss_torch) if reduction == "none" else None)
        torch_grad = preds_torch.grad.detach().numpy()

        # 比较损失
        if isinstance(np_loss, np.ndarray):
            assert np.allclose(np_loss, loss_val, atol=1e-5), f"Loss mismatch: {np_loss} vs {loss_val}"
        else:
            assert abs(np_loss - loss_val) < 1e-5, f"Loss mismatch: {np_loss} vs {loss_val}"

        # 比较梯度
        assert np.allclose(np_grad, torch_grad, atol=1e-5), f"Grad mismatch\nCustom: {np_grad}\nTorch: {torch_grad}"


class TestCrossEntropyLoss:
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.5])
    @pytest.mark.parametrize("use_ignore_index", [True, False])
    def test_cross_entropy_match_pytorch(self, reduction, label_smoothing, use_ignore_index):
        np.random.seed(42)
        torch.manual_seed(42)

        batch_size = 6
        num_classes = 5

        # 随机生成 logits 和 targets
        logits_np = np.random.randn(batch_size, num_classes).astype(np.float32)
        targets_np = np.random.randint(0, num_classes, size=batch_size)

        # 设置 ignore_index
        ignore_index = 2 if use_ignore_index else None
        if ignore_index is not None:
            targets_np[1] = ignore_index  # 设置一个标签为 ignore_index

        # === 自定义 CrossEntropyLoss ===
        custom_loss_fn = CrossEntropyLoss(
            reduction=reduction,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

        np_loss = custom_loss_fn.forward(logits_np, targets_np, retain_derived=True)
        np_grad = custom_loss_fn.backward()

        # === PyTorch ===
        logits_torch = torch.tensor(logits_np, requires_grad=True)
        targets_torch = torch.tensor(targets_np, dtype=torch.long)

        torch_loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index if use_ignore_index else -100,  # 默认值
            reduction=reduction,
            label_smoothing=label_smoothing
        )
        loss_torch = torch_loss_fn(logits_torch, targets_torch)
        if reduction != "none":
            loss_val = loss_torch.item()
        else:
            loss_val = loss_torch.detach().numpy()

        # 反向传播
        logits_torch.grad = None
        loss_torch.backward(torch.ones_like(loss_torch) if reduction == "none" else None)
        torch_grad = logits_torch.grad.detach().numpy()

        # 比较损失
        if isinstance(np_loss, np.ndarray):
            assert np.allclose(np_loss, loss_val, atol=1e-5), f"Loss mismatch: {np_loss} vs {loss_val}"
        else:
            assert abs(np_loss - loss_val) < 1e-5, f"Loss mismatch: {np_loss} vs {loss_val}"

        # 比较梯度
        assert np.allclose(np_grad, torch_grad, atol=1e-5), f"Grad mismatch\nCustom: {np_grad}\nTorch: {torch_grad}"


if __name__ == "__main__":
    pytest.main(["-v", "nn/testing/test_loss.py"])
