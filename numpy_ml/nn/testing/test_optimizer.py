import numpy as np
import torch
import pytest
from nn.optimizer import SGD as MySGD, Adam as MyAdam  # ✅ 添加你的Adam实现
from torch.optim import SGD as TorchSGD, Adam as TorchAdam


class OptimizerTestCase:
    def __init__(
        self,
        my_optimizer_cls,
        torch_optimizer_cls,
        param_name="W",
        shape=(3, 3),
        lr=0.01,
        seed=42,
        **optimizer_kwargs
    ):
        self.my_optimizer_cls = my_optimizer_cls
        self.torch_optimizer_cls = torch_optimizer_cls
        self.param_name = param_name
        self.shape = shape
        self.lr = lr
        self.seed = seed
        self.optimizer_kwargs = optimizer_kwargs
        self._init_data()

    def _init_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.np_param = self._create_my_param()
        self.torch_param = torch.nn.Parameter(
            torch.tensor(self.np_param.data.copy(), dtype=torch.float32, requires_grad=True)
        )

        self.grad = np.random.randn(*self.shape).astype(np.float32)
        self.torch_grad = torch.tensor(self.grad.copy(), dtype=torch.float32)

    def _create_my_param(self):
        from nn.param import Parameter  # lazy import to avoid circular import
        return Parameter(np.random.randn(*self.shape).astype(np.float32))

    def apply_gradients(self):
        self.np_param.grad = self.grad.copy()
        self.torch_param.grad = self.torch_grad.clone()

        my_optimizer = self.my_optimizer_cls(
            parameters=[{self.param_name: self.np_param}],
            lr=self.lr,
            **self.optimizer_kwargs
        )
        torch_optimizer = self.torch_optimizer_cls([self.torch_param], lr=self.lr, **self.optimizer_kwargs)

        my_optimizer.step()
        torch_optimizer.step()

        return self.np_param.data, self.torch_param.detach().numpy()


class TestOptimizers:
    @pytest.mark.parametrize("shape", [(4, 2), (2, 1), (1, 5)])
    def test_sgd_vs_torch(self, shape):
        case = OptimizerTestCase(
            my_optimizer_cls=MySGD,
            torch_optimizer_cls=TorchSGD,
            shape=shape,
            lr=0.05,
            param_name="W"
        )
        my_data, torch_data = case.apply_gradients()
        np.testing.assert_allclose(my_data, torch_data, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("shape", [(4, 2), (2, 1), (1, 5)])
    def test_adam_vs_torch(self, shape):
        case = OptimizerTestCase(
            my_optimizer_cls=MyAdam,
            torch_optimizer_cls=TorchAdam,
            shape=shape,
            lr=0.001,
            param_name="W",
            betas=(0.9, 0.999),
            eps=1e-8
        )
        my_data, torch_data = case.apply_gradients()
        np.testing.assert_allclose(my_data, torch_data, atol=1e-6, rtol=1e-5)
