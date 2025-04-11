from math import erf
from abc import ABC, abstractmethod

import numpy as np
from ..utils import sigmoid

eps = np.finfo(float).eps


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """ 抽象的前向传播方法 """
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        """ 抽象的反向传播方法 """
        raise NotImplementedError


class Sigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return self.__class__.__name__

    def _check_input(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input type must be numpy array")

        if x.ndim <= 1:
            raise ValueError("Input dim must > 1")

    def forward(self, x: np.ndarray):
        r"""
        Evaluate the logistic sigmoid, :math:`\sigma`, on the elements of input `x`.

        .. math::

            \sigma(x_i) = \frac{1}{1 + e^{-x_i}}
        """
        self._check_input(x)

        return sigmoid(x)

    def backward(self, x: np.ndarray):
        r"""
        Evaluate the first derivative of the logistic sigmoid on the elements of `x`.

        .. math::

            \frac{\partial \sigma}{\partial x_i} = \sigma(x_i) (1 - \sigma(x_i))
        """
        self._check_input(x)

        fn_x = self.forward(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        r"""
        Evaluate the second derivative of the logistic sigmoid on the elements of `x`.

        .. math::

            \frac{\partial^2 \sigma}{\partial x_i^2} =
                \frac{\partial \sigma}{\partial x_i} (1 - 2 \sigma(x_i))
        """
        self._check_input(x)
        fn_x = self.forward(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(ActivationBase):
    def __init__(self):
        super().__init__()
        self.X = None
        self.gradients = {}

    def __str__(self):
        return "ReLU"

    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        self.X = x  # 保存前向输入，用于计算 mask
        return np.maximum(0, x)

    def grad(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.where(x > 0, 1.0, 0.0)

    def backward(self, grad_in: np.ndarray, retain_grads=True):
        if self.X is None:
            raise ValueError("Must call forward before backward")

        mask = (self.X > 0).astype(float)
        grad_out = grad_in * mask  # 链式法则： dL/dx = dL/dy * dy/dx

        if retain_grads:
            self.gradients["x"] = grad_out  # 存的是输入 x 对应的梯度

        return grad_out


class LeakyReLU(ActivationBase):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return f"LeakyReLU(alpha={self.alpha})"

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        _x = x.copy()
        _x[_x < 0] = _x[_x < 0] * self.alpha
        return _x

    def backward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        out = np.ones_like(x)
        out[x < 0] = self.alpha

        return out


class GELU(ActivationBase):
    def __init__(self, approximate: bool = True):
        self.approximate = approximate
        super().__init__()

    def __str__(self):
        return f"GELU(approx={self.approximate})"

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        pi, sqrt, tanh = np.pi, np.sqrt, np.tanh
        if self.approximate:
            return 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x ** 3)))
        else:
            return 0.5 * x * (1 + erf(x / sqrt(2)))

    def backward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        pi, exp, sqrt, tanh = np.pi, np.exp, np.sqrt, np.tanh

        s = x / sqrt(2)
        def erf_prime(x): return (2 / sqrt(pi)) * exp(-(x ** 2))  # noqa: E731

        if self.approximate:
            approx = tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3))
            dx = 0.5 + 0.5 * approx + ((0.5 * x * erf_prime(s)) / sqrt(2))
        else:
            dx = 0.5 + 0.5 * erf(s) + ((0.5 * x * erf_prime(s)) / sqrt(2))
        return dx


class Tanh(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"

    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.tanh(x)

    def backward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return 1 - np.tanh(x) ** 2


class Affine(ActivationBase):
    def __init__(self, slope=1, intercept=0):
        self.slope = slope
        self.intercept = intercept
        super().__init__()

    def __str__(self):
        return f"Affine(a={self.slope}, b={self.intercept})"

    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return self.slope * x + self.intercept

    def backward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.ones_like(x) * self.slope


class Identity(Affine):
    def __init__(self):
        super().__init__(slope=1, intercept=0)

    def __str__(self):
        return "Identity"


class ELU(ActivationBase):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return f"ELU(alpha={self.alpha})"

    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.where(x > 0, x, self.alpha*(np.exp(x) - 1))

    def backward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.where(x > 0, np.ones_like(x), self.alpha*np.exp(x))


class SELU(ActivationBase):
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.elu = ELU(alpha=self.alpha)
        super().__init__()

    def __str__(self):
        return "SELU"

    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return self.scale * self.elu.forward(x)

    def backward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.where(
            x >= 0, np.ones_like(x) * self.scale, np.exp(x) * self.alpha * self.scale
        )


class HardSigmoid(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "HardSigmoid"

    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.clip((0.2 * x) + 0.5, 0.0, 1.0)

    def backward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.where((x >= -2.5) & (x <= 2.5), 0.2, 0)


class SoftPlus(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SoftPlus"

    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return np.log(1 + np.exp(x))

    def backward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return 1.0 / (1.0 + np.exp(-x))


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(-10, 10, 0.01)
    models = [Sigmoid(), ReLU(), LeakyReLU(0.3), GELU(True), Tanh(),
              Affine(1, 1), ELU(), SELU(), HardSigmoid(), SoftPlus()]  # 可扩展更多模型

    # 动态计算子图布局（最多4列）
    n_models = len(models)
    n_cols = min(4, n_models)  # 每排最多4个图
    n_rows = (n_models + n_cols - 1) // n_cols  # 向上取整计算行数

    # 优化画布尺寸（每行高度5，每列宽度2.5，预留边距）
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*2.5 + 1, n_rows*5),  # 动态宽度计算
        dpi=100,
        # sharex='col',  # 同列共享x轴
        # sharey='row'   # 同行共享y轴
    )

    # 统一处理一维/二维ax数组
    axs = axs.flatten()  # 展平为一维数组

    for i, (model, ax) in enumerate(zip(models, axs)):
        y = model(x).squeeze()
        grad = model.backward(x).squeeze()  # 假设backward返回梯度

        # 绘制激活函数曲线（红色主曲线）
        ax.plot(x, y, c="#E74C3C", lw=2, label="s(x)")

        # 绘制梯度曲线（蓝色次曲线，透明度优化）
        ax.plot(x, grad, c="#3498DB", lw=1.5, alpha=0.8, label="s'(x)")

        # 标题优化（带图标和颜色）
        ax.set_title(
            f"{str(model)}",
            fontsize=14,
            pad=12,
            color="#34495E",
            loc='left'
        )

        # 坐标轴美化
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例外置

        # 隐藏空的子图
        if i >= n_models:
            ax.set_axis_off()

    # 智能调整布局（预留顶部标题空间）
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()
