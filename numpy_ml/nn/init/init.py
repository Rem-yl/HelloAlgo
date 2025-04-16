import numpy as np
import math


def uniform_(x: np.ndarray, a=0.0, b=1.0):
    x[:] = np.random.uniform(low=a, high=b, size=x.shape)


def normal_(x: np.ndarray, mean=0.0, std=1.0):
    x[:] = np.random.normal(loc=mean, scale=std, size=x.shape)


def trunc_normal_(x: np.ndarray, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    r"""Fill the input Tensor with values drawn from a truncated normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        x: an n-dimensional `np.array`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = np.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    if x.ndim <= 0:
        raise ValueError("Input must be at least 1D")

    size = x.shape
    num_attempts = 0
    max_attempts = 10000
    n = np.prod(size)
    samples = np.empty(n, dtype=x.dtype)
    filled = 0

    while filled < n:
        num_attempts += 1
        if num_attempts > max_attempts:
            raise RuntimeError("Too many attempts to generate truncated normal values")

        sample = np.random.normal(loc=mean, scale=std, size=2 * (n-filled))  # 每次生成2倍缺口的随机数
        accept = sample[(sample >= a) & (sample <= b)][:n-filled]  # 防止溢出
        samples[filled:filled + accept.shape[0]] = accept
        filled += accept.shape[0]

    x[:] = samples.reshape(size)


def constant_(x: np.ndarray, val: float):
    x[:] = val


def ones_(x: np.ndarray):
    x[:] = np.ones_like(x)


def zeros_(x: np.ndarray):
    x[:] = np.zeros_like(x)


def eye_(x: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Input shape must be 2!")

    x[:] = np.eye(x.shape[0], x.shape[1])


def calculate_gain(nonlinearity: str, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalization
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    nonlinearity_ori = nonlinearity
    nonlinearity = nonlinearity.lower()
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity_ori}")


def _calculate_fan_in_and_fan_out(x: np.ndarray):
    dimensions = x.ndim
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for np.array with fewer than 2 dimensions"
        )

    num_input_fmaps = x.shape[1]
    num_output_fmaps = x.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        for s in x.shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def lecun_normal_(x: np.ndarray):
    fan_in, _ = _calculate_fan_in_and_fan_out(x)
    std = np.sqrt(1.0 / fan_in)
    x[:] = np.random.normal(0.0, std, size=x.shape)


def xavier_uniform_(x: np.ndarray, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    x[:] = np.random.uniform(-limit, limit, size=x.shape)


def xavier_normal_(x: np.ndarray, gain=1.0):
    fan_in, fan_out = x.shape
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    x[:] = np.random.normal(0.0, std, size=x.shape)


def _calculate_correct_fan(x: np.ndarray, mode: str):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(x: np.ndarray, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_correct_fan(x, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std
    x[:] = np.random.uniform(-bound, bound, size=x.shape)


def kaiming_normal_(x: np.ndarray, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_correct_fan(x, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    x[:] = np.random.normal(loc=0.0, scale=std, size=x.shape)
