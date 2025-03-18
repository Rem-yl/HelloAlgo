import numpy as np


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)


def log_gaussian_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    r"f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu)\right)"
    d = len(mu)
    a = d * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma)

    y = np.linalg.solve(sigma, x-mu)    # 等价于求 \sigma^-1 * (x - \mu)
    c = np.dot(x-mu, y)

    return -0.5 * (a + b + c)


def gaussian_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """
    计算 d 维高斯分布的概率密度函数 (PDF)

    参数：
    - x: np.ndarray, 形状 (d,)，表示输入向量
    - mu: np.ndarray, 形状 (d,)，表示均值向量
    - sigma: np.ndarray, 形状 (d, d)，表示协方差矩阵

    返回：
    - float: x 在该高斯分布下的概率密度值
    """
    d = len(mu)  # 维度
    det_sigma = np.linalg.det(sigma)  # 计算协方差矩阵的行列式
    inv_sigma = np.linalg.inv(sigma)  # 计算协方差矩阵的逆矩阵
    norm_const = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_sigma))  # 归一化因子

    # 计算指数部分 (x - mu)^T * Sigma^-1 * (x - mu)
    x_mu = x - mu
    exponent = -0.5 * np.dot(np.dot(x_mu.T, inv_sigma), x_mu)

    return norm_const * np.exp(exponent)
