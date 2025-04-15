import numpy as np


def constant(x: np.ndarray, val: float):
    x[:] = val


def normal(x: np.ndarray, mean=0.0, std=1.0):
    x[:] = np.random.normal(loc=mean, scale=std, size=x.shape)


def uniform(x: np.ndarray, a=0.0, b=1.0):
    x[:] = np.random.uniform(low=a, high=b, size=x.shape)


def xavier_uniform(x: np.ndarray):
    fan_in, fan_out = x.shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    x[:] = np.random.uniform(-limit, limit, size=x.shape)


def xavier_normal(x: np.ndarray):
    fan_in, fan_out = x.shape
    std = np.sqrt(2.0 / (fan_in + fan_out))
    x[:] = np.random.normal(0.0, std, size=x.shape)


def kaiming_uniform(x: np.ndarray, a=0, mode='fan_in'):
    fan = x.shape[1] if mode == 'fan_in' else x.shape[0]
    bound = np.sqrt(6.0 / ((1 + a ** 2) * fan))
    x[:] = np.random.uniform(-bound, bound, size=x.shape)


def kaiming_normal(x: np.ndarray, a=0, mode='fan_in'):
    fan = x.shape[1] if mode == 'fan_in' else x.shape[0]
    std = np.sqrt(2.0 / ((1 + a ** 2) * fan))
    x[:] = np.random.normal(0.0, std, size=x.shape)
