""" 存放一些数值计算的函数 """
import numpy as np


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: np.ndarray):
    return np.maximum(0.0, x)
