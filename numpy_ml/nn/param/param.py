import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        self.grad.fill(0.0)

    def __repr__(self):
        return f"Parameter(data={self.data}, grad={self.grad})"
