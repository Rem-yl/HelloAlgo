import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray, required_grads: bool = True):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data)
        self.required_grads = required_grads

    def zero_grad(self):
        self.grad.fill(0.0)

    def freeze(self):
        self.required_grads = False

    def unfreeze(self):
        self.required_grads = True

    def __repr__(self):
        return f"Parameter(data={self.data}, grad={self.grad})"
