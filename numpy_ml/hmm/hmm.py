import numpy as np
from utils.misc import logsumexp


class HMM:
    def __init__(self, A=None, B=None, pi=None, eps=None):
        eps = np.finfo(float).eps if eps is None else eps

        if pi is not None:
            pi[pi == 0] = eps

        N = None
        if A is not None:
            N = A.shape[0]
            A[A == 0] = eps

        M = None
        if B is not None:
            M = B.shape[1]
            B[B == 0] = eps

        self.hyper_param = {
            "eps": eps,
        }
        self.param = {
            "A": A,
            "B": B,
            "pi": pi,
        }
        self.N = N
        self.M = M

    def _init_param(self):
        param = self.param
        A, B, pi = param["A"], param["B"], param["pi"]

        # 初识时, 每个隐状态的概率都是均等的
        if pi is None:
            pi = np.ones(self.N)
            pi = pi / np.sum(pi)

        # 初始化时, 隐状态之间的转移概率是均等的
        if A is None:
            A = np.ones((self.N, self.N))   # A的行向量是概率和, 即q_i转移到所有其他状态的概率和
            A = A / np.sum(A, axis=1)[:, None]

        if B is None:
            B = np.random.randn(self.N, self.M)
            B = B / np.sum(B, axis=1)[:, None]

        param["A"], param["B"], param["pi"] = A, B, pi

    def _forward(self, O: np.ndarray):
        param = self.param
        A, B, pi = param["A"], param["B"], param["pi"]

        T = O.shape[0]
        self.alpha = np.zeros((T, self.N))
        self.alpha[0, :] = pi * B[:, O[0]]

        for t in range(1, T):
            self.alpha[t, :] = (self.alpha[t-1, :] @ A) * B[:, O[t]]

    def _backward(self, O: np.ndarray):
        param = self.param
        A, B, pi = param["A"], param["B"], param["pi"]
        T = O.shape[0]
        self.beta = np.zeros((T, self.N))
        self.beta[T-1, :] = 1.0
        for t in range(T-2, -1, -1):
            self.beta[t, :] = (A * B[:, O[t+1]]) @ self.beta[t+1, :]

    def _E_step(self, O: np.ndarray):
        pass
