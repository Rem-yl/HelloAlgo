import numpy as np
from utils.misc import logsumexp


class HMM:
    def __init__(self, A=None, B=None, pi=None, eps=None):
        self.eps = np.finfo(float).eps if eps is None else eps

        if pi is not None:
            pi[pi == 0] = self.eps
            self.pi = pi

        self.N = None
        if A is not None:
            self.N = A.shape[0]
            A[A == 0] = self.eps
            self.A = A

        self.M = None
        if B is not None:
            self.M = B.shape[1]
            B[B == 0] = self.eps
            self.B = B

    def _init_param(self):
        if pi is None:
            pi = np.ones(self.N)
            pi = pi / np.sum(pi)
            self.pi = pi

        if A is None:
            A = np.ones((self.N, self.N))
            A = A / np.sum(A, axis=1)[:, None]
            self.A = A

        if B is None:
            B = np.ones((self.N, self.M))
            B = B / np.sum(B, axis=1)[:, None]
            self.B = B

    def _forward(self, O: np.ndarray):
        T = O.shape[0]
        alpha = np.zeros((T, self.N))
        alpha[0, :] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            alpha[t, :] = (alpha[t-1, :] @ self.A) * self.B[:, O[t]]

        return alpha

    def _backward(self, O: np.ndarray):
        T = O.shape[0]
        beta = np.zeros((T, self.N))
        beta[T-1, :] = 1.0
        for t in range(T-2, -1, -1):
            beta[t, :] = (self.A * self.B[:, O[t+1]]) @ beta[t+1, :]

        return beta

    def _E_step(self, O: np.ndarray):
        T = O.shape[0]

        alpha = self._forward(O)
        beta = self._backward(O)

        self.gamma = np.zeros((T, self.N))
        for t in range(T):
            for i in range(self.N):
                tmp = 0.0
                for j in range(self.N):
                    tmp += alpha[t, j] * beta[t, j]

                self.gamma[t, i] = alpha[t, i] * beta[t, i] / tmp

        self.epsilon = np.zeros((T, self.N, self.N))
        for t in range(T-1):
            tmp = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    self.epsilon[t, i, j] = alpha[t, i] * self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]
                    tmp += alpha[t, i] * self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]

            self.epsilon[t, :, :] /= tmp

    def _M_step(self, O: np.ndarray):
        M = self.B.shape[1]
        self.pi = self.gamma[0, :]

        self.A = np.sum(self.epsilon[:-1, :, :], axis=0) / np.sum(self.gamma[:-1, :], axis=0)[:, None]
        self.B = np.zeros((self.N, M))

        for j in range(self.N):
            for k in range(M):
                mask = (O == k)
                self.B[j, k] = np.sum(self.gamma[mask, j]) / np.sum(self.gamma[:, j])

    def train(self, O: np.ndarray, pi=None, epoch=100, verbose=None):
        self._init_param()

        for _ in range(epoch):
            self._E_step(O)
            self._M_step(O)

    def predict(self, O: np.ndarray):
        T = O.shape[0]
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)

        delta[0, :] = self.pi * self.B[:, O[0]]
        psi[0, :] = 0

        for t in range(1, T):
            for j in range(self.N):
                prob = delta[t-1, :] * self.A[:, j]
                psi[t, j] = np.argmax(prob)
                delta[t, j] = np.max(prob) * self.B[j, O[t]]

        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[T-1, :])

        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return path
