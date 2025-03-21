import numpy as np
from hmm import HMM


def test1():
    # 《统计学习方法》例题10.2

    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])

    O = np.array([0, 1, 0])

    T = 3
    N = A.shape[0]
    M = B.shape[1]

    alpha = np.zeros((T, N))
    # 1. 计算初值
    for i in range(N):
        alpha[0, i] = pi[i] * B[i, O[0]]

    # 2. 递推
    for t in range(1, T):
        for i in range(N):
            tmp = 0.0
            for j in range(N):
                tmp += alpha[t-1, j] * A[j, i]

            alpha[t, i] = tmp * B[i, O[t]]

    alpha = np.zeros((T, N))  # 存储前向概率

    # 1. 初始化
    alpha[0, :] = pi * B[:, O[0]]

    # 2. 递推
    for t in range(1, T):
        alpha[t, :] = (alpha[t-1, :] @ A) * B[:, O[t]]

    res = np.sum(alpha[-1, :])
    print(res)

    beta = np.zeros((T, N))

    beta[T-1, :] = 1.0
    for t in range(T-2, -1, -1):
        for i in range(N):
            tmp = 0.0
            for j in range(N):
                tmp += A[i, j] * B[j, O[t+1]] * beta[t+1, j]

            beta[t, i] = tmp

    res = 0.0
    for i in range(N):
        res += pi[i] * B[i, O[0]] * beta[0, i]

    print(res)

    beta = np.zeros((T, N))

    beta[T-1, :] = 1.0
    for t in range(T-2, -1, -1):
        beta[t, :] = (A * B[:, O[t+1]]) @ beta[t+1, :]

    res = np.sum(pi * B[:, O[0]] * beta[0, :])
    print(res)


test1()
