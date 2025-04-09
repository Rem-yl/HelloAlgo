from typing import List, Optional
import numpy as np

#######################################################################
#                                 Utils                               #
#######################################################################


def list2batch(x: List) -> Optional[np.ndarray]:
    """ change list to np.array, shape: [n_samples, *]"""
    if not x:  # 空列表
        return None

    # 1. 统一转换为数组（支持标量/列表/数组）
    elements = [np.asarray(elem) for elem in x]

    # 2. 检测是否为纯标量列表（一维情况）
    is_1d = all(elem.ndim == 0 for elem in elements)
    if is_1d:
        # 标量列表 → 转为 1×N 矩阵
        return np.array([elements], dtype=elements[0].dtype)

    # 3. 多维情况：检查所有元素形状一致性
    base_shape = elements[0].shape
    for elem in elements[1:]:
        if elem.shape != base_shape:
            raise ValueError("Inconsistent dimensions")

    # 4. 构建数组（保留原始维度，外层为批次维度）
    return np.array(elements, dtype=elements[0].dtype)


#######################################################################
#                           Training Utils                            #
#######################################################################


def miniBatch(X: np.ndarray, batch_size=256, shuffle=True):
    if X.ndim < 2:
        raise ValueError("X.dim must >= 2")
    if batch_size <= 0:
        raise ValueError("batch_size must > 0")

    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N / batch_size))

    if shuffle:
        np.random.shuffle(ix)

    def gen():
        for i in range(n_batches):
            yield ix[i * batch_size:(i + 1) * batch_size]

    return gen(), n_batches
