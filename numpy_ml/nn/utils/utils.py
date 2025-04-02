import numpy as np

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
