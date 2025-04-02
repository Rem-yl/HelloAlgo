from nn.utils import *


def test_miniBatch():
    X1 = np.zeros((1000, 10))
    X2 = np.zeros((1000, 10))

    # 设置相同随机种子
    np.random.seed(42)
    gen1, n_batches = miniBatch(X1, shuffle=True)
    shapes = []
    for data in gen1:
        shapes.append(data.shape[0])

    assert shapes == [256, 256, 256, 232]
