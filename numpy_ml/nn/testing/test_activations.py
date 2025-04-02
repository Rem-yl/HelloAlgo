from nn.activations import *

import numpy as np


def test_sigmoid(inputs):
    model = Sigmoid()
    res = np.array(
        [[0.73105858, 0.88079708, 0.95257413],
         [0.98201379, 0.99330715, 0.99752738]]
    )
    assert np.allclose(model(inputs), res)
    res = np.array(
        [[0.19661193, 0.10499359, 0.04517666],
         [0.01766271, 0.00664806, 0.00246651]]
    )
    assert np.allclose(model.backward(inputs), res)


def test_relu(inputs):
    model = ReLU()
    assert np.allclose(inputs, model(inputs))
