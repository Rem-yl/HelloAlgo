import pytest
import numpy as np


@pytest.fixture
def inputs():
    return np.array([
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    ])


@pytest.fixture
def random_inputs():
    return np.random.randn((10, 4))
