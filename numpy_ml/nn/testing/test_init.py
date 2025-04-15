import numpy as np
import torch
import torch.nn.init as torch_init
import pytest

from nn.init import (
    constant, normal, uniform,
    xavier_uniform, xavier_normal,
    kaiming_uniform, kaiming_normal
)

shape = (1000, 500)
RTOL = 1e-2  # 相对误差容许范围
ATOL = 1e-2  # 绝对误差容许范围


def stats(arr):
    return arr.mean(), arr.std()


@pytest.mark.parametrize("val", [0.42, -1.0, 3.14])
def test_constant(val):
    torch_tensor = torch.empty(shape)
    torch_init.constant_(torch_tensor, val)

    np_tensor = np.empty(shape)
    constant(np_tensor, val)

    mean_t, std_t = stats(torch_tensor.numpy())
    mean_np, std_np = stats(np_tensor)

    assert np.isclose(mean_np, mean_t, rtol=RTOL, atol=ATOL)
    assert np.isclose(std_np, std_t, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mean, std", [(0.0, 1.0), (2.0, 0.5)])
def test_normal(mean, std):
    torch_tensor = torch.empty(shape)
    torch_init.normal_(torch_tensor, mean=mean, std=std)

    np_tensor = np.empty(shape)
    normal(np_tensor, mean=mean, std=std)

    mean_t, std_t = stats(torch_tensor.numpy())
    mean_np, std_np = stats(np_tensor)

    assert np.isclose(mean_np, mean_t, rtol=RTOL, atol=ATOL)
    assert np.isclose(std_np, std_t, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("a, b", [(-1.0, 1.0), (0.0, 2.0)])
def test_uniform(a, b):
    torch_tensor = torch.empty(shape)
    torch_init.uniform_(torch_tensor, a=a, b=b)

    np_tensor = np.empty(shape)
    uniform(np_tensor, a=a, b=b)

    mean_t, std_t = stats(torch_tensor.numpy())
    mean_np, std_np = stats(np_tensor)

    assert np.isclose(mean_np, mean_t, rtol=RTOL, atol=ATOL)
    assert np.isclose(std_np, std_t, rtol=RTOL, atol=ATOL)


def test_xavier_uniform():
    torch_tensor = torch.empty(shape)
    torch_init.xavier_uniform_(torch_tensor)

    np_tensor = np.empty(shape)
    xavier_uniform(np_tensor)

    mean_t, std_t = stats(torch_tensor.numpy())
    mean_np, std_np = stats(np_tensor)

    assert np.isclose(mean_np, mean_t, rtol=RTOL, atol=ATOL)
    assert np.isclose(std_np, std_t, rtol=RTOL, atol=ATOL)


def test_xavier_normal():
    torch_tensor = torch.empty(shape)
    torch_init.xavier_normal_(torch_tensor)

    np_tensor = np.empty(shape)
    xavier_normal(np_tensor)

    mean_t, std_t = stats(torch_tensor.numpy())
    mean_np, std_np = stats(np_tensor)

    assert np.isclose(mean_np, mean_t, rtol=RTOL, atol=ATOL)
    assert np.isclose(std_np, std_t, rtol=RTOL, atol=ATOL)


def test_kaiming_uniform():
    torch_tensor = torch.empty(shape)
    torch_init.kaiming_uniform_(torch_tensor, a=0, mode='fan_in', nonlinearity='relu')

    np_tensor = np.empty(shape)
    kaiming_uniform(np_tensor, a=0, mode='fan_in')

    mean_t, std_t = stats(torch_tensor.numpy())
    mean_np, std_np = stats(np_tensor)

    assert np.isclose(mean_np, mean_t, rtol=RTOL, atol=ATOL)
    assert np.isclose(std_np, std_t, rtol=RTOL, atol=ATOL)


def test_kaiming_normal():
    torch_tensor = torch.empty(shape)
    torch_init.kaiming_normal_(torch_tensor, a=0, mode='fan_in', nonlinearity='relu')

    np_tensor = np.empty(shape)
    kaiming_normal(np_tensor, a=0, mode='fan_in')

    mean_t, std_t = stats(torch_tensor.numpy())
    mean_np, std_np = stats(np_tensor)

    assert np.isclose(mean_np, mean_t, rtol=RTOL, atol=ATOL)
    assert np.isclose(std_np, std_t, rtol=RTOL, atol=ATOL)
