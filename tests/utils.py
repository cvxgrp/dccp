"""Testing utilities for DCCP."""

import numpy as np
from numpy.testing import assert_allclose


def assert_almost_equal(
    a: float | np.ndarray, b: float | np.ndarray, rtol: float = 1e-6, atol: float = 1e-6
) -> None:
    """Assert that two arrays are almost equal."""
    assert_allclose(
        np.asarray(a), b, rtol=rtol, atol=atol, err_msg="Arrays are not almost equal."
    )


def assert_almost_in(
    a: float | np.ndarray,
    b: list[float | np.ndarray],
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """Assert that a is almost equal to one of the items in b."""
    for item in b:
        try:
            assert_almost_equal(a, item, rtol=rtol, atol=atol)
        except AssertionError:
            continue
        else:
            return
    msg = f"{a} is not almost equal to any item in {b}."
    raise AssertionError(msg)
