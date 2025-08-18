"""Unit tests for DCCP example problems."""

import cvxpy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from dccp import convexify_constr, convexify_obj, is_dccp, linearize


def assert_almost_equal(
    a: float | np.ndarray, b: float | np.ndarray, rtol: float = 1e-6, atol: float = 1e-6
) -> None:
    """Assert that two arrays are almost equal."""
    assert_allclose(
        np.asarray(a), b, rtol=rtol, atol=atol, err_msg="Arrays are not almost equal."
    )


# a must be almost equal to one of the items in b
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


class TestExample:
    """Unit tests example."""

    @pytest.fixture
    def prob(self) -> cp.Problem:
        """Fixture to create a problem."""
        y = cp.Variable(2, name="y")
        z = cp.Variable(2, name="z")
        return cp.Problem(
            cp.Maximize(cp.norm(y - z, 2)),
            [y >= 0, y <= 1, z >= 0, z <= 1],
        )

    def test_readme_example(self, prob: cp.Problem) -> None:
        """Test the example in the readme.

        self.sol - All known possible solutions to the problem in the readme.
        """
        sol = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        assert not prob.is_dcp()
        assert is_dccp(prob)
        result = prob.solve(method="dccp")
        y = prob.var_dict["y"]
        z = prob.var_dict["z"]
        assert y.value is not None
        assert z.value is not None
        assert result is not None

        assert_almost_in(y.value, list(sol))
        assert_almost_in(z.value, list(sol))
        assert_almost_equal(float(result), np.sqrt(2))  # type: ignore

    def test_linearize(self) -> None:
        """Test the linearize function."""
        z = cp.Variable((1, 5))
        expr = cp.square(z)
        z.value = np.reshape(np.array([1, 2, 3, 4, 5]), (1, 5))
        lin = linearize(expr)
        assert lin is not None
        assert lin.value is not None
        assert lin.shape == (1, 5)
        assert_almost_equal(lin.value[0], np.array([1, 4, 9, 16, 25]))

    def test_convexify_obj(self) -> None:
        """Test convexify objective."""
        x = cp.Variable(2)
        x.value = [1, 1]
        obj = cp.Maximize(cp.sum(cp.square(x)))
        obj_conv = convexify_obj(obj)
        prob_conv = cp.Problem(obj_conv, [x <= -1])
        prob_conv.solve()
        assert prob_conv.status == cp.OPTIMAL
        assert prob_conv.value is not None
        assert_almost_equal(float(prob_conv.value), -6)  # type: ignore

        a = cp.Variable(1, value=[1])
        obj = cp.Minimize(cp.sqrt(a))
        obj_conv = convexify_obj(obj)
        assert obj_conv is not None
        prob_conv = cp.Problem(obj_conv, cp.sqrt(a).domain)
        prob_conv.solve()
        assert prob_conv.status == cp.OPTIMAL
        assert prob_conv.value is not None
        assert_almost_equal(float(prob_conv.value), 0.5)  # type: ignore

    def test_convexify_constr(self) -> None:
        """Test convexify constraint."""
        x = cp.Variable(2)
        a = cp.Variable(1)
        constr = cp.norm(x) >= 1
        x.value = [1, 1]
        constr_conv = convexify_constr(constr)
        prob_conv = cp.Problem(cp.Minimize(cp.norm(x)), [constr_conv.constr])
        prob_conv.solve()
        assert prob_conv.status == cp.OPTIMAL
        assert prob_conv.value is not None
        assert_almost_equal(float(prob_conv.value), 1)  # type: ignore

        constr = cp.sqrt(a) <= 1
        a.value = [1]
        constr_conv = convexify_constr(constr)
        prob_conv = cp.Problem(
            cp.Minimize(a), [constr_conv.constr, *constr_conv.domain]
        )
        prob_conv.solve()
        assert prob_conv.status == cp.OPTIMAL
        assert a.value is not None
        assert_almost_equal(float(a.value[0]), 0)  # type:

    def test_vector_constr(self) -> None:
        """Test DCCP with vector constraints."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(x[0]), [x >= 0])
        result = prob.solve(method="dccp", verbose=True)
        assert prob.status == cp.OPTIMAL
        assert result is not None
        assert x.value is not None
        assert_almost_equal(float(result), 0)  # type: ignore
        assert_almost_equal(x.value[0], 0)  # type: ignore

    def test_circle_packing(self) -> None:
        """Test the circle packing example."""
        n = 10
        r = np.linspace(1, 5, n)

        c = cp.Variable((n, 2))
        constr = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                constr += [cp.norm(c[i, :] - c[j, :]) >= r[i] + r[j]]
        prob_cp = cp.Problem(cp.Minimize(cp.max(cp.max(cp.abs(c), axis=1) + r)), constr)
        assert not prob_cp.is_dcp()
        assert is_dccp(prob_cp)
        prob_cp.solve(method="dccp", solver="ECOS", ep=1e-3, max_slack=1e-3, seed=0)
        assert prob_cp.status == cp.OPTIMAL

        le = cp.max(cp.max(cp.abs(c), axis=1) + r).value * 2  # type: ignore
        ratio = np.pi * cp.sum(cp.square(r)).value / cp.square(le).value  # type: ignore
        assert ratio > 0.68  # the ratio should be greater than 0.68 for a valid packing
