"""Unit tests for DCCP example problems."""

import cvxpy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import dccp
import dccp.problem
from dccp.linearize import linearize


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
        assert dccp.is_dccp(prob)
        result = prob.solve(method="dccp")
        y = prob.var_dict["y"]
        z = prob.var_dict["z"]
        assert y.value is not None
        assert z.value is not None
        assert result is not None

        assert_almost_in(y.value, list(sol))
        assert_almost_in(z.value, list(sol))
        assert_almost_equal(float(result), np.sqrt(2))  # type: ignore

    def test_linearize(self):
        """Test the linearize function."""
        z = cp.Variable((1, 5))
        expr = cp.square(z)
        z.value = np.reshape(np.array([1, 2, 3, 4, 5]), (1, 5))
        lin = linearize(expr)
        assert lin is not None
        assert lin.value is not None
        assert lin.shape == (1, 5)
        assert_almost_equal(lin.value[0], [1, 4, 9, 16, 25])

    # def test_convexify_obj(self):
    #     """Test convexify objective"""
    #     obj = cp.Maximize(cp.sum(cp.square(self.x)))
    #     self.x.value = [1, 1]
    #     obj_conv = convexify_obj(obj)
    #     prob_conv = cp.Problem(obj_conv, [self.x <= -1])
    #     prob_conv.solve()
    #     self.assertAlmostEqual(prob_conv.value, -6)

    #     obj = cp.Minimize(cp.sqrt(self.a))
    #     self.a.value = [1]
    #     obj_conv = convexify_obj(obj)
    #     prob_conv = cp.Problem(obj_conv, cp.sqrt(self.a).domain)
    #     prob_conv.solve()
    #     self.assertAlmostEqual(prob_conv.value, 0.5)

    # def test_convexify_constr(self):
    #     """Test convexify constraint"""
    #     constr = cp.norm(self.x) >= 1
    #     self.x.value = [1, 1]
    #     constr_conv = convexify_constr(constr)
    #     prob_conv = cp.Problem(cp.Minimize(cp.norm(self.x)), [constr_conv[0]])
    #     prob_conv.solve()
    #     self.assertAlmostEqual(prob_conv.value, 1)

    #     constr = cp.sqrt(self.a) <= 1
    #     self.a.value = [1]
    #     constr_conv = convexify_constr(constr)
    #     prob_conv = cp.Problem(cp.Minimize(self.a), [constr_conv[0], constr_conv[1][0]])
    #     prob_conv.solve()
    #     self.assertAlmostEqual(self.a.value[0], 0)

    # def test_vector_constr(self):
    #     """Test DCCP with vector cosntraints."""
    #     prob = cp.Problem(cp.Minimize(self.x[0]), [self.x >= 0])
    #     # doesn't crash with solver params.
    #     result = prob.solve(method="dccp", verbose=True)
    #     self.assertAlmostEqual(result[0], 0)
    #     self.assertAlmostEqual(result[0], 0)
