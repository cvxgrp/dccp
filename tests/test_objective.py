"""Unit tests for DCCP example problems."""

import cvxpy as cp

from dccp import convexify_obj

from .utils import assert_almost_equal


class TestObjective:
    """Unit test convexification of objective."""

    def test_convexify_pow2(self) -> None:
        """Test convexify objective."""
        x = cp.Variable(2)
        x.value = [1, 1]
        obj = cp.Maximize(cp.sum(cp.square(x)))
        obj_conv = convexify_obj(obj)
        prob_conv = cp.Problem(obj_conv, [x <= -1])  # type: ignore
        prob_conv.solve()
        assert prob_conv.status == cp.OPTIMAL
        assert prob_conv.value is not None
        assert_almost_equal(float(prob_conv.value), 6)  # type: ignore

    def test_convexify_sqrt(self) -> None:
        """Test convexify objective."""
        a = cp.Variable(1, value=[1])
        obj = cp.Minimize(cp.sqrt(a))
        obj_conv = convexify_obj(obj)
        assert obj_conv is not None
        prob_conv = cp.Problem(obj_conv, cp.sqrt(a).domain)
        prob_conv.solve()
        assert prob_conv.status == cp.OPTIMAL
        assert prob_conv.value is not None
        assert_almost_equal(float(prob_conv.value), 0.5)  # type: ignore
