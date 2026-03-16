"""Unit tests for DCCP example problems."""

import cvxpy as cp
import pytest

from dccp import convexify_obj
from dccp.problem import DCCP, DCCPSettings
from dccp.utils import NonDCCPError
from tests.utils import assert_almost_equal


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

    def test_convexify_obj_damping_limit_line(self) -> None:
        """Test that convexify_obj raises NonDCCPError (objective.py:47)."""
        x = cp.Variable(name="x0")
        prob = cp.Problem(cp.Maximize(x**2))
        dccp_solver = DCCP(prob, settings=DCCPSettings(max_iter_damp=0))
        x.value = None

        with pytest.raises(
            NonDCCPError, match="Damping did not yield a convexified objective"
        ):
            dccp_solver._construct_subproblem()

    def test_convexify_obj_returns_none(self) -> None:
        """Test convexify_obj returns None if linearization fails."""
        x = cp.Variable()
        obj = cp.Maximize(cp.square(x))
        assert not obj.is_dcp()

        res = convexify_obj(obj)
        assert res is None
