"""Unit tests for DCCP example problems."""

import cvxpy as cp
import pytest

from dccp import convexify_obj
from dccp.problem import DCCP, DCCPSettings
from dccp.utils import NonDCCPError
from tests.utils import FakeExpression, assert_almost_equal

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

    def test_convexify_obj_damping_limit_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that convexify_obj raises NonDCCPError (objective.py:47)."""
        x = cp.Variable(name="x0")
        prob = cp.Problem(cp.Maximize(x**2))
        dccp_solver = DCCP(prob, settings=DCCPSettings(max_iter_damp=2))

        # Accept any arguments
        monkeypatch.setattr(
            "dccp.problem.convexify_obj", lambda *_args, **_kwargs: None
        )

        with pytest.raises(
            NonDCCPError, match="Damping did not yield a convexified objective"
        ):
            dccp_solver._construct_subproblem()

    def test_convexify_obj_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test convexify_obj returns None if linearization fails."""
        x = cp.Variable()
        # Mock expression to force linearize to fail
        mock_expr = FakeExpression(shape=(), value=1.0)
        mock_expr._variables = [x]
        mock_expr.grad = {x: None}
        mock_expr._is_complex = False
        mock_expr._is_constant = False
        mock_expr._is_affine = False
        mock_expr._is_convex = False
        mock_expr._is_concave = False
        mock_expr._parameters = []
        mock_expr._name = "mock_expr"

        # Create non-dcp objective
        obj = cp.Minimize(mock_expr)

        # Force is_dcp to False so it doesn't return early
        monkeypatch.setattr(cp.Minimize, "is_dcp", lambda _self: False)

        res = convexify_obj(obj)
        assert res is None
