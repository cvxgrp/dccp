"""Unit tests for DCCP example problems."""

import cvxpy as cp

from dccp import convexify_constr

from .utils import assert_almost_equal


class TestExample:
    """Unit tests example."""

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
