"""Unit tests for DCCP example problems."""

import cvxpy as cp
import numpy as np

from dccp import is_dccp

from .utils import assert_almost_equal, assert_almost_in


class TestExamples:
    """Unit test examples."""

    def test_readme_example(self) -> None:
        """Test the example in the readme.

        self.sol - All known possible solutions to the problem in the readme.
        """
        y = cp.Variable(2, name="y")
        z = cp.Variable(2, name="z")
        prob = cp.Problem(
            cp.Maximize(cp.norm(y - z, 2)),
            [y >= 0, y <= 1, z >= 0, z <= 1],
        )
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

    def test_circle_packing(self) -> None:
        """Test the circle packing example."""
        n = 10
        r = np.linspace(1, 5, n)
        c = cp.Variable((n, 2))

        # create constraints s.t. circles don't overlap: ||c[i] - c[j]|| >= r[i] + r[j]
        constr = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                constr += [cp.norm(c[i, :] - c[j, :]) >= r[i] + r[j]]

        # create the problem
        prob_cp = cp.Problem(cp.Minimize(cp.max(cp.max(cp.abs(c), axis=1) + r)), constr)

        assert not prob_cp.is_dcp()
        assert is_dccp(prob_cp)
        prob_cp.solve(method="dccp", solver="ECOS", ep=1e-3, max_slack=1e-3, seed=0)
        assert prob_cp.status == cp.OPTIMAL

        le = cp.max(cp.max(cp.abs(c), axis=1) + r).value * 2  # type: ignore
        ratio = np.pi * cp.sum(cp.square(r)).value / cp.square(le).value  # type: ignore
        assert ratio > 0.68  # the ratio should be greater than 0.68 for a valid packing
