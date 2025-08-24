"""Unit tests for DCCP example problems."""

import cvxpy as cp
import numpy as np
import pytest
from cvxpy.constraints.constraint import Constraint

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
        prob_cp.solve(method="dccp", solver="SCS", ep=1e-3, max_slack=1e-3, seed=0)
        assert prob_cp.status == cp.OPTIMAL

        le = cp.max(cp.max(cp.abs(c), axis=1) + r).value * 2  # type: ignore
        ratio = np.pi * cp.sum(cp.square(r)).value / cp.square(le).value  # type: ignore
        assert ratio > 0.68  # the ratio should be greater than 0.68 for a valid packing

    def test_log(self) -> None:
        """Test example with logarithm."""
        x = cp.Variable(1)
        constr: list[Constraint] = [cp.log(x) <= 1, x >= 1]
        prob = cp.Problem(cp.Minimize(cp.log(x)), constr)
        result = prob.solve(
            method="dccp", solver="Clarabel", ep=1e-3, max_slack=1e-3, seed=0
        )
        assert prob.status == cp.OPTIMAL
        assert result is not None
        assert_almost_equal(result, 0)  # type: ignore
        assert x.value is not None
        assert_almost_equal(float(x.value[0]), 1)  # type: ignore

    def test_damping(self) -> None:
        """Test DCCP algorithm works with settings that will trigger damping."""
        x = cp.Variable(1, name="x")
        obj = cp.Minimize(cp.sqrt(x))
        constr: list[cp.Constraint] = [x >= -1]
        prob = cp.Problem(obj, constr)
        result = prob.solve(
            method="dccp",
            seed=0,
        )
        assert prob.status == cp.OPTIMAL
        assert result is not None
        assert_almost_equal(float(result), 0)  # type: ignore
        assert x.value is not None
        assert_almost_equal(float(x.value[0]), 0)  # type: ignore

    @pytest.mark.skip(reason="Bilinear problems are not supported in DCCP yet.")
    def test_bilinear(self) -> None:
        """Test bilinear problem."""
        x = cp.Variable(1, name="x", nonneg=True)
        y = cp.Variable(1, name="y", nonneg=True)
        obj = cp.Minimize(cp.square(x + y) + cp.square(x - y))
        constr: list[cp.Constraint] = [cp.square(x - 2) + cp.square(y - 2) == 2]
        prob = cp.Problem(obj, constr)
        result = prob.solve(
            method="dccp",
            verify_dccp=False,
            seed=0,
            ep=1e-2,
        )
        assert prob.status == cp.OPTIMAL
        assert result is not None
        assert_almost_equal(float(result), 4.0)  # type: ignore
        assert x.value is not None
        assert_almost_equal(float(x.value[0]), 1.0)  # type
        assert y.value is not None
        assert_almost_equal(float(y.value[0]), 1.0)  # type

    def test_boolean_least_squares(self) -> None:
        """Test boolean least squares problem."""
        n = 100
        noise_sigma = np.sqrt(n / 4)
        rng = np.random.default_rng(seed=0)
        A = rng.standard_normal((n, n))  # noqa: N806
        x0 = rng.integers(0, 2, size=(n, 1))
        x0 = x0 * 2 - 1
        v = rng.standard_normal((n, 1)) * noise_sigma
        y = np.dot(A, x0) + v
        x = cp.Variable((n, 1))
        constr: list[Constraint] = [cp.square(x) == 1]

        # solve by dccp
        prob = cp.Problem(cp.Minimize(cp.norm(A @ x - y, 2)), constr)
        result = prob.solve(method="dccp", ep=1e-3, max_slack=1e-3, seed=0)
        assert prob.status == cp.OPTIMAL
        assert result is not None
        solution = list(x.value)  # type: ignore
        recover = np.array(solution)
        err = np.linalg.norm(recover - x0, 2)
        assert err < 8.0
        # check all x[i] are close to either +1 or -1
        for i in range(n):
            assert abs(abs(recover[i, 0]) - 1) < 1e-2

    def test_multi_initialization(self) -> None:
        """Test solving with multiple random initializations."""
        y = cp.Variable(2, name="y")
        z = cp.Variable(2, name="z")
        prob = cp.Problem(
            cp.Maximize(cp.norm(y - z, 2)),
            [y >= 0, y <= 1, z >= 0, z <= 1],
        )

        # solve with multiple initializations
        result_multi = prob.solve(method="dccp", k_ccp=5)

        assert prob.status == cp.OPTIMAL
        assert result_multi is not None
        assert_almost_equal(float(result_multi), np.sqrt(2))  # type: ignore
