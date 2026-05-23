"""Tests for cvxpy API compatibility."""

from __future__ import annotations

import cvxpy as cp
from cvxpy.constraints.zero import Equality
from cvxpy.reductions.solution import Solution

from dccp.problem import _set_problem_status


class TestCvxpyCompatibility:
    """Test compatibility with cvxpy API."""

    def test_status_setting(self) -> None:
        """Test that we can set problem status using our helper."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x))

        _set_problem_status(prob, cp.OPTIMAL)
        assert prob.status == cp.OPTIMAL

        _set_problem_status(prob, cp.INFEASIBLE)
        assert prob.status == cp.INFEASIBLE

    def test_status_setting_unsolved_problem(self) -> None:
        """Test that we can set status on an unsolved problem."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x**2))

        # Problem hasn't been solved, so it has no _solution yet
        assert not hasattr(prob, "_solution") or prob._solution is None

        _set_problem_status(prob, cp.OPTIMAL)
        assert prob.status == cp.OPTIMAL

    def test_var_dict_exists(self) -> None:
        """Test that var_dict is available."""
        x = cp.Variable(name="x")
        y = cp.Variable(name="y")
        prob = cp.Problem(cp.Minimize(x + y))

        assert hasattr(prob, "var_dict")
        assert "x" in prob.var_dict
        assert "y" in prob.var_dict
        assert prob.var_dict["x"] is x
        assert prob.var_dict["y"] is y

    def test_grad_property_exists(self) -> None:
        """Test that Expression.grad property exists."""
        x = cp.Variable()
        expr = x**2
        x.value = 1.0

        assert hasattr(expr, "grad")
        grad_map = expr.grad
        assert isinstance(grad_map, dict)
        assert x in grad_map

    def test_equality_constraint_import(self) -> None:
        """Test that Equality constraint can be imported and used."""
        x = cp.Variable()
        constraint = x == 0

        assert isinstance(constraint, Equality)

    def test_curvature_strings(self) -> None:
        """Test that curvature is still represented as strings."""
        x = cp.Variable()

        # Affine expression
        affine = x + 1
        assert isinstance(affine.curvature, str)
        assert affine.curvature in ["AFFINE", "CONSTANT", "UNKNOWN"]

        # Convex expression
        convex = x**2
        assert isinstance(convex.curvature, str)
        assert convex.curvature == "CONVEX"

        # Concave expression
        concave = -(x**2)
        assert isinstance(concave.curvature, str)
        assert concave.curvature == "CONCAVE"

    def test_solution_object_structure(self) -> None:
        """Test that Solution object can be created and has expected structure."""
        sol = Solution(cp.OPTIMAL, 1.0, {}, {}, {})
        assert sol.status == cp.OPTIMAL
        assert sol.opt_val == 1.0
