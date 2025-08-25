"""Unit tests for DCCP problem module."""

import cvxpy as cp
import numpy as np
import pytest

from dccp.problem import DCCP, DCCPIter, dccp
from dccp.utils import DCCPSettings, NonDCCPError


class TestDCCPIter:
    """Test the DCCPIter class."""

    def test_dccp_iter_initialization(self) -> None:
        """Test DCCPIter initialization with default values."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])

        iter_obj = DCCPIter(prob=prob)

        assert iter_obj.k == 0
        assert iter_obj.cost == np.inf
        assert iter_obj.tau.value == 0.005
        assert not iter_obj.vars_slack

    def test_slack_property_no_slack_vars(self) -> None:
        """Test slack property when no slack variables exist."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        iter_obj = DCCPIter(prob=prob)

        assert iter_obj.slack == 0.0

    def test_slack_property_with_slack_vars(self) -> None:
        """Test slack property with slack variables."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        slack1 = cp.Variable(2, nonneg=True)
        slack2 = cp.Variable(1, nonneg=True)
        slack1.value = np.array([1.0, 2.0])
        slack2.value = np.array([0.5])

        iter_obj = DCCPIter(prob=prob, vars_slack=[slack1, slack2])

        assert iter_obj.slack == 2.0  # max of all slack values

    def test_slack_property_with_none_values(self) -> None:
        """Test slack property when some slack variables have None values."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        slack1 = cp.Variable(1, nonneg=True)
        slack2 = cp.Variable(1, nonneg=True)
        slack1.value = None
        slack2.value = np.array([1.5])
        iter_obj = DCCPIter(prob=prob, vars_slack=[slack1, slack2])

        assert iter_obj.slack == 1.5

    def test_slack_sum_property(self) -> None:
        """Test slack_sum property."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        slack1 = cp.Variable(2, nonneg=True)
        slack2 = cp.Variable(1, nonneg=True)
        slack1.value = np.array([1.0, 2.0])
        slack2.value = np.array([0.5])

        iter_obj = DCCPIter(prob=prob, vars_slack=[slack1, slack2])

        assert iter_obj.slack_sum == 3.5  # sum of all slack values

    def test_slack_sum_with_none_values(self) -> None:
        """Test slack_sum property when some slack variables have None values."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        slack1 = cp.Variable(1, nonneg=True)
        slack2 = cp.Variable(1, nonneg=True)
        slack1.value = None
        slack2.value = np.array([1.5])

        iter_obj = DCCPIter(prob=prob, vars_slack=[slack1, slack2])

        assert iter_obj.slack_sum == 1.5

    def test_cost_ns_property(self) -> None:
        """Test cost_ns property (objective value minus tau * sum(slack))."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        x.value = np.array([2.0])
        prob.solve(solver="SCS")
        slack1 = cp.Variable(1, nonneg=True)
        slack1.value = np.array([1.0])

        tau = cp.Parameter(value=0.01)
        iter_obj = DCCPIter(prob=prob, tau=tau, vars_slack=[slack1])
        obj_val = prob.objective.value
        assert obj_val is not None
        expected_cost_ns = float(obj_val) - 0.01 * 1.0  # type: ignore
        assert abs(iter_obj.cost_ns - expected_cost_ns) < 1e-6

    def test_cost_ns_with_none_objective(self) -> None:
        """Test cost_ns property when objective value is None."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        iter_obj = DCCPIter(prob=prob)

        assert iter_obj.cost_ns == np.inf

    def test_solve_method(self) -> None:
        """Test the solve method of DCCPIter."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        iter_obj = DCCPIter(prob=prob)
        result = iter_obj.solve(solver="SCS")

        assert iter_obj.k == 1
        assert result is not None
        assert iter_obj.cost == result

    def test_solve_method_infeasible(self) -> None:
        """Test the solve method when problem is infeasible."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])  # infeasible
        iter_obj = DCCPIter(prob=prob)
        result = iter_obj.solve()

        assert iter_obj.k == 1
        if result is not None:
            assert iter_obj.cost == result


class TestDCCP:
    """Test the DCCP class."""

    def test_dccp_init_with_dcp_problem(self) -> None:
        """Test DCCP initialization with a DCP problem should raise error."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])  # This is DCP

        with pytest.raises(NonDCCPError, match="The problem is DCP compliant"):
            DCCP(prob)

    def test_dccp_init_with_non_dccp_problem(self) -> None:
        """Test DCCP initialization with non-DCCP problem raises error."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(cp.exp(x) + cp.log(x)), [x >= 0.1])

        settings = DCCPSettings(verify_dccp=True)
        with pytest.raises(NonDCCPError, match="Problem is not DCCP"):
            DCCP(prob, settings=settings)

    def test_dccp_init_with_settings(self) -> None:
        """Test DCCP initialization with custom settings."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(cp.sqrt(x)), [x >= 0, x <= 1])

        settings = DCCPSettings(
            max_iter=50,
            tau_ini=0.01,
            mu=1.5,
            tau_max=1000,
            ep=1e-4,
            max_slack=1e-2,
            seed=42,
            verify_dccp=False,
        )

        dccp_solver = DCCP(prob, settings=settings)

        assert dccp_solver.conf.max_iter == 50
        assert dccp_solver.conf.tau_ini == 0.01
        assert dccp_solver.conf.mu == 1.5
        assert dccp_solver.conf.tau_max == 1000
        assert dccp_solver.conf.ep == 1e-4
        assert dccp_solver.conf.max_slack == 1e-2
        assert dccp_solver.conf.seed == 42
        assert dccp_solver.conf.verify_dccp is False


class TestDccpFunction:
    """Test the dccp function."""

    def test_dccp_function_basic(self) -> None:
        """Test basic dccp function call."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result = dccp(prob, verify_dccp=False)

        assert result is not None
        assert prob.status == cp.OPTIMAL

    def test_dccp_function_with_params(self) -> None:
        """Test dccp function with custom parameters."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result = dccp(
            prob,
            max_iter=50,
            tau_ini=0.01,
            mu=1.5,
            tau_max=1000,
            ep=1e-4,
            max_slack=1e-2,
            seed=42,
            verify_dccp=False,
            solver="SCS",
        )

        assert result is not None
        assert prob.status == cp.OPTIMAL

    def test_dccp_function_multi_initialization(self) -> None:
        """Test dccp function with multiple initializations."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result = dccp(prob, k_ccp=3, verify_dccp=False, seed=42)

        assert result is not None
        assert prob.status == cp.OPTIMAL

    def test_dccp_function_single_initialization(self) -> None:
        """Test dccp function with single initialization (k_ccp=1)."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result = dccp(prob, k_ccp=1, verify_dccp=False)

        assert result is not None
        assert prob.status == cp.OPTIMAL
