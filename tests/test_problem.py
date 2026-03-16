"""Unit tests for DCCP problem module."""

import cvxpy as cp
import numpy as np
import pytest

import dccp.problem as dccp_impl  # Aliased to avoid conflict with dccp function
from dccp.problem import DCCP, DCCPIter, dccp
from dccp.utils import DCCPSettings, NonDCCPError
from tests.utils import FakeExecutor, FakeFuture, FakeLinearizationData


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

    def test_slack_property_no_slack_vars_coverage(self) -> None:
        """Test slack property when no slack variables exist (problem.py:37)."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        iter_obj = DCCPIter(prob=prob)

        # Ensure vars_slack is empty
        assert not iter_obj.vars_slack
        # This returns 0.0 at line 37
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

    def test_cost_no_slack_property(self) -> None:
        """Test cost_no_slack property (objective value minus tau * sum(slack))."""
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
        expected_cost_no_slack = float(obj_val) - 0.01 * 1.0  # type: ignore
        assert abs(iter_obj.cost_no_slack - expected_cost_no_slack) < 1e-6

    def test_cost_no_slack_with_none_objective(self) -> None:
        """Test cost_no_slack property when objective value is None."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x**2), [x >= 0])
        iter_obj = DCCPIter(prob=prob)

        assert iter_obj.cost_no_slack == np.inf

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

    def test_dccp_function_multi_init_sequential(self) -> None:
        """Test dccp function with multiple initializations in sequential mode."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result = dccp(prob, k_ccp=3, parallel=False, verify_dccp=False, seed=42)

        assert result is not None
        assert prob.status == cp.OPTIMAL

    def test_dccp_function_multi_init_parallel(self) -> None:
        """Test dccp function with multiple initializations in parallel mode."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result = dccp(
            prob, k_ccp=3, parallel=True, max_workers=2, verify_dccp=False, seed=42
        )

        assert result is not None
        assert prob.status == cp.OPTIMAL

    def test_dccp_sequential_vs_parallel_consistency(self) -> None:
        """Test that sequential and parallel modes produce similar results."""
        x = cp.Variable(2)
        prob_seq = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])
        prob_par = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result_seq = dccp(
            prob_seq, k_ccp=2, parallel=False, verify_dccp=False, seed=123
        )
        result_par = dccp(prob_par, k_ccp=2, parallel=True, verify_dccp=False, seed=123)

        # Both should find optimal solutions (values may differ due to randomness)
        assert result_seq is not None
        assert result_par is not None
        assert prob_seq.status == cp.OPTIMAL
        assert prob_par.status == cp.OPTIMAL


class TestMaximization:
    """Test that maximization problems are handled correctly."""

    def test_maximization_sequential_returns_correct_result(self) -> None:
        """Test sequential maximization returns correct positive value."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result = dccp(prob, k_ccp=3, parallel=False, seed=42)

        # For Maximize(norm(x)) with x in [0,1]^2, optimal is x=[1,1], norm=sqrt(2)
        expected = np.sqrt(2)
        assert result > 0, "Maximization result should be positive"
        assert np.isclose(result, expected, atol=0.1), f"Expected ~1.414, got {result}"

    def test_maximization_parallel_returns_correct_result(self) -> None:
        """Test parallel maximization returns correct positive value."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        result = dccp(prob, k_ccp=3, parallel=True, seed=42)

        # For Maximize(norm(x)) with x in [0,1]^2, optimal is x=[1,1], norm=sqrt(2)
        expected = np.sqrt(2)
        assert result > 0, "Maximization result should be positive"
        assert np.isclose(result, expected, atol=0.1), f"Expected ~1.414, got {result}"


class TestSolveMultiInit:
    """Test the solve_multi_init method and helpers."""

    def test_solve_one_init(self) -> None:
        """Test _solve_one_init helper method."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        solver = DCCP(prob, settings=DCCPSettings(verify_dccp=False, seed=42))
        cost, var_values = solver._solve_one_init()

        assert cost is not None
        assert var_values is not None
        assert len(var_values) == 1  # One variable
        assert x.id in var_values  # Keyed by variable id
        assert solver.prob_in.status == cp.OPTIMAL

    def test_solve_multi_sequential(self) -> None:
        """Test _solve_multi_sequential helper method."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        solver = DCCP(prob, settings=DCCPSettings(verify_dccp=False, seed=42))
        best_cost, best_var_values, best_status = solver._solve_multi_sequential(3)

        assert best_cost != np.inf
        assert best_var_values is not None
        assert best_status == cp.OPTIMAL

    def test_solve_multi_parallel(self) -> None:
        """Test _solve_multi_parallel helper method."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        solver = DCCP(prob, settings=DCCPSettings(verify_dccp=False, seed=42))
        best_cost, best_var_values, best_status = solver._solve_multi_parallel(
            3, max_workers=2, mp_context=None
        )

        assert best_cost != np.inf
        assert best_var_values is not None
        assert best_status == cp.OPTIMAL

    def test_solve_multi_init_with_num_inits_one(self) -> None:
        """Test solve_multi_init returns early when num_inits <= 1."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])

        solver = DCCP(prob, settings=DCCPSettings(verify_dccp=False, seed=42))
        result = solver.solve_multi_init(1)

        assert result is not None
        assert prob.status == cp.OPTIMAL

    def test_update_linearizations_damping_failure_lines(self) -> None:
        """Test parameter update failure triggers damping."""
        x = cp.Variable(name="x_var")
        x.value = np.array(1.0)
        prob = cp.Problem(cp.Maximize(x**2))
        dccp_solver = DCCP(prob, settings=DCCPSettings(max_iter_damp=2))
        dccp_solver._prev_var_values = {x: np.array(2.0)}
        dccp_solver.linearization_map = {1: FakeLinearizationData()}

        with pytest.raises(
            NonDCCPError, match="Damping did not yield valid parameters"
        ):
            dccp_solver._update_linearizations()

    def test_solve_loop_termination_infeasible_line(self) -> None:
        """Test that solve loop termination sets status coverage."""
        x = cp.Variable()
        # Minimal problem: Maximize convex function is non-DCP
        prob = cp.Problem(cp.Maximize(x**2), [x >= 0])
        # Force loop to run exactly once and then terminate due to max_iter
        dccp_solver = DCCP(prob, settings=DCCPSettings(max_iter=0))

        dccp_solver._solve()

        # Should not have converged in 1 iteration, so set to INFEASIBLE
        assert prob._status == cp.INFEASIBLE

    def test_solve_multi_init_parallel_handles_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test solve_multi_parallel handles errors in futures."""
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(x**2), [x >= 1])
        dccp_solver = DCCP(prob)

        # We construct futures with specific outcomes
        f1 = FakeFuture(exc=NonDCCPError("Worker fail"))
        f2 = FakeFuture(res=(5.0, {x.id: 5.0}))

        def fake_as_completed(_futures: list) -> list[FakeFuture]:
            return [f1, f2]

        # Monkeypatch dccp.problem references
        monkeypatch.setattr(dccp_impl, "ProcessPoolExecutor", FakeExecutor)
        monkeypatch.setattr(dccp_impl, "as_completed", fake_as_completed)

        cost, _vars, status = dccp_solver._solve_multi_parallel(2, None, None)

        assert cost == 5.0
        assert status == cp.OPTIMAL

    def test_solve_multi_init_sequential_handles_error(self) -> None:
        """Test solve_multi_sequential continues on NonDCCPError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(x**2), [x >= 1])
        dccp_solver = DCCP(prob)

        # Mock _solve_one_init to handle sequential calls manually
        call_count = 0

        def side_effect() -> tuple[float, dict] | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                err_msg = "Fail"
                raise NonDCCPError(err_msg)
            return (10.0, {x.id: 10.0})

        dccp_solver._solve_one_init = side_effect

        cost, _vars, status = dccp_solver._solve_multi_sequential(2)
        assert cost == 10.0
        assert status == cp.OPTIMAL
