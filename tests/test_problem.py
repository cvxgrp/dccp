"""Unit tests for DCCP problem module."""

from typing import ClassVar

import cvxpy as cp
import numpy as np
import pytest

from dccp.problem import DCCP, DCCPIter, dccp
from dccp.utils import DCCPSettings, NonDCCPError


class ValueExprStub:
    """Configurable expression-like stub for linearization map tests."""

    def __init__(
        self,
        values: list[float | None] | None = None,
        *,
        save_raises: bool = False,
        save_result: float = 1.0,
    ) -> None:
        """Initialize value sequence and save behavior."""
        self._values = [None] if values is None else values
        self._index = 0
        self._save_raises = save_raises
        self._save_result = save_result

    @property
    def value(self) -> float | None:
        """Return next value from sequence, then keep returning the last one."""
        idx = min(self._index, len(self._values) - 1)
        self._index += 1
        return self._values[idx]

    def save_value(self, _value: object) -> float:
        """Return save result or raise if configured."""
        if self._save_raises:
            msg = "Save failed"
            raise ValueError(msg)
        return self._save_result

    def __str__(self) -> str:
        """Return display name."""
        return "ValueExprStub"


class LinearizationDataStub:
    """Minimal linearization data holder for update path tests."""

    def __init__(self, expr: object | None = None) -> None:
        """Initialize with expression and update flag."""
        self.expr = (
            ValueExprStub(values=[None], save_raises=True) if expr is None else expr
        )
        self.updated = False

    def update(self) -> None:
        """Record that update was called."""
        self.updated = True


class FutureStub:
    """Mimic concurrent.futures.Future for local test control."""

    def __init__(
        self,
        res: tuple[float, dict[int, object]] | None = None,
        exc: Exception | None = None,
    ) -> None:
        """Initialize future outcome."""
        self._res = res
        self._exc = exc

    def result(self) -> tuple[float, dict[int, object]] | None:
        """Return the result or raise configured exception."""
        if self._exc is not None:
            raise self._exc
        return self._res


class TauNoneSubproblemDCCP(DCCP):
    """DCCP stub that bypasses subproblem construction."""

    def _construct_subproblem(self) -> None:
        """Skip subproblem construction."""


class IterStub:
    """Minimal iteration state for solve-loop branch tests."""

    def __init__(self, tau: cp.Parameter) -> None:
        """Initialize fixed iteration values."""
        self.tau = tau
        self.k = 0
        self.cost = np.inf

    @property
    def cost_no_slack(self) -> float:
        """Return fixed non-converged objective value."""
        return np.inf

    @property
    def slack(self) -> float:
        """Return fixed positive slack value."""
        return 1.0

    def solve(self, **_kwargs: object) -> None:
        """Advance one iteration."""
        self.k += 1


class NonOptimalSolveDCCP(DCCP):
    """DCCP stub that forces non-optimal solve status."""

    def _solve(self) -> float:
        """Set infeasible status and return infinite objective."""
        self.prob_in._status = cp.INFEASIBLE
        return np.inf


class NoBestSequentialDCCP(DCCP):
    """DCCP stub that returns no valid sequential solution."""

    def _solve_multi_sequential(
        self, _num_inits: int
    ) -> tuple[float, dict[int, object] | None, str]:
        """Return no best candidate."""
        return np.inf, None, cp.INFEASIBLE


class FutureQueueExecutor:
    """Executor stub returning pre-seeded futures in submission order."""

    _seeded_futures: ClassVar[list[FutureStub]] = []

    @classmethod
    def seed(cls, futures: list[FutureStub]) -> None:
        """Set deterministic future sequence for the next executor instance."""
        cls._seeded_futures = futures.copy()

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Initialize submission counter."""
        self._calls = 0

    def __enter__(self) -> "FutureQueueExecutor":
        """Enter context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit context manager."""

    def submit(self, _fn: object, *_args: object, **_kwargs: object) -> FutureStub:
        """Return next pre-seeded future."""
        future = self._seeded_futures[self._calls]
        self._calls += 1
        return future


class OneErrorThenSuccessDCCP(DCCP):
    """DCCP stub that errors once then returns a fixed successful result."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize base class and call state."""
        super().__init__(*args, **kwargs)
        self._call_count = 0
        self.solution_var_id: int | None = None

    def _solve_one_init(self) -> tuple[float | None, dict[int, object] | None]:
        """Raise on first call and return success afterwards."""
        self._call_count += 1
        if self._call_count == 1:
            msg = "Fail"
            raise NonDCCPError(msg)
        if self.solution_var_id is None:
            return None, None
        return 10.0, {self.solution_var_id: 10.0}


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

    def test_store_and_damping_skip_none_variable(self) -> None:
        """Test store and damping both skip variables with None value."""
        x = cp.Variable(name="x")
        y = cp.Variable(name="y")
        prob = cp.Problem(cp.Maximize(x**2 + y**2), [y >= 0])

        solver = DCCP(prob, settings=DCCPSettings(verify_dccp=False))
        x.value = None
        y.value = np.array(3.0)
        solver._prev_var_values = {}

        solver._store_previous_values()

        assert x not in solver._prev_var_values
        assert y in solver._prev_var_values

        y.value = np.array(4.0)
        solver._prev_var_values = {y: np.array(2.0)}
        solver._apply_damping()

        assert y.value is not None
        assert np.isclose(float(y.value), 3.6)

    def test_update_linearizations_expr_value_becomes_available(self) -> None:
        """Test update_linearizations path where expr.value transitions to valid."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Maximize(x**2), [x >= 0])
        solver = DCCP(prob, settings=DCCPSettings(verify_dccp=False))
        data = LinearizationDataStub(
            expr=ValueExprStub(values=[None, 1.0], save_result=1.0)
        )
        solver.linearization_map = {1: data}

        solver._update_linearizations()

        assert data.updated

    def test_solve_tau_none_skips_tau_update_branch(self) -> None:
        """Test solve loop handles None tau value without attempting tau update."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Maximize(x**2), [x >= 0])
        solver = TauNoneSubproblemDCCP(
            prob, settings=DCCPSettings(verify_dccp=False, max_iter=0)
        )
        solver.iter = IterStub(solver.tau)  # type: ignore[assignment]
        solver.iter.tau.value = None

        result = solver._solve()

        assert result == np.inf
        assert prob.status == cp.INFEASIBLE


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

    def test_solve_one_init_returns_none_when_not_optimal(self) -> None:
        """Test _solve_one_init returns (None, None) for non-optimal status."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])
        solver = NonOptimalSolveDCCP(
            prob, settings=DCCPSettings(verify_dccp=False, seed=42)
        )

        cost, var_values = solver._solve_one_init()

        assert cost is None
        assert var_values is None

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

    def test_solve_multi_init_restores_original_values_if_no_best_solution(
        self,
    ) -> None:
        """Test solve_multi_init restores original variable values on failure."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Maximize(cp.norm(x)), [x >= 0, x <= 1])
        x.value = np.array([0.25, 0.75])

        solver = NoBestSequentialDCCP(
            prob, settings=DCCPSettings(verify_dccp=False, seed=42)
        )

        result = solver.solve_multi_init(2, parallel=False)

        assert np.isneginf(result)
        assert prob.status == cp.INFEASIBLE
        assert x.value is not None
        assert np.allclose(x.value, np.array([0.25, 0.75]))

    def test_update_linearizations_damping_failure_lines(self) -> None:
        """Test parameter update failure triggers damping."""
        x = cp.Variable(name="x_var")
        x.value = np.array(1.0)
        prob = cp.Problem(cp.Maximize(x**2))
        dccp_solver = DCCP(prob, settings=DCCPSettings(max_iter_damp=2))
        dccp_solver._prev_var_values = {x: np.array(2.0)}
        dccp_solver.linearization_map = {1: LinearizationDataStub()}

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

    def test_solve_multi_init_parallel_handles_error(self) -> None:
        """Test solve_multi_parallel handles errors in futures."""
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(x**2), [x >= 1])
        dccp_solver = DCCP(prob)

        # We construct futures with specific outcomes
        f1 = FutureStub(exc=NonDCCPError("Worker fail"))
        f2 = FutureStub(res=(5.0, {x.id: 5.0}))

        def fake_as_completed(_futures: list[FutureStub]) -> list[FutureStub]:
            return [f1, f2]

        FutureQueueExecutor.seed([f1, f2])

        cost, _vars, status = dccp_solver._solve_multi_parallel(
            2,
            None,
            None,
            executor_cls=FutureQueueExecutor,
            as_completed_fn=fake_as_completed,
        )

        assert cost == 5.0
        assert status == cp.OPTIMAL

    def test_solve_multi_init_sequential_handles_error(self) -> None:
        """Test solve_multi_sequential continues on NonDCCPError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(x**2), [x >= 1])
        dccp_solver = OneErrorThenSuccessDCCP(prob)
        dccp_solver.solution_var_id = x.id

        cost, _vars, status = dccp_solver._solve_multi_sequential(2)
        assert cost == 10.0
        assert status == cp.OPTIMAL
