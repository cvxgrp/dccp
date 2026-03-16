"""Testing utilities for DCCP."""

from typing import Self

import cvxpy as cp
import numpy as np
from numpy.testing import assert_allclose


class FakeFuture:
    """Mimic concurrent.futures.Future."""

    def __init__(
        self,
        res: tuple[float, dict] | None = None,
        exc: Exception | None = None,
    ) -> None:
        """Initialize the fake future."""
        self._res = res
        self._exc = exc

    def result(self) -> tuple[float, dict] | None:
        """Return the result or raise an exception."""
        if self._exc:
            raise self._exc
        return self._res


class FakeExecutor:
    """Mimic ProcessPoolExecutor."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Initialize the fake executor."""

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit the context manager."""

    def submit(self, _fn: object, *_args: object, **_kwargs: object) -> FakeFuture:
        """Submit a task to the fake executor."""
        return FakeFuture()


def assert_almost_equal(
    a: float | np.ndarray, b: float | np.ndarray, rtol: float = 1e-6, atol: float = 1e-6
) -> None:
    """Assert that two arrays are almost equal."""
    assert_allclose(
        np.asarray(a), b, rtol=rtol, atol=atol, err_msg="Arrays are not almost equal."
    )


def assert_almost_in(
    a: float | np.ndarray,
    b: list[float | np.ndarray],
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """Assert that a is almost equal to one of the items in b."""
    for item in b:
        try:
            assert_almost_equal(a, item, rtol=rtol, atol=atol)
        except AssertionError:
            continue
        else:
            return
    msg = f"{a} is not almost equal to any item in {b}."
    raise AssertionError(msg)

class FakeExpr:
    """A fake expression that always fails."""

    def __init__(self) -> None:
        """Initialize the fake expression."""
        self.value = None

    def save_value(self, _val: float) -> float:
        """Raise exception as requested."""
        msg = "Save failed"
        raise ValueError(msg)

    def __str__(self) -> str:
        """Return a string representation."""
        return "FakeExpr"


class FakeLinearizationData:
    """Minimal LinearizationData for testing."""

    def __init__(self) -> None:
        """Initialize the fake linearization data."""
        self.expr = FakeExpr()

    def update(self) -> None:
        """Do nothing."""


class FakeExpression(cp.Expression):
    """Mimic cp.Expression."""

    def __init__(
        self, shape: tuple[int, ...] = (), value: float | np.ndarray | None = None
    ) -> None:
        """Initialize the fake expression."""
        self._shape = shape
        self._value = value
        self.grad_map: dict = {}
        self._variables: list = []
        self._parameters_list: list = []
        self._name = "fake_expr"
        self._is_affine = False
        self._is_convex = False
        self._is_concave = False
        self._is_constant = False
        self._is_complex = False
        super().__init__()

    def name(self) -> str:
        """Return name."""
        return self._name

    def is_constant(self) -> bool:
        """Return is_constant."""
        return self._is_constant

    def is_affine(self) -> bool:
        """Return is_affine."""
        return self._is_affine

    def is_convex(self) -> bool:
        """Return is_convex."""
        return self._is_convex

    def is_concave(self) -> bool:
        """Return is_concave."""
        return self._is_concave

    def is_dpp(self, _context: str = "dcp") -> bool:
        """Return is_dpp."""
        return True

    def is_log_log_convex(self) -> bool:
        """Return is_log_log_convex."""
        return False

    def is_log_log_concave(self) -> bool:
        """Return is_log_log_concave."""
        return False

    # Properties
    @property
    def value(self) -> float | np.ndarray | None:
        """Return value."""
        return self._value

    @value.setter
    def value(self, val: float | np.ndarray | None) -> None:
        """Set value."""
        self._value = val

    @property
    def grad(self) -> dict:
        """Return grad."""
        return self.grad_map

    @grad.setter
    def grad(self, val: dict) -> None:
        """Set grad."""
        self.grad_map = val

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape."""
        return self._shape

    @shape.setter
    def shape(self, val: tuple[int, ...]) -> None:
        """Set shape."""
        self._shape = val

    # Abstract methods from Expression
    def variables(self) -> list:
        """Return variables."""
        return self._variables

    def parameters(self) -> list:
        """Return parameters."""
        return self._parameters_list

    def constants(self) -> list:
        """Return constants."""
        return []

    def is_nonneg(self) -> bool:
        """Return is_nonneg."""
        return False

    def is_nonpos(self) -> bool:
        """Return is_nonpos."""
        return False

    def is_imag(self) -> bool:
        """Return is_imag."""
        return False

    def is_complex(self) -> bool:
        """Return is_complex."""
        return self._is_complex

    def is_linearizable_convex(self) -> bool:
        """Return is_linearizable_convex."""
        return False

    def is_linearizable_concave(self) -> bool:
        """Return is_linearizable_concave."""
        return False

    def get_bounds(self) -> tuple[float | None, float | None]:
        """Return bounds."""
        return None, None

    def domain(self) -> list[cp.Constraint]:
        """Return domain."""
        return []
