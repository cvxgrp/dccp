"""Test initialization of DCCP problems."""

import cvxpy as cp
import numpy as np

from dccp.initialization import initialize


class TestInitialization:
    """Unit tests for DCCP problem initialization."""

    def test_preserves_existing_values_when_random_false(self) -> None:
        """Variables with existing values retain those values when random=False."""
        x = cp.Variable(2, name="x")
        y = cp.Variable(2, name="y")
        z = cp.Variable(2, name="z")
        prob = cp.Problem(cp.Minimize(cp.norm(x - y + z, 2)), [x >= 0, y >= 0, z >= 0])

        # set initial values
        initial_x = np.array([1.0, 2.0])
        initial_y = np.array([3.0, 4.0])
        x.value = initial_x
        y.value = initial_y

        initialize(prob, random=False, seed=42)
        np.testing.assert_array_equal(x.value, initial_x)
        np.testing.assert_array_equal(y.value, initial_y)

    def test_reinitializes_all_when_random_true(self) -> None:
        """All variables should be reinitialized when random=True."""
        x = cp.Variable(2, name="x")
        y = cp.Variable(2, name="y")
        prob = cp.Problem(cp.Minimize(cp.norm(x - y, 2)), [x >= 0, y >= 0])

        # set initial values
        initial_x = np.array([1.0, 2.0])
        initial_y = np.array([3.0, 4.0])
        x.value = initial_x
        y.value = initial_y

        initialize(prob, random=True, seed=42, k_ini=4)

        # values should have changed
        assert not np.array_equal(x.value, initial_x)
        assert not np.array_equal(y.value, initial_y)
        assert x.value is not None
        assert y.value is not None

    def test_initializes_unset_variables(self) -> None:
        """Variables without initial values should be initialized."""
        x = cp.Variable(2, name="x")
        y = cp.Variable(2, name="y")
        prob = cp.Problem(cp.Minimize(cp.norm(x - y, 2)), [x >= 0, y >= 0])

        # leave variables uninitialized
        assert x.value is None
        assert y.value is None
        initialize(prob, random=False, seed=42)

        # variables should now have values
        assert x.value is not None
        assert y.value is not None
        assert x.value.shape == (2,)
        assert y.value.shape == (2,)

    def test_no_variables_to_initialize(self) -> None:
        """Function should handle case with no variables needing initialization."""
        x = cp.Variable(2, name="x")
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 0])

        # set initial value
        x.value = np.array([1.0, 2.0])
        initialize(prob, seed=42)
        np.testing.assert_array_equal(x.value, np.array([1.0, 2.0]))
