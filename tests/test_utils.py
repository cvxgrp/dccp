"""Unit tests for DCCP utils module."""

import cvxpy as cp
import pytest

from dccp.utils import DCCPSettings, NonDCCPError, is_dccp


class TestUtils:
    """Unit tests for utils module."""

    def test_non_dccp_error_default_message(self) -> None:
        """Test NonDCCPError with default message."""
        with pytest.raises(NonDCCPError) as exc_info:
            raise NonDCCPError

        assert str(exc_info.value) == "Problem is not DCCP compliant."

    def test_non_dccp_error_custom_message(self) -> None:
        """Test NonDCCPError with custom message."""
        custom_message = "Custom error message"
        with pytest.raises(NonDCCPError) as exc_info:
            raise NonDCCPError(custom_message)

        assert str(exc_info.value) == custom_message

    def test_is_dccp_non_dccp_objective(self) -> None:
        """Test is_dccp with non-DCCP objective."""
        x = cp.Variable(1)
        objective = cp.Minimize(cp.multiply(cp.log(x), cp.sqrt(x)))
        problem = cp.Problem(objective, [x >= 1])
        result = is_dccp(problem)
        assert result is False

    def test_dccp_settings_custom_values(self) -> None:
        """Test DCCPSettings with custom values."""
        custom_values = {
            "max_iter": 200,
            "max_iter_damp": 20,
            "tau_ini": 0.01,
            "mu": 1.5,
            "tau_max": 2e8,
            "k_ini": 2,
            "k_ccp": 2,
            "max_slack": 1e-4,
            "ep": 1e-6,
            "std": 20.0,
            "seed": 42,
            "verify_dccp": False,
        }

        settings = DCCPSettings(**custom_values)
        for key, expected_value in custom_values.items():
            assert getattr(settings, key) == expected_value
