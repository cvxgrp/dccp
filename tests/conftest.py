"""Pytest configuration for DCCP tests."""

import logging
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def setup_debug_logging() -> Generator[None, None, None]:
    """Set up debug logging for tests without changing default module behavior."""
    dccp_logger = logging.getLogger("dccp")
    original_level = dccp_logger.level
    dccp_logger.setLevel(logging.DEBUG)
    yield
    dccp_logger.setLevel(original_level)
