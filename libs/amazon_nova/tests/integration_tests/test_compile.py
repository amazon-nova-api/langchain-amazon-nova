"""Placeholder test for CI compile check."""

import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Placeholder test for compile marker.

    This test is used by the CI workflow to verify that integration tests
    can be imported and compiled without actually running them. The workflow
    runs: pytest -m compile tests/integration_tests
    """
    pass
