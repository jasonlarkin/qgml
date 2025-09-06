"""Configuration for pytest."""
import pytest

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "quantum: mark test as quantum computation related"
    )
    config.addinivalue_line(
        "markers",
        "ground_state: mark test as related to ground state computation"
    )
    config.addinivalue_line(
        "markers",
        "edge_case: mark test as dealing with numerical edge cases"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

@pytest.fixture
def test_output_dir(tmp_path):
    """Fixture to provide a temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir 