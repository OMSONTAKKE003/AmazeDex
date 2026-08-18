"""
Shared pytest configuration.

Adds the project root to sys.path so test modules can import
amazing_hand_gui and amazing_hand_cmd directly, regardless of
how pytest is invoked.

Registers the ``--hardware`` / ``--port`` / ``--baudrate`` options for
real-hardware system tests.
"""
import sys
from pathlib import Path

import pytest

# Project root is one level above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser):
    """Register CLI options used by test_system_hardware.py."""
    parser.addoption(
        "--hardware", action="store_true", default=False,
        help="Run tests that require real servo hardware",
    )
    parser.addoption(
        "--port", default="/dev/ttyACM0",
        help="Serial port for hardware tests (default: /dev/ttyACM0)",
    )
    parser.addoption(
        "--hw-baudrate", default=1000000, type=int,
        help="Baudrate for hardware tests (default: 1000000)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.hardware tests unless --hardware is passed."""
    if not config.getoption("--hardware"):
        skip_hw = pytest.mark.skip(reason="needs --hardware option to run")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hw)
