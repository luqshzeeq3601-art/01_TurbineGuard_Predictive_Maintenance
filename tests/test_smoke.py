"""Smoke tests for environment and package installation."""

import sys

import turbineguard


def test_package_import():
    """Verify package imports correctly with version."""
    assert turbineguard.__version__ == "0.1.0"


def test_python_version():
    """Verify runtime Python is 3.11+."""
    assert sys.version_info >= (3, 11)
