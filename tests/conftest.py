"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root to sys.path so tests can import voicetyper directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication  # noqa: E402

# Ensure QApplication is initialized for the test suite
q_app: QApplication | None = QApplication.instance() or QApplication(sys.argv)

