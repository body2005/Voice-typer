"""Unit tests for paths and logging setup."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from voicetyper.paths import (
    get_app_data_dir,
    get_config_path,
    get_log_dir,
    get_log_path,
    resource_path,
    setup_logging,
)


def test_resource_path_dev() -> None:
    """Test resource_path resolves correctly in development mode."""
    icon_path = resource_path("assets/icon.png")
    assert icon_path.exists()
    assert icon_path.name == "icon.png"


def test_resource_path_frozen() -> None:
    """Test resource_path resolves using _MEIPASS when frozen."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_asset = Path(tmp_dir) / "assets" / "fake.txt"
        fake_asset.parent.mkdir(parents=True)
        fake_asset.write_text("frozen_content", encoding="utf-8")

        with patch("sys._MEIPASS", tmp_dir, create=True):
            resolved = resource_path("assets/fake.txt")
            assert resolved == fake_asset
            assert resolved.read_text(encoding="utf-8") == "frozen_content"


def test_appdata_paths() -> None:
    """Test get_app_data_dir, get_config_path, get_log_dir, and get_log_path."""
    with tempfile.TemporaryDirectory() as tmp_appdata:
        with patch.dict("os.environ", {"APPDATA": tmp_appdata}):
            app_dir = get_app_data_dir()
            assert app_dir.exists()
            assert app_dir.name == "VoiceTyper"

            cfg_path = get_config_path()
            assert cfg_path.parent == app_dir
            assert cfg_path.name == "config.json"

            log_dir = get_log_dir()
            assert log_dir.exists()
            assert log_dir.name == "logs"

            log_file = get_log_path()
            assert log_file.parent == log_dir
            assert log_file.name == "voicetyper.log"


def test_setup_logging() -> None:
    """Test logging initialization with rotating file handler."""
    with tempfile.TemporaryDirectory() as tmp_appdata:
        with patch.dict("os.environ", {"APPDATA": tmp_appdata}):
            logger = setup_logging(force_reconfigure=True)
            assert logger is not None
            logger.info("Test message for logger")
            for h in logger.handlers:
                h.flush()
            log_file = get_log_path()
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert "Test message for logger" in content
            # Clean up handlers so files aren't kept open in temp dir on Windows
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)
