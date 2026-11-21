"""Unit tests for config management, keyring handling, and startup toggle."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from voicetyper.config import Config, ConfigManager, MockStartupManager


def test_config_defaults() -> None:
    """Test default values of Config dataclass."""
    cfg = Config()
    assert cfg.hotkey == "ctrl+shift"
    assert cfg.mic_device is None
    assert cfg.start_stop_beeps is True
    assert cfg.language_hint == "Auto"
    assert cfg.max_duration_sec == 120.0
    assert cfg.min_duration_sec == 0.3
    assert cfg.start_with_windows is False
    assert cfg.paste_fallback_typing is False
    assert cfg.toggle_mode is False
    assert cfg.custom_vocabulary == ""
    assert cfg.transcription_engine == "local"
    assert cfg.whisper_model == "large-v3-turbo"
    assert cfg.api_key == ""

    # Ensure api_key is excluded from to_dict()
    d = cfg.to_dict()
    assert "api_key" not in d
    assert "hotkey" in d


def test_config_save_load_with_keyring() -> None:
    """Test saving and loading configuration with keyring functioning."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = Path(tmp_dir) / "config.json"
        startup = MockStartupManager(initial_enabled=False)
        mgr = ConfigManager(config_path=cfg_file, startup_manager=startup)

        # Mock keyring
        mock_vault: dict[tuple[str, str], str] = {}

        def mock_get_pw(srv: str, user: str) -> str | None:
            return mock_vault.get((srv, user))

        def mock_set_pw(srv: str, user: str, pw: str) -> None:
            mock_vault[(srv, user)] = pw

        with patch("keyring.get_password", side_effect=mock_get_pw), \
             patch("keyring.set_password", side_effect=mock_set_pw), \
             patch.object(mgr, "_test_keyring", return_value=True):

            cfg = Config(
                hotkey="alt+space",
                language_hint="Arabic",
                start_with_windows=True,
                api_key="AIzaSyMockKeyForTesting123",
            )
            mgr.save_config(cfg)

            # Check config file was created and does NOT contain plain text API key
            assert cfg_file.exists()
            with open(cfg_file, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
            assert "api_key_fallback" not in saved_json
            assert saved_json["hotkey"] == "alt+space"
            assert saved_json["language_hint"] == "Arabic"
            assert startup.is_enabled() is True

            # Load configuration back
            loaded = mgr.load_config()
            assert loaded.hotkey == "alt+space"
            assert loaded.language_hint == "Arabic"
            assert loaded.start_with_windows is True
            assert loaded.api_key == "AIzaSyMockKeyForTesting123"


def test_config_save_load_fallback_when_keyring_unavailable() -> None:
    """Test fallback to config.json when keyring backend fails."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = Path(tmp_dir) / "config.json"
        startup = MockStartupManager()
        mgr = ConfigManager(config_path=cfg_file, startup_manager=startup)

        # Force keyring unavailable
        with patch.object(mgr, "_test_keyring", return_value=False):
            cfg = Config(api_key="FallbackSecretKey123")
            mgr.save_config(cfg)

            # File should contain api_key_fallback
            with open(cfg_file, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
            assert saved_json.get("api_key_fallback") == "FallbackSecretKey123"

            # Loading back should retrieve it
            loaded = mgr.load_config()
            assert loaded.api_key == "FallbackSecretKey123"


def test_config_overlay_position_save_and_reset() -> None:
    """Test saving and resetting custom overlay position."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = Path(tmp_dir) / "config.json"
        startup = MockStartupManager()
        mgr = ConfigManager(config_path=cfg_file, startup_manager=startup)

        # Initially None
        cfg = mgr.load_config()
        assert cfg.overlay_pos is None

        # Save custom position
        mgr.save_overlay_position(350, 520)
        loaded = mgr.load_config()
        assert loaded.overlay_pos == [350, 520]

        # Reset position
        mgr.reset_overlay_position()
        reloaded = mgr.load_config()
        assert reloaded.overlay_pos is None


def test_config_overlay_theme_persistence() -> None:
    """Test default overlay theme is 'classic' and can be persisted across saves."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = Path(tmp_dir) / "config.json"
        mgr = ConfigManager(config_path=cfg_file, startup_manager=MockStartupManager())

        cfg = mgr.load_config()
        assert cfg.overlay_theme == "classic"

        # Change theme and save
        cfg.overlay_theme = "neon_glass"
        mgr.save_config(cfg)

        reloaded = mgr.load_config()
        assert reloaded.overlay_theme == "neon_glass"
