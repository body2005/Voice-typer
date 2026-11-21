"""Configuration management, secure API key storage via keyring, and Windows startup toggle.

Platform-specific registry access is isolated behind an interface to allow
easy testing and cross-platform unit test mockability.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import keyring

from voicetyper.paths import get_config_path

KEYRING_SERVICE_NAME = "VoiceTyper"
KEYRING_USERNAME = "gemini_api_key"
REGISTRY_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
REGISTRY_VALUE_NAME = "VoiceTyper"

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """User-configurable settings for VoiceTyper."""

    hotkey: str = "ctrl+shift"
    mic_device: str | int | None = None
    start_stop_beeps: bool = True
    language_hint: str = "Auto"  # "Auto", "Arabic", "English"
    max_duration_sec: float = 120.0
    min_duration_sec: float = 0.3
    start_with_windows: bool = False
    paste_fallback_typing: bool = False
    toggle_mode: bool = False
    custom_vocabulary: str = ""
    overlay_pos: list[int] | None = None
    overlay_theme: str = "classic"  # "classic", "neon_glass", "matte_island", "fluent_mica"
    restore_clipboard: bool = False  # False = keep transcribed text in Ctrl+V clipboard
    capitalize_english: str = "word"  # "word" (First Letter Capital of Each Word), "sentence", "none"
    transcription_engine: str = "local"  # "local" (faster-whisper GPU/CPU) or "gemini"
    whisper_model: str = "large-v3-turbo"  # "small", "medium", "large-v3-turbo"
    whisper_device: str = "auto"  # "auto", "cuda", "cpu"
    whisper_compute_type: str = "auto"  # "auto", "float16", "int8_float16", "int8"
    api_key: str = field(default="", repr=False)  # Excluded from repr for privacy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary without exposing API key by default."""
        data = asdict(self)
        # Never store raw API key in standard dictionary export
        data.pop("api_key", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], api_key: str = "") -> Config:
        """Create Config from dictionary and explicitly provided API key."""
        valid_keys = {
            "hotkey",
            "mic_device",
            "start_stop_beeps",
            "language_hint",
            "max_duration_sec",
            "min_duration_sec",
            "start_with_windows",
            "paste_fallback_typing",
            "toggle_mode",
            "custom_vocabulary",
            "overlay_pos",
            "overlay_theme",
            "restore_clipboard",
            "capitalize_english",
            "transcription_engine",
            "whisper_model",
            "whisper_device",
            "whisper_compute_type",
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data, api_key=api_key)


class StartupManagerProtocol(Protocol):
    """Protocol for managing application auto-start on user logon."""

    def is_enabled(self) -> bool:
        """Return True if startup entry exists."""
        ...

    def set_enabled(self, enable: bool) -> bool:
        """Enable or disable startup entry. Returns True if successful."""
        ...


class WindowsStartupManager:
    """Windows registry-based implementation of StartupManagerProtocol."""

    def __init__(self, value_name: str = REGISTRY_VALUE_NAME) -> None:
        self.value_name = value_name

    def _get_target_command(self) -> str:
        """Get the executable command line for startup."""
        if getattr(sys, "frozen", False):
            # In frozen PyInstaller app, sys.executable points to the actual .exe
            return f'"{sys.executable}"'
        else:
            # In dev mode, run pythonw with main.py if possible, or python
            main_script = Path(__file__).resolve().parent / "main.py"
            py_exe = sys.executable
            # Prefer pythonw if available next to python.exe
            pyw_exe = Path(py_exe).parent / "pythonw.exe"
            exe_to_use = str(pyw_exe) if pyw_exe.exists() else py_exe
            return f'"{exe_to_use}" "{main_script}"'

    def is_enabled(self) -> bool:
        """Check HKCU Run key for our entry."""
        if sys.platform != "win32":
            return False

        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_RUN_KEY,
                0,
                winreg.KEY_READ,
            ) as key:
                val, _ = winreg.QueryValueEx(key, self.value_name)
                return bool(val)
        except (FileNotFoundError, OSError):
            return False

    def set_enabled(self, enable: bool) -> bool:
        """Add or remove entry from HKCU Run key."""
        if sys.platform != "win32":
            logger.warning("Startup registry modification skipped: non-Windows platform")
            return False

        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_RUN_KEY,
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                if enable:
                    cmd = self._get_target_command()
                    winreg.SetValueEx(key, self.value_name, 0, winreg.REG_SZ, cmd)
                    logger.info("Enabled Windows startup entry: %s", cmd)
                else:
                    try:
                        winreg.DeleteValue(key, self.value_name)
                        logger.info("Removed Windows startup entry")
                    except FileNotFoundError:
                        pass
            return True
        except OSError as err:
            logger.error("Failed to update Windows startup registry: %s", err)
            return False


class MockStartupManager:
    """Mock startup manager for unit tests and non-Windows environments."""

    def __init__(self, initial_enabled: bool = False) -> None:
        self.enabled = initial_enabled

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enable: bool) -> bool:
        self.enabled = enable
        return True


class ConfigManager:
    """Manages loading, saving, and secret handling for VoiceTyper."""

    def __init__(
        self,
        config_path: Path | None = None,
        startup_manager: StartupManagerProtocol | None = None,
    ) -> None:
        self.config_path = config_path or get_config_path()
        self.startup_manager = startup_manager or WindowsStartupManager()
        self._keyring_available: bool | None = None

    def _test_keyring(self) -> bool:
        """Test if the system keyring is functional."""
        if self._keyring_available is not None:
            return self._keyring_available
        try:
            # Check if keyring backend is functional
            backend = keyring.get_keyring()
            if "fail" in backend.__class__.__name__.lower():
                self._keyring_available = False
            else:
                self._keyring_available = True
        except Exception as err:
            logger.warning("Keyring backend test failed: %s", err)
            self._keyring_available = False
        return self._keyring_available

    def get_api_key(self) -> str:
        """Retrieve API key from keyring, falling back to config.json if needed."""
        # 1. Check environment variable first (convenient for dev/CI)
        env_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if env_key:
            return env_key

        # 2. Try keyring
        if self._test_keyring():
            try:
                secret = keyring.get_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
                if secret:
                    return secret.strip()
            except Exception as err:
                logger.warning("Could not read API key from keyring: %s", err)

        # 3. Fallback to reading from config file if stored there
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    fallback_key = raw.get("api_key_fallback", "").strip()
                    if fallback_key:
                        logger.warning("Loaded API key from config.json fallback (plain text)")
                        return fallback_key
            except Exception as err:
                logger.warning("Failed to read fallback API key from config.json: %s", err)

        return ""

    def set_api_key(self, api_key: str) -> bool:
        """Save API key to keyring (or fallback to config.json if keyring unavailable)."""
        clean_key = api_key.strip()
        keyring_success = False

        if self._test_keyring():
            try:
                if clean_key:
                    keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, clean_key)
                else:
                    try:
                        keyring.delete_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
                    except keyring.errors.PasswordDeleteError:
                        pass
                keyring_success = True
            except Exception as err:
                logger.warning("Could not store API key in keyring: %s", err)
                keyring_success = False

        # If keyring failed, persist in config.json as fallback
        if not keyring_success:
            logger.warning("Keyring unavailable; persisting API key to config.json fallback")
            self._write_fallback_api_key(clean_key)
        else:
            # Remove any previous fallback key from config.json
            self._write_fallback_api_key("")

        return True

    def _write_fallback_api_key(self, api_key: str) -> None:
        """Internal helper to write or remove api_key_fallback in config.json."""
        data: dict[str, Any] = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if api_key:
            data["api_key_fallback"] = api_key
        else:
            data.pop("api_key_fallback", None)

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.config_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.config_path)
        except Exception as err:
            logger.error("Failed to write config file %s: %s", self.config_path, err)

    def load_config(self) -> Config:
        """Load configuration from disk and keyring."""
        data: dict[str, Any] = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as err:
                logger.warning("Error reading config file %s: %s; using defaults", self.config_path, err)

        api_key = self.get_api_key()
        cfg = Config.from_dict(data, api_key=api_key)

        # Synchronize start_with_windows from actual registry state if on Windows
        try:
            cfg.start_with_windows = self.startup_manager.is_enabled()
        except Exception as err:
            logger.debug("Could not query startup status: %s", err)

        return cfg

    def save_config(self, cfg: Config) -> None:
        """Save configuration to disk, keyring, and startup registry."""
        # 1. Save API key securely
        if cfg.api_key is not None:
            self.set_api_key(cfg.api_key)

        # 2. Save JSON settings
        data = cfg.to_dict()
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # If fallback key exists in existing file, preserve it
            if self.config_path.exists():
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        if "api_key_fallback" in old_data and not self._test_keyring():
                            data["api_key_fallback"] = old_data["api_key_fallback"]
                except Exception:
                    pass

            tmp_path = self.config_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.config_path)
            logger.info("Saved configuration to %s", self.config_path)
        except Exception as err:
            logger.error("Failed to save config to %s: %s", self.config_path, err)

        # 3. Synchronize startup registry
        try:
            self.startup_manager.set_enabled(cfg.start_with_windows)
        except Exception as err:
            logger.error("Failed to update startup configuration: %s", err)

    def save_overlay_position(self, x: int, y: int) -> None:
        """Persist custom overlay window position."""
        cfg = self.load_config()
        cfg.overlay_pos = [int(x), int(y)]
        self.save_config(cfg)

    def reset_overlay_position(self) -> None:
        """Reset overlay position back to default (bottom-right above clock)."""
        cfg = self.load_config()
        cfg.overlay_pos = None
        self.save_config(cfg)
