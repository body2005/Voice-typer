"""Path resolution and application logging management.

Provides reliable access to bundled assets in both development and frozen
(PyInstaller) modes, standard Windows APPDATA paths, and rotating file logging.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

APP_NAME = "VoiceTyper"


def resource_path(relative_path: str | Path) -> Path:
    """Get absolute path to a resource, works for dev and PyInstaller bundle.

    PyInstaller creates a temp folder and stores path in _MEIPASS.
    """
    if hasattr(sys, "_MEIPASS"):
        # Running from PyInstaller bundle
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # Running from source: check package directory, then parent repo root
        pkg_root = Path(__file__).resolve().parent
        if (pkg_root / relative_path).exists():
            base_path = pkg_root
        else:
            base_path = pkg_root.parent

    return (base_path / relative_path).resolve()


def get_app_data_dir() -> Path:
    """Return %APPDATA%/VoiceTyper, creating it if it doesn't exist."""
    appdata_env = os.environ.get("APPDATA")
    if appdata_env:
        base = Path(appdata_env)
    else:
        base = Path.home() / "AppData" / "Roaming"

    app_dir = base / APP_NAME
    try:
        app_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logging.warning("Failed to create app data directory %s: %s", app_dir, err)
    return app_dir


def get_config_path() -> Path:
    """Return path to config.json."""
    return get_app_data_dir() / "config.json"


def get_log_dir() -> Path:
    """Return path to logs directory, creating it if it doesn't exist."""
    log_dir = get_app_data_dir() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logging.warning("Failed to create log directory %s: %s", log_dir, err)
    return log_dir


def get_log_path() -> Path:
    """Return path to voicetyper.log."""
    return get_log_dir() / "voicetyper.log"


def setup_logging(level: int = logging.INFO, force_reconfigure: bool = False) -> logging.Logger:
    """Configure rotating file and console logging for VoiceTyper.

    Privacy rule: Never log raw API keys, passwords, or transcribed user speech.
    """
    logger = logging.getLogger("voicetyper")
    if logger.handlers and not force_reconfigure:
        return logger

    logger.setLevel(level)

    # Remove existing handlers if reconfiguring
    if force_reconfigure:
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler (5 MB max, 3 backups)
    log_path = get_log_path()
    try:
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as err:
        logging.warning("Failed to set up rotating log handler at %s: %s", log_path, err)

    logger.info("Logging initialized. Log file: %s", log_path)
    return logger


def install_global_exception_handler() -> None:
    """Install sys.excepthook and threading.excepthook to log unhandled errors."""

    def handle_sys_exception(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Unhandled top-level exception:", exc_info=(exc_type, exc_value, exc_traceback))

    def handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        exc_val = args.exc_value if args.exc_value is not None else args.exc_type()
        logging.critical(
            "Unhandled thread exception in [%s]:",
            args.thread.name if args.thread else "UnknownThread",
            exc_info=(args.exc_type, exc_val, args.exc_traceback),
        )

    sys.excepthook = handle_sys_exception
    threading.excepthook = handle_thread_exception
