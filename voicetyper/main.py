"""VoiceTyper entry point and system tray application wiring.

Responsibilities:
- Enforce single instance via named Win32 mutex.
- Initialize rotating logging and global uncaught exception hooks.
- Wire together AudioRecorder, Transcriber, Win32HotkeyManager, TextInjector,
  VoiceTyperController, OverlayPill, and SettingsDialog.
- Provide background system tray icon with Pause/Resume, Settings, and Exit.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
import winsound
from typing import Any

# Enable fast downloads via hf-transfer and suppress symlink warnings
if importlib.util.find_spec("hf_transfer") is not None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from voicetyper import __app_name__, __version__
from voicetyper.audio_recorder import AudioRecorder
from voicetyper.config import ConfigManager
from voicetyper.controller import AppState, VoiceTyperController
from voicetyper.hotkey_manager import Win32HotkeyManager, format_hotkey_display
from voicetyper.paths import (
    install_global_exception_handler,
    resource_path,
    setup_logging,
)
from voicetyper.text_injector import WindowsTextInjector
from voicetyper.transcriber import Transcriber
from voicetyper.ui.overlay import OverlayPill
from voicetyper.ui.settings_dialog import SettingsDialog

logger = logging.getLogger("voicetyper")

MUTEX_NAME = "Local\\VoiceTyper_SingleInstance_Mutex"
ERROR_ALREADY_EXISTS = 183


def acquire_single_instance_mutex() -> Any:
    """Ensure only one instance runs per user session on Windows."""
    if sys.platform != "win32":
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    last_err = ctypes.get_last_error()
    if last_err == ERROR_ALREADY_EXISTS or not handle:
        return None
    return handle


class VoiceTyperTrayApp:
    """Manages the lifecycle of VoiceTyper from the Windows system tray."""

    def __init__(self, app: QApplication) -> None:
        self.app = app
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()

        # Initialize audio capture
        self.recorder = AudioRecorder(
            device=self.config.mic_device,
            min_duration_sec=self.config.min_duration_sec,
        )

        # Initialize transcriber
        self.transcriber = Transcriber(
            api_key=self.config.api_key,
            engine=self.config.transcription_engine,
            whisper_model=self.config.whisper_model,
            whisper_device=self.config.whisper_device,
            whisper_compute_type=self.config.whisper_compute_type,
        )

        # Initialize text injector
        self.injector = WindowsTextInjector()

        # Initialize controller
        self.controller = VoiceTyperController(
            config=self.config,
            config_manager=self.config_manager,
            audio_recorder=self.recorder,
            transcriber=self.transcriber,
            text_injector=self.injector,
        )

        # Initialize overlay
        self.overlay = OverlayPill(
            audio_recorder=self.recorder,
            config=self.config,
            config_manager=self.config_manager,
        )

        # Initialize global hotkey manager
        self.hotkey_mgr = Win32HotkeyManager(hotkey_str=self.config.hotkey)
        self.controller.hotkey_manager = self.hotkey_mgr

        # Assets
        self.icon_path = resource_path("assets/icon.png")
        self.start_beep_path = resource_path("assets/start.wav")
        self.stop_beep_path = resource_path("assets/stop.wav")

        self.settings_dialog: SettingsDialog | None = None
        self._init_tray()
        self._wire_signals()

    def _init_tray(self) -> None:
        """Create system tray icon and context menu."""
        if self.icon_path.exists():
            self.tray_icon = QSystemTrayIcon(QIcon(str(self.icon_path)), self.app)
        else:
            self.tray_icon = QSystemTrayIcon(self.app)

        self._update_tray_tooltip()

        # Tray Menu
        self.menu = QMenu()

        # Header info
        title_action = QAction(f"{__app_name__} v{__version__}", self.menu)
        title_action.setEnabled(False)
        self.menu.addAction(title_action)
        self.menu.addSeparator()

        # Settings
        settings_action = QAction("Settings...", self.menu)
        settings_action.triggered.connect(self.open_settings)
        self.menu.addAction(settings_action)

        # Pause / Resume Hotkey
        self.pause_action = QAction("Pause Hotkey", self.menu)
        self.pause_action.setCheckable(True)
        self.pause_action.setChecked(False)
        self.pause_action.triggered.connect(self._toggle_pause_hotkey)
        self.menu.addAction(self.pause_action)

        # Start with Windows
        self.startup_action = QAction("Start with Windows", self.menu)
        self.startup_action.setCheckable(True)
        self.startup_action.setChecked(self.config.start_with_windows)
        self.startup_action.triggered.connect(self._toggle_startup)
        self.menu.addAction(self.startup_action)

        # Reset Overlay Position
        reset_pos_action = QAction("Reset Overlay Position", self.menu)
        reset_pos_action.triggered.connect(self.overlay.reset_position)
        self.menu.addAction(reset_pos_action)

        self.menu.addSeparator()

        # Exit
        exit_action = QAction("Exit", self.menu)
        exit_action.triggered.connect(self.shutdown)
        self.menu.addAction(exit_action)

        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _update_tray_tooltip(self) -> None:
        status = "PAUSED" if self.hotkey_mgr.is_paused() else f"Active ({self.config.hotkey})"
        self.tray_icon.setToolTip(f"{__app_name__} - {status}")

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle click on tray icon."""
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Single click opens settings
            self.open_settings()

    def _wire_signals(self) -> None:
        """Connect all components via Qt signals and slots."""
        # Hotkey events -> Controller
        self.hotkey_mgr.hotkey_down.connect(self.controller.on_hotkey_down)
        self.hotkey_mgr.hotkey_up.connect(self.controller.on_hotkey_up)
        self.hotkey_mgr.hotkey_cancel.connect(self.controller.on_cancel)
        self.controller.state_changed.connect(
            lambda old, new: self.hotkey_mgr.set_recording_active(new == AppState.RECORDING.value)
        )

        # Controller -> Overlay UI
        self.controller.overlay_show_recording.connect(self.overlay.show_recording)
        self.controller.overlay_show_processing.connect(self.overlay.show_processing)
        self.controller.overlay_show_error.connect(self.overlay.show_error)
        self.controller.overlay_hide.connect(self.overlay.show_standby)

        # Overlay Quick Settings Gear Click -> Open Settings Dialog
        self.overlay.settings_clicked.connect(self.open_settings)

        # Controller -> Audio Beeps
        self.controller.play_beep_requested.connect(self._play_sound)

        # Controller -> Text Injector (fallback signal if needed)
        self.controller.text_injection_requested.connect(
            lambda txt: self.injector.inject_text(txt, fallback_typing=self.config.paste_fallback_typing)
        )

        # Overlay Cancel Download Button -> Cancel Transcriber Download
        self.overlay.cancel_download_clicked.connect(self.transcriber.cancel_download)

        # Wire live model download listener to update desktop overlay in real time
        def on_model_dl_progress(msg: str, pct: float | None = None, model_name: str = "") -> None:
            if "cancelled" in msg.lower() or "إلغاء" in msg:
                QTimer.singleShot(0, self.overlay.cancel_active_download)
            elif "error" in msg.lower() or "خطأ" in msg:
                QTimer.singleShot(0, lambda m=msg: self.overlay.finish_download(False, m))
            elif "ready" in msg.lower() or "جاهز" in msg:
                QTimer.singleShot(0, lambda: self.overlay.finish_download(True))
            else:
                p_val = float(pct) if pct is not None else 0.0
                QTimer.singleShot(0, lambda m=msg, p=p_val, n=model_name: self.overlay.update_download_progress(m, p, n))

        if hasattr(self.transcriber, "add_download_listener"):
            self.transcriber.add_download_listener(on_model_dl_progress)

    def _play_sound(self, kind: str) -> None:
        """Play asynchronous start or stop chime sound."""
        if not self.config.start_stop_beeps or sys.platform != "win32":
            return

        sound_file = self.start_beep_path if kind == "start" else self.stop_beep_path
        if sound_file.exists():
            try:
                winsound.PlaySound(str(sound_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as err:
                logger.debug("Sound playback error: %s", err)

    def _toggle_pause_hotkey(self, checked: bool) -> None:
        if checked:
            self.hotkey_mgr.pause()
            self.pause_action.setText("Resume Hotkey")
        else:
            self.hotkey_mgr.resume()
            self.pause_action.setText("Pause Hotkey")
        self._update_tray_tooltip()

    def _toggle_startup(self, checked: bool) -> None:
        self.config.start_with_windows = checked
        self.config_manager.save_config(self.config)

    def open_settings(self) -> None:
        """Open or focus the Settings dialog."""
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(
                self.config,
                self.config_manager,
                overlay=self.overlay,
                transcriber=self.transcriber,
                hotkey_manager=self.hotkey_mgr,
            )
            self.settings_dialog.finished.connect(self._on_settings_closed)

        self.settings_dialog.show()
        self.settings_dialog.activateWindow()
        self.settings_dialog.raise_()

    def _on_settings_closed(self) -> None:
        """Handle updated configuration when SettingsDialog closes."""
        prev_model = self.controller.config.whisper_model
        prev_engine = self.controller.config.transcription_engine
        self.controller.update_config(self.config)
        self.hotkey_mgr.set_hotkey(self.config.hotkey)
        if hasattr(self.hotkey_mgr, "set_toggle_mode"):
            self.hotkey_mgr.set_toggle_mode(self.config.toggle_mode)
        self.startup_action.setChecked(self.config.start_with_windows)
        if not getattr(self.transcriber, "is_downloading", False):
            self.overlay.reposition()
        self._update_tray_tooltip()
        self.settings_dialog = None

        if self.config.transcription_engine == "local" and (
            self.config.whisper_model != prev_model or prev_engine != "local"
        ):
            import threading

            threading.Thread(
                target=self._preload_whisper_model,
                name="WhisperPreloadThread",
                daemon=True,
            ).start()

    def _notify_model_download_complete(self) -> None:
        """Play chime, show tray balloon notification, and update overlay checkmark."""
        # 1. Update overlay with green checkmark
        self.overlay.finish_download(True)

        # 2. Play subtle chime sound
        self._play_sound("stop")

        # 3. Windows System Tray Balloon Notification
        lang = getattr(self.config, "ui_language", "ar")
        title = "VoiceTyper - تم التحميل" if lang == "ar" else "VoiceTyper - Download Complete"
        model_name = getattr(self.config, "whisper_model", "large-v3-turbo")
        msg = (
            f"تم تحميل موديل الذكاء الاصطناعي ({model_name}) بنجاح وهو جاهز للاستخدام الفوري!"
            if lang == "ar"
            else f"AI model ({model_name}) downloaded successfully and ready for use!"
        )
        try:
            self.tray_icon.showMessage(
                title,
                msg,
                QSystemTrayIcon.MessageIcon.Information,
                3500,
            )
        except Exception as err:
            logger.debug("Failed to show tray notification: %s", err)

    def _preload_whisper_model(self) -> None:
        """Preload local faster-whisper model in the background at startup with progress."""
        logger.info("Background preloading local faster-whisper model...")

        def on_preload_progress(msg: str, pct: float | None = None) -> None:
            if "cancelled" in msg.lower() or "إلغاء" in msg:
                QTimer.singleShot(0, self.overlay.cancel_active_download)
            elif "error" in msg.lower() or "خطأ" in msg:
                QTimer.singleShot(0, lambda m=msg: self.overlay.finish_download(False, m))
            elif pct is not None and pct > 0:
                QTimer.singleShot(0, lambda m=msg, p=pct: self.overlay.update_download_progress(m, p))

        from voicetyper.transcriber import is_model_cached

        model_name = getattr(self.config, "whisper_model", "large-v3-turbo")
        was_cached = is_model_cached(model_name)

        success = self.transcriber.preload_model(on_progress=on_preload_progress)
        if success:
            logger.info("Local faster-whisper model preloaded successfully.")
            if not was_cached:
                QTimer.singleShot(0, self._notify_model_download_complete)
            else:
                QTimer.singleShot(0, lambda: self.overlay.finish_download(True))
        else:
            logger.warning("Local faster-whisper model preload did not finish or was skipped.")

    def start(self) -> None:
        """Start background services and global hooks."""
        logger.info("Starting %s v%s...", __app_name__, __version__)

        # Start continuous audio stream for pre-roll
        self.recorder.start_stream()

        # Start low-level keyboard hook
        self.hotkey_mgr.start()

        # Show floating pill overlay in subtle standby mode so it is immediately visible
        self.overlay.show_standby()

        # Start background preload of Whisper model if using local engine
        if self.config.transcription_engine == "local":
            import threading

            threading.Thread(
                target=self._preload_whisper_model,
                name="WhisperPreloadThread",
                daemon=True,
            ).start()

        # Prompt for API key if missing ONLY when using Gemini engine
        if self.config.transcription_engine == "gemini" and not self.config.api_key:
            self.tray_icon.showMessage(
                "VoiceTyper Setup",
                "Gemini API key is required. Please enter your key in Settings.",
                QSystemTrayIcon.MessageIcon.Information,
                5000,
            )
            self.open_settings()
        else:
            engine_label = "Local (faster-whisper)" if self.config.transcription_engine == "local" else "Gemini API"
            self.tray_icon.showMessage(
                f"{__app_name__} Ready",
                f"Hold [{format_hotkey_display(self.config.hotkey)}] to dictate speech.\nEngine: {engine_label}",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )

    def shutdown(self) -> None:
        """Cleanly stop all background threads, hooks, and exit."""
        logger.info("Exiting %s...", __app_name__)
        self.hotkey_mgr.stop()
        self.controller.shutdown()
        self.overlay.hide_overlay()
        self.overlay.close()
        self.tray_icon.hide()
        self.app.quit()


def main() -> int:
    """Application main entry point."""
    setup_logging()
    install_global_exception_handler()

    # Single instance mutex check
    mutex = acquire_single_instance_mutex()
    if sys.platform == "win32" and mutex is None:
        logger.warning("Another instance of VoiceTyper is already running. Exiting.")
        # Create minimal temporary app to notify user
        _ = QApplication(sys.argv)
        QMessageBox.information(
            None,
            "VoiceTyper Already Running",
            "VoiceTyper is already running in your system tray.\nLook for the microphone icon in the taskbar notification area.",
        )
        return 0

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running in system tray

    # Set application icon
    icon_path = resource_path("assets/icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    tray_app = VoiceTyperTrayApp(app)
    tray_app.start()

    ret = app.exec()

    # Close single instance mutex handle
    if mutex and sys.platform == "win32":
        ctypes.windll.kernel32.CloseHandle(mutex)

    return ret


if __name__ == "__main__":
    sys.exit(main())
