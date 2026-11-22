"""Modern dark-themed Settings Dialog for VoiceTyper.

Features:
- Transcription Engine selection: 100% Offline Local GPU/CPU (faster-whisper) vs Google Gemini API.
- Local Whisper Model selector (large-v3-turbo, medium, small, base) with hardware detection badge.
- Interactive keyboard Hotkey Recorder supporting 1-, 2-, and 3-key combinations (F8, Right Alt, Ctrl+Shift, etc.).
- Gemini API Key management with masking, eye toggle, and live validation.
- Microphone input device selection with dynamic refresh.
- Audio beeps on start/stop.
- Language bias hint (Auto, Arabic, English).
- Max recording duration hard cap.
- Text injection mode (Paste vs Unicode typing fallback).
- Toggle dictation mode.
- Custom vocabulary / terminology.
- Windows Startup entry.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QIcon, QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from voicetyper.audio_recorder import AudioRecorder
from voicetyper.config import Config, ConfigManager
from voicetyper.hotkey_manager import format_hotkey_display
from voicetyper.paths import resource_path
from voicetyper.transcriber import (
    MODELS_CATALOG,
    Transcriber,
    get_optimal_device,
    is_model_cached,
)
from voicetyper.ui.overlay import OverlayState, ThemePreviewWidget

logger = logging.getLogger(__name__)

DARK_STYLESHEET = """
QDialog {
    background-color: #1E1F20;
    color: #E8EAED;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QLabel {
    color: #E8EAED;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #2D2F31;
    color: #FFFFFF;
    border: 1px solid #3C4043;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #8AB4F8;
    background-color: #35373A;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #2D2F31;
    color: #FFFFFF;
    selection-background-color: #4285F4;
    border: 1px solid #3C4043;
    outline: none;
}
QCheckBox {
    color: #E8EAED;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #5F6368;
    background-color: #2D2F31;
}
QCheckBox::indicator:checked {
    background-color: #4285F4;
    border-color: #4285F4;
}
QPushButton {
    background-color: #2D2F31;
    color: #E8EAED;
    border: 1px solid #3C4043;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #3C4043;
}
QPushButton#primaryButton {
    background-color: #4285F4;
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background-color: #1A73E8;
}
QFrame#separator {
    background-color: #3C4043;
    max-height: 1px;
}
"""


class HotkeyRecorderWidget(QWidget):
    """Interactive hotkey capture widget supporting 1, 2, or 3 key combinations."""

    hotkey_changed = pyqtSignal(str)  # type: ignore[call-arg]
    _listener_key_signal = pyqtSignal(str, bool)  # type: ignore[call-arg] # (key_name, is_down)

    def __init__(
        self,
        current_hotkey: str = "ctrl+shift",
        hotkey_manager: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self._current_hotkey = current_hotkey
        self._hotkey_manager = hotkey_manager
        self._is_recording = False
        self._held_keys: list[str] = []
        self._max_combo: list[str] = []
        self._listener: Any = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Hotkey Display
        self.display_label = QLabel(format_hotkey_display(current_hotkey))
        self.display_label.setStyleSheet(
            "background-color: #2D2F31; color: #FFFFFF; font-weight: bold; "
            "padding: 6px 14px; border-radius: 6px; border: 1px solid #3C4043; min-width: 130px;"
        )
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Record Button
        self.record_btn = QPushButton("Record Hotkey")
        self.record_btn.setFixedWidth(135)
        self.record_btn.clicked.connect(self._toggle_recording)

        # Presets Dropdown
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Presets...", "")
        presets = [
            ("Ctrl + Shift (Default)", "ctrl+shift"),
            ("Ctrl + Space", "ctrl+space"),
            ("Right Alt (AltGr)", "right alt"),
            ("F8 Key", "f8"),
            ("F9 Key", "f9"),
            ("CapsLock", "capslock"),
            ("Insert Key", "insert"),
            ("Ctrl + Shift + V", "ctrl+shift+v"),
            ("Alt + Z", "alt+z"),
        ]
        for label, val in presets:
            self.preset_combo.addItem(label, val)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)

        layout.addWidget(self.display_label)
        layout.addWidget(self.record_btn)
        layout.addWidget(self.preset_combo)

        # Connect internal signal for thread-safe UI updates from pynput thread
        self._listener_key_signal.connect(self._handle_listener_key)

    def get_hotkey(self) -> str:
        """Return the normalized hotkey string."""
        return self._current_hotkey

    def set_hotkey(self, hotkey_str: str) -> None:
        """Update hotkey display and stored value."""
        self._current_hotkey = hotkey_str.strip()
        self.display_label.setText(format_hotkey_display(self._current_hotkey))
        self.hotkey_changed.emit(self._current_hotkey)

    def _on_preset_selected(self, index: int) -> None:
        val = self.preset_combo.itemData(index)
        if val:
            self.set_hotkey(val)
            self.preset_combo.setCurrentIndex(0)

    def _toggle_recording(self) -> None:
        if self._is_recording:
            self._cancel_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self._is_recording = True
        self._held_keys.clear()
        self._max_combo.clear()
        if self._hotkey_manager is not None and hasattr(self._hotkey_manager, "pause"):
            self._hotkey_manager.pause()
        self.display_label.setText("Press keys... (Esc)")
        self.display_label.setStyleSheet(
            "background-color: #1E3A8A; color: #FFFFFF; font-weight: bold; "
            "padding: 6px 14px; border-radius: 6px; border: 1px solid #38BDF8; min-width: 130px;"
        )
        self.record_btn.setText("Cancel (Esc)")
        self.record_btn.setStyleSheet("background-color: #B91C1C; color: #FFFFFF; font-weight: bold;")

        # Start pynput keyboard hook listener for hotkey recording
        try:
            from pynput import keyboard

            self._listener = keyboard.Listener(
                on_press=self._on_pynput_press,
                on_release=self._on_pynput_release,
            )
            self._listener.daemon = True
            self._listener.start()
        except Exception as err:
            logger.warning("Could not start pynput listener for hotkey recording: %s", err)

    def _cancel_recording(self) -> None:
        self._is_recording = False
        self._stop_listener()
        self.display_label.setText(format_hotkey_display(self._current_hotkey))
        self.display_label.setStyleSheet(
            "background-color: #2D2F31; color: #FFFFFF; font-weight: bold; "
            "padding: 6px 14px; border-radius: 6px; border: 1px solid #3C4043; min-width: 130px;"
        )
        self.record_btn.setText("Record Hotkey")
        self.record_btn.setStyleSheet("")

    def _stop_listener(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        if self._hotkey_manager is not None and hasattr(self._hotkey_manager, "resume"):
            self._hotkey_manager.resume()

    def _on_pynput_press(self, key: Any) -> None:
        if not self._is_recording:
            return
        name = self._normalize_pynput_key(key)
        if name:
            self._listener_key_signal.emit(name, True)

    def _on_pynput_release(self, key: Any) -> None:
        if not self._is_recording:
            return
        name = self._normalize_pynput_key(key)
        if name:
            self._listener_key_signal.emit(name, False)

    def _normalize_pynput_key(self, key: Any) -> str:
        """Map pynput Key or KeyCode to canonical string."""
        from pynput.keyboard import Key

        if key == Key.esc:
            return "esc"
        if key in (Key.ctrl, Key.ctrl_l, Key.ctrl_r) or getattr(key, "name", "") in ("ctrl", "ctrl_l", "ctrl_r"):
            return "ctrl"
        if key in (Key.shift, Key.shift_r) or getattr(key, "name", "") in ("shift", "shift_l", "shift_r"):
            return "shift"
        if key in (Key.alt, Key.alt_l) or getattr(key, "name", "") in ("alt", "alt_l"):
            return "alt"
        if key in (Key.alt_r, Key.alt_gr) or getattr(key, "name", "") in ("alt_r", "alt_gr"):
            return "right alt"
        if key in (Key.cmd, Key.cmd_r) or getattr(key, "name", "") in ("cmd", "cmd_l", "cmd_r"):
            return "win"
        if key == Key.space:
            return "space"
        if key == Key.caps_lock:
            return "capslock"
        if key == Key.insert:
            return "insert"
        if key == Key.delete:
            return "delete"
        if key == Key.tab:
            return "tab"

        # Check F1-F24
        name = getattr(key, "name", "")
        if name and name.startswith("f") and name[1:].isdigit():
            return name.lower()

        # Check characters
        char = getattr(key, "char", "")
        if char:
            return char.lower()

        # Check virtual key code fallback
        vk = getattr(key, "vk", None)
        if vk is not None:
            if 0x41 <= vk <= 0x5A:
                return chr(vk).lower()
            if 0x30 <= vk <= 0x39:
                return chr(vk)

        return ""

    def _handle_listener_key(self, key_name: str, is_down: bool) -> None:
        """Thread-safe handler for key events."""
        if not self._is_recording:
            return

        if key_name == "esc":
            self._cancel_recording()
            return

        if is_down:
            if key_name not in self._held_keys and len(self._held_keys) < 3:
                self._held_keys.append(key_name)
                combo_str = "+".join(self._held_keys)
                self.display_label.setText(format_hotkey_display(combo_str))
                if len(self._held_keys) > len(self._max_combo):
                    self._max_combo = list(self._held_keys)
        else:
            # On release, if we have a captured combination, finalize it
            if self._max_combo:
                final_str = "+".join(self._max_combo)
                self._stop_listener()
                self._is_recording = False
                self.set_hotkey(final_str)
                self.record_btn.setText("Record Hotkey")
                self.record_btn.setStyleSheet("")
                self.display_label.setStyleSheet(
                    "background-color: #2D2F31; color: #FFFFFF; font-weight: bold; "
                    "padding: 6px 14px; border-radius: 6px; border: 1px solid #3C4043; min-width: 130px;"
                )
            if key_name in self._held_keys:
                self._held_keys.remove(key_name)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        """Support Qt key events for direct typing or headless unit tests."""
        if not event or not self._is_recording:
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key.Key_Escape:
            self._cancel_recording()
            event.accept()
            return

        key_name = ""
        k = event.key()
        if k in (Qt.Key.Key_Control, Qt.Key.Key_Control):
            key_name = "ctrl"
        elif k == Qt.Key.Key_Shift:
            key_name = "shift"
        elif k == Qt.Key.Key_Alt:
            key_name = "alt"
        elif k == Qt.Key.Key_Space:
            key_name = "space"
        elif k == Qt.Key.Key_CapsLock:
            key_name = "capslock"
        elif k == Qt.Key.Key_Insert:
            key_name = "insert"
        elif Qt.Key.Key_F1 <= k <= Qt.Key.Key_F24:
            f_num = k - Qt.Key.Key_F1 + 1
            key_name = f"f{f_num}"
        elif Qt.Key.Key_A <= k <= Qt.Key.Key_Z:
            key_name = chr(k).lower()

        if key_name:
            self._handle_listener_key(key_name, True)
            self._handle_listener_key(key_name, False)
            event.accept()
        else:
            super().keyPressEvent(event)


class SettingsDialog(QDialog):
    """VoiceTyper configuration dialog."""

    model_progress_signal = pyqtSignal(str, object)  # type: ignore[call-arg]  # (status_text, progress_pct)

    def __init__(
        self,
        config: Config,
        config_manager: ConfigManager,
        overlay: Any = None,
        transcriber: Any = None,
        hotkey_manager: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.config_manager = config_manager
        self.overlay = overlay
        self.transcriber = transcriber
        self.hotkey_manager = hotkey_manager
        self._initial_theme = getattr(self.config, "overlay_theme", "classic")
        self._current_preview_state = OverlayState.RECORDING

        self.model_progress_signal.connect(self._on_model_progress_updated)

        self.setWindowTitle("VoiceTyper Settings")
        self.setMinimumWidth(540)
        self.setStyleSheet(DARK_STYLESHEET)

        # Set window icon if available
        icon_path = resource_path("assets/icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._init_ui()
        self._load_values()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)

        # Title / Header
        title_label = QLabel("VoiceTyper Settings")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        subtitle_label = QLabel("Configure your Push-to-Talk voice dictation preferences")
        subtitle_label.setStyleSheet("color: #9AA0A6; font-size: 12px;")
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(sep)

        form_layout = QFormLayout()
        form_layout.setSpacing(14)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        # 1. Transcription Engine
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Local (faster-whisper) - 100% Offline GPU/CPU", "local")
        self.engine_combo.addItem("Google Gemini API - Cloud", "gemini")
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        form_layout.addRow("Transcription Engine:", self.engine_combo)

        # 2. Local Whisper Model & Hardware Info
        self.whisper_container = QWidget()
        whisper_layout = QVBoxLayout(self.whisper_container)
        whisper_layout.setContentsMargins(0, 0, 0, 0)
        whisper_layout.setSpacing(6)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(8)

        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.currentIndexChanged.connect(self._on_whisper_model_changed)
        model_row.addWidget(self.whisper_model_combo)

        # Hardware readout badge
        _, _, dev_label = get_optimal_device()
        self.device_badge = QLabel(f"Active Hardware: {dev_label}")
        self.device_badge.setStyleSheet(
            "color: #38BDF8; font-weight: 600; font-size: 11px; padding: 2px 8px; "
            "background-color: rgba(56, 189, 248, 25); border-radius: 4px; border: 1px solid rgba(56, 189, 248, 60);"
        )
        model_row.addWidget(self.device_badge)

        whisper_layout.addLayout(model_row)

        # Status & cache refresh row
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)

        self.model_status_label = QLabel("Model ready ✓")
        self.model_status_label.setStyleSheet("color: #9AA0A6; font-size: 11px;")
        status_row.addWidget(self.model_status_label)

        status_row.addStretch()

        self.download_now_btn = QPushButton("Download Model ⬇")
        self.download_now_btn.setFixedHeight(22)
        self.download_now_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 8px; background-color: #1A73E8; color: #FFFFFF; border-radius: 3px; border: none; font-weight: bold; }"
            "QPushButton:hover { background-color: #4285F4; }"
        )
        self.download_now_btn.clicked.connect(self._start_manual_download)
        self.download_now_btn.hide()
        status_row.addWidget(self.download_now_btn)

        self.cancel_download_btn = QPushButton("Cancel ⏹")
        self.cancel_download_btn.setFixedHeight(22)
        self.cancel_download_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 8px; background-color: #C5221F; color: #FFFFFF; border-radius: 3px; border: none; font-weight: bold; }"
            "QPushButton:hover { background-color: #D93025; }"
            "QPushButton:disabled { background-color: #5F6368; color: #9AA0A6; }"
        )
        self.cancel_download_btn.clicked.connect(self._cancel_active_download)
        self.cancel_download_btn.hide()
        status_row.addWidget(self.cancel_download_btn)

        self.refresh_cache_btn = QPushButton("Check Cache")
        self.refresh_cache_btn.setFixedHeight(22)
        self.refresh_cache_btn.setStyleSheet(
            "font-size: 11px; padding: 2px 8px; background-color: #2D2F31; border: 1px solid #3C4043; color: #E8EAED;"
        )
        self.refresh_cache_btn.clicked.connect(self._refresh_model_cache_labels)
        status_row.addWidget(self.refresh_cache_btn)

        whisper_layout.addLayout(status_row)

        # Download / load progress bar with clear styling
        self.model_progress_bar = QProgressBar()
        self.model_progress_bar.setFixedHeight(12)
        self.model_progress_bar.setTextVisible(True)
        self.model_progress_bar.setStyleSheet(
            "QProgressBar { background-color: #2D2F31; border-radius: 4px; border: none; text-align: center; color: #FFFFFF; font-size: 9px; font-weight: bold; }"
            "QProgressBar::chunk { background-color: #4285F4; border-radius: 4px; }"
        )
        self.model_progress_bar.hide()
        whisper_layout.addWidget(self.model_progress_bar)

        form_layout.addRow("Local Whisper Model:", self.whisper_container)

        # 3. Gemini API Key (Required for Gemini, Optional for Local)
        self.key_container = QWidget()
        key_layout = QHBoxLayout(self.key_container)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(8)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Gemini API Key (Optional when using Local Engine)")

        self.toggle_key_btn = QPushButton("Show")
        self.toggle_key_btn.setFixedWidth(60)
        self.toggle_key_btn.clicked.connect(self._toggle_api_key_visibility)

        self.test_key_btn = QPushButton("Test")
        self.test_key_btn.setFixedWidth(60)
        self.test_key_btn.clicked.connect(self._test_api_key)

        key_layout.addWidget(self.api_key_input)
        key_layout.addWidget(self.toggle_key_btn)
        key_layout.addWidget(self.test_key_btn)

        self.api_key_row_label = QLabel("Gemini API Key:")
        form_layout.addRow(self.api_key_row_label, self.key_container)

        # 4. Global Hotkey (Interactive Keyboard Recorder)
        self.hotkey_recorder = HotkeyRecorderWidget(
            current_hotkey=self.config.hotkey,
            hotkey_manager=self.hotkey_manager,
        )
        form_layout.addRow("PTT Hotkey:", self.hotkey_recorder)

        # 5. Overlay Theme (Interactive Live Preview)
        theme_container = QWidget()
        theme_vbox = QVBoxLayout(theme_container)
        theme_vbox.setContentsMargins(0, 0, 0, 0)
        theme_vbox.setSpacing(8)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Classic (Default) - الشكل الأصلي", "classic")
        self.theme_combo.addItem("Neon Glass Pill (Image 1)", "neon_glass")
        self.theme_combo.addItem("Matte Dynamic Island (Image 2)", "matte_island")
        self.theme_combo.addItem("Fluent Mica Card (Image 3)", "fluent_mica")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        theme_vbox.addWidget(self.theme_combo)

        # Embedded live visualizer preview box
        preview_frame = QFrame()
        preview_frame.setObjectName("themePreviewFrame")
        preview_frame.setStyleSheet(
            "QFrame#themePreviewFrame { "
            "  background-color: #121316; "
            "  border: 1px solid #303236; "
            "  border-radius: 8px; "
            "  padding: 8px; "
            "}"
        )
        p_layout = QVBoxLayout(preview_frame)
        p_layout.setContentsMargins(8, 8, 8, 8)
        p_layout.setSpacing(8)

        self.theme_preview = ThemePreviewWidget(theme=self.config.overlay_theme)
        p_layout.addWidget(self.theme_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        # State switcher buttons: Recording, Processing, Error, Standby
        state_bar = QHBoxLayout()
        state_bar.setSpacing(6)
        state_bar.setContentsMargins(0, 0, 0, 0)

        self.state_btns: dict[OverlayState, QPushButton] = {}
        preview_states = [
            (OverlayState.RECORDING, "🔴 Recording"),
            (OverlayState.PROCESSING, "🌊 Processing"),
            (OverlayState.ERROR, "⚠️ Error"),
            (OverlayState.STANDBY, "⚪ Standby"),
        ]
        for st, label in preview_states:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, s=st: self._on_preview_state_selected(s))
            self.state_btns[st] = btn
            state_bar.addWidget(btn)

        p_layout.addLayout(state_bar)
        theme_vbox.addWidget(preview_frame)

        form_layout.addRow("Overlay Theme:", theme_container)

        # 5. Microphone Selector
        mic_container = QWidget()
        mic_layout = QHBoxLayout(mic_container)
        mic_layout.setContentsMargins(0, 0, 0, 0)
        mic_layout.setSpacing(8)

        self.mic_combo = QComboBox()
        self.refresh_mic_btn = QPushButton("Refresh")
        self.refresh_mic_btn.setFixedWidth(70)
        self.refresh_mic_btn.clicked.connect(self._refresh_microphones)

        mic_layout.addWidget(self.mic_combo)
        mic_layout.addWidget(self.refresh_mic_btn)

        form_layout.addRow("Microphone:", mic_container)

        # 6. Language Hint
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Auto (Arabic, English & Mixed)", "Auto")
        self.lang_combo.addItem("Arabic (Egyptian & Modern Standard)", "Arabic")
        self.lang_combo.addItem("English", "English")
        form_layout.addRow("Language Bias:", self.lang_combo)

        # 7. Max Recording Duration
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 300)
        self.duration_spin.setSuffix(" seconds")
        form_layout.addRow("Max Duration:", self.duration_spin)

        # 8. English Capitalization
        self.capitalization_combo = QComboBox()
        self.capitalization_combo.addItem("Capitalize Every Word (e.g. Hello World)", "word")
        self.capitalization_combo.addItem("Capitalize Sentences (e.g. Hello world. How are you)", "sentence")
        self.capitalization_combo.addItem("None (Keep as spoken)", "none")
        form_layout.addRow("English Casing:", self.capitalization_combo)

        # 9. Custom Vocabulary
        self.vocab_input = QLineEdit()
        self.vocab_input.setPlaceholderText("Comma-separated terms (e.g. PySide6, Docker, FastAPI)")
        form_layout.addRow("Custom Vocabulary:", self.vocab_input)

        main_layout.addLayout(form_layout)

        # Checkboxes
        check_layout = QVBoxLayout()
        check_layout.setSpacing(10)

        self.beeps_checkbox = QCheckBox("Play subtle chime sound on recording start and stop")
        self.clipboard_keep_checkbox = QCheckBox("Keep transcript in Clipboard & Win+V History")
        self.startup_checkbox = QCheckBox("Start VoiceTyper automatically when Windows starts")
        self.typing_checkbox = QCheckBox("Type text via Unicode keystrokes instead of Ctrl+V paste")
        self.toggle_checkbox = QCheckBox("Toggle dictation mode (press once to start, press again to stop)")

        check_layout.addWidget(self.beeps_checkbox)
        check_layout.addWidget(self.clipboard_keep_checkbox)
        check_layout.addWidget(self.startup_checkbox)
        check_layout.addWidget(self.typing_checkbox)
        check_layout.addWidget(self.toggle_checkbox)
        main_layout.addLayout(check_layout)

        # Action Buttons (Reset Position / Cancel / Save)
        btn_layout = QHBoxLayout()

        reset_pos_btn = QPushButton("Reset Overlay Position")
        reset_pos_btn.clicked.connect(self._reset_overlay_position)
        btn_layout.addWidget(reset_pos_btn)

        desktop_shortcut_btn = QPushButton("Desktop Shortcut 🖥️")
        desktop_shortcut_btn.setToolTip("Create or refresh VoiceTyper shortcut on your Windows Desktop")
        desktop_shortcut_btn.clicked.connect(self._create_desktop_shortcut)
        btn_layout.addWidget(desktop_shortcut_btn)

        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._save_and_close)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        main_layout.addLayout(btn_layout)

    def _on_theme_changed(self, index: int) -> None:
        """Apply theme immediately for live preview on both widget and desktop overlay."""
        theme = str(self.theme_combo.itemData(index))
        self.config.overlay_theme = theme
        if hasattr(self, "theme_preview"):
            self.theme_preview.set_theme(theme)
        if self.overlay is not None:
            self.overlay.set_theme(theme)
            self.overlay.preview_state(self._current_preview_state)

    def _on_preview_state_selected(self, state: OverlayState) -> None:
        """Switch the preview state on both the embedded widget and desktop overlay."""
        self._current_preview_state = state
        if hasattr(self, "theme_preview"):
            self.theme_preview.set_preview_state(state)
        if self.overlay is not None:
            self.overlay.preview_state(state)
        self._update_preview_button_styles()

    def _update_preview_button_styles(self) -> None:
        """Highlight active preview state button."""
        for st, btn in getattr(self, "state_btns", {}).items():
            if st == self._current_preview_state:
                btn.setStyleSheet(
                    "background-color: #4285F4; color: #FFFFFF; font-weight: bold; border: 1px solid #1A73E8; border-radius: 4px; padding: 3px 8px;"
                )
            else:
                btn.setStyleSheet(
                    "background-color: #2D2F31; color: #9AA0A6; border: 1px solid #3C4043; border-radius: 4px; padding: 3px 8px;"
                )

    def reject(self) -> None:
        """Revert previewed theme if user cancels without saving."""
        if self.overlay is not None:
            self.overlay.set_theme(self._initial_theme)
            self.overlay.show_standby()
        super().reject()

    def closeEvent(self, event: Any) -> None:
        """Ensure overlay returns to standby when settings dialog is closed."""
        if self.overlay is not None:
            self.overlay.set_theme(self._initial_theme)
            self.overlay.show_standby()
        super().closeEvent(event)

    def _on_engine_changed(self, index: int) -> None:
        """Update UI fields based on active transcription engine."""
        engine = self.engine_combo.itemData(index)
        if engine == "local":
            self.whisper_container.setEnabled(True)
            self.api_key_row_label.setText("Gemini API Key (Optional):")
            self.api_key_input.setPlaceholderText("Gemini API Key (Optional for Local Engine)")
        else:
            self.whisper_container.setEnabled(False)
            self.api_key_row_label.setText("Gemini API Key (Required):")
            self.api_key_input.setPlaceholderText("Paste Gemini API Key from Google AI Studio")

    def _refresh_model_cache_labels(self) -> None:
        """Populate or update model items showing [Downloaded ✓] vs download size."""
        current_data = (
            self.whisper_model_combo.currentData()
            if self.whisper_model_combo.count() > 0
            else self.config.whisper_model
        )
        self.whisper_model_combo.blockSignals(True)
        self.whisper_model_combo.clear()

        for model_id, title, size in MODELS_CATALOG:
            cached = is_model_cached(model_id)
            tag = "[Downloaded ✓]" if cached else f"({size})"
            display_text = f"{title} {tag}"
            self.whisper_model_combo.addItem(display_text, model_id)

        idx = self.whisper_model_combo.findData(current_data)
        if idx >= 0:
            self.whisper_model_combo.setCurrentIndex(idx)
        else:
            self.whisper_model_combo.setCurrentIndex(0)
        self.whisper_model_combo.blockSignals(False)

        sel_model = str(self.whisper_model_combo.currentData())
        if is_model_cached(sel_model):
            self.model_status_label.setText("Model cached on disk [Ready to load]")
            self.model_progress_bar.hide()
            self.download_now_btn.hide()
            self.cancel_download_btn.hide()
        else:
            sz = "~1.5 GB"
            for m, _, s in MODELS_CATALOG:
                if m == sel_model:
                    sz = s
                    break
            self.model_status_label.setText(f"Not cached ({sz}). Ready to download.")
            self.download_now_btn.show()
            self.download_now_btn.setEnabled(True)
            self.model_progress_bar.hide()
            self.cancel_download_btn.hide()

    def _on_whisper_model_changed(self, index: int) -> None:
        """Handle user selecting a different Whisper model with status label update."""
        model_id = str(self.whisper_model_combo.itemData(index))
        if not model_id:
            return

        cached = is_model_cached(model_id)
        if cached:
            self.model_status_label.setText("Model cached on disk [Ready to load]")
            self.model_progress_bar.hide()
            self.download_now_btn.hide()
            self.cancel_download_btn.hide()
        else:
            sz = "~1.5 GB"
            for m, _, s in MODELS_CATALOG:
                if m == model_id:
                    sz = s
                    break
            self.model_status_label.setText(f"Not cached ({sz}). Ready to download.")
            self.download_now_btn.show()
            self.download_now_btn.setEnabled(True)
            self.model_progress_bar.hide()
            self.cancel_download_btn.hide()

    def _start_manual_download(self) -> None:
        """Trigger background download of the selected model with progress and cancellation."""
        sel_model = str(self.whisper_model_combo.currentData())
        if is_model_cached(sel_model):
            self.model_status_label.setText("Model already cached on disk ✓")
            self.download_now_btn.hide()
            return

        if self.transcriber is not None:
            self.transcriber.reset_cancel_download()
            self.transcriber.set_whisper_model(sel_model)

        self.download_now_btn.hide()
        self.cancel_download_btn.setEnabled(True)
        self.cancel_download_btn.setText("Cancel ⏹")
        self.cancel_download_btn.show()
        self.model_progress_bar.show()
        self.model_progress_bar.setRange(0, 100)
        self.model_progress_bar.setValue(0)
        self.model_status_label.setText(f"Connecting to Hugging Face to download '{sel_model}'...")

        import threading

        def bg_download() -> None:
            def progress_cb(text: str, pct: float | None = None) -> None:
                self.model_progress_signal.emit(text, pct)

            if self.transcriber is not None:
                self.transcriber.preload_model(on_progress=progress_cb)

        threading.Thread(target=bg_download, name="ManualModelDownloader", daemon=True).start()

    def _cancel_active_download(self) -> None:
        """Cancel ongoing model download immediately."""
        if self.transcriber is not None:
            self.transcriber.cancel_download()
        self.cancel_download_btn.setEnabled(False)
        self.cancel_download_btn.setText("Cancelling...")
        self.model_status_label.setText("Cancelling download... (جاري الإلغاء)")

    @pyqtSlot(str, object)
    def _on_model_progress_updated(self, status_text: str, progress_pct: float | None) -> None:
        """Update status label and progress bar without blocking GUI."""
        self.model_status_label.setText(status_text)
        is_cancelled = "cancelled" in status_text.lower() or "إلغاء" in status_text
        is_ready = "ready" in status_text.lower() or "✓" in status_text

        if is_cancelled:
            self.model_progress_bar.hide()
            self.cancel_download_btn.hide()
            self.download_now_btn.show()
            self.download_now_btn.setEnabled(True)
            self._refresh_model_cache_labels()
            return

        if is_ready:
            self.model_progress_bar.hide()
            self.cancel_download_btn.hide()
            self.download_now_btn.hide()
            self._refresh_model_cache_labels()
            return

        if progress_pct is not None:
            self.model_progress_bar.show()
            self.cancel_download_btn.show()
            self.download_now_btn.hide()
            self.model_progress_bar.setRange(0, 100)
            self.model_progress_bar.setValue(int(progress_pct))
            if progress_pct >= 100.0:
                self.model_progress_bar.hide()
                self.cancel_download_btn.hide()
                self._refresh_model_cache_labels()
        else:
            if "loading" in status_text.lower():
                self.model_progress_bar.show()
                self.model_progress_bar.setRange(0, 0)
                self.cancel_download_btn.hide()

    def _refresh_microphones(self) -> None:
        """Scan and populate available input devices with clean, deduplicated names."""
        self.mic_combo.clear()
        self.mic_combo.addItem("Default System Microphone", None)

        devices = AudioRecorder.get_input_devices()
        selected_index = 0

        for dev in devices:
            # Clean label without any "(Device X)" suffix
            label = dev["name"]
            self.mic_combo.addItem(label, dev["index"])
            if self.config.mic_device == dev["index"] or (
                isinstance(self.config.mic_device, str) and self.config.mic_device == dev["name"]
            ):
                selected_index = self.mic_combo.count() - 1

        self.mic_combo.setCurrentIndex(selected_index)

    def _load_values(self) -> None:
        """Populate widgets with current Config values."""
        # Engine
        idx = self.engine_combo.findData(self.config.transcription_engine)
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)
        else:
            self.engine_combo.setCurrentIndex(0)
        self._on_engine_changed(self.engine_combo.currentIndex())

        # Whisper Model
        self._refresh_model_cache_labels()
        idx_model = self.whisper_model_combo.findData(self.config.whisper_model)
        if idx_model >= 0:
            self.whisper_model_combo.setCurrentIndex(idx_model)
        elif "large" in self.config.whisper_model or "turbo" in self.config.whisper_model:
            self.whisper_model_combo.setCurrentIndex(0)

        # Overlay Theme
        theme_idx = self.theme_combo.findData(self.config.overlay_theme)
        if theme_idx >= 0:
            self.theme_combo.setCurrentIndex(theme_idx)
        else:
            self.theme_combo.setCurrentIndex(0)
        self._update_preview_button_styles()
        if hasattr(self, "theme_preview"):
            self.theme_preview.set_theme(self.config.overlay_theme)
            self.theme_preview.set_preview_state(self._current_preview_state)
        if self.overlay is not None:
            self.overlay.preview_state(self._current_preview_state)

        # Gemini API Key
        self.api_key_input.setText(self.config.api_key)

        # Hotkey
        self.hotkey_recorder.set_hotkey(self.config.hotkey)

        # Microphones
        self._refresh_microphones()

        # Language Hint
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == self.config.language_hint:
                self.lang_combo.setCurrentIndex(i)
                break

        # Max duration
        self.duration_spin.setValue(int(self.config.max_duration_sec))

        # Vocabulary
        self.vocab_input.setText(self.config.custom_vocabulary)

        # English Capitalization
        idx_cap = self.capitalization_combo.findData(self.config.capitalize_english)
        if idx_cap >= 0:
            self.capitalization_combo.setCurrentIndex(idx_cap)

        # Checkboxes
        self.beeps_checkbox.setChecked(self.config.start_stop_beeps)
        self.clipboard_keep_checkbox.setChecked(not self.config.restore_clipboard)
        self.startup_checkbox.setChecked(self.config.start_with_windows)
        self.typing_checkbox.setChecked(self.config.paste_fallback_typing)
        self.toggle_checkbox.setChecked(self.config.toggle_mode)

    def _toggle_api_key_visibility(self) -> None:
        if self.api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.toggle_key_btn.setText("Hide")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.toggle_key_btn.setText("Show")

    def _test_api_key(self) -> None:
        """Quickly test API key connectivity with Gemini."""
        key = self.api_key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "API Key Missing", "Please enter an API key first.")
            return

        self.test_key_btn.setEnabled(False)
        self.test_key_btn.setText("Testing...")

        try:
            transcriber = Transcriber(api_key=key, engine="gemini")
            client = transcriber._get_client()
            _ = client.models.get(model=transcriber.model_name)
            QMessageBox.information(self, "API Key Valid", "Gemini API key is valid and connected successfully!")
        except Exception as err:
            QMessageBox.critical(self, "API Key Failed", f"Validation failed:\n{err}")
        finally:
            self.test_key_btn.setEnabled(True)
            self.test_key_btn.setText("Test")

    def _reset_overlay_position(self) -> None:
        """Reset pill position to bottom-right above taskbar clock."""
        self.config_manager.reset_overlay_position()
        self.config.overlay_pos = None
        QMessageBox.information(
            self,
            "Position Reset",
            "Overlay position reset to default (bottom-right above the taskbar clock).",
        )

    def _create_desktop_shortcut(self) -> None:
        """Create or update a Desktop shortcut for VoiceTyper."""
        try:
            import os
            import sys
            import winshell
            from win32com.client import Dispatch

            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "VoiceTyper.lnk")

            if getattr(sys, "frozen", False):
                target = sys.executable
                work_dir = os.path.dirname(target)
                args = ""
            else:
                repo_root = resource_path("").resolve().parent
                standalone_exe = repo_root / "dist" / "VoiceTyper" / "VoiceTyper.exe"
                if standalone_exe.exists():
                    target = str(standalone_exe)
                    work_dir = str(standalone_exe.parent)
                    args = ""
                else:
                    target = sys.executable
                    work_dir = str(repo_root)
                    args = f'"{work_dir}\\main.py"'

            shell = Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = target
            if args:
                shortcut.Arguments = args
            shortcut.WorkingDirectory = work_dir
            icon_file = resource_path("assets/icon.ico")
            if icon_file.exists():
                shortcut.IconLocation = str(icon_file)
            shortcut.Description = "VoiceTyper - Windows Voice-to-Text Dictation"
            shortcut.save()

            QMessageBox.information(
                self,
                "Desktop Shortcut",
                "Desktop shortcut created successfully on your Windows Desktop!\n\nتم إنشاء اختصار VoiceTyper على سطح المكتب بنجاح.",
            )
        except Exception as err:
            QMessageBox.warning(
                self,
                "Desktop Shortcut Error",
                f"Failed to create desktop shortcut:\n{err}",
            )

    def _save_and_close(self) -> None:
        """Validate and persist updated configuration."""
        engine = str(self.engine_combo.currentData())
        new_key = self.api_key_input.text().strip()

        # Validate API key ONLY if Gemini engine is selected
        if engine == "gemini" and not new_key:
            reply = QMessageBox.question(
                self,
                "API Key Missing",
                "You selected Google Gemini API but haven't entered an API key.\n\n"
                "Would you like to switch to 100% Offline Local Engine (faster-whisper) instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.engine_combo.setCurrentIndex(0)
                engine = "local"
            else:
                return

        new_model = str(self.whisper_model_combo.currentData())
        if self.config.whisper_model != new_model:
            if self.transcriber and hasattr(self.transcriber, "set_whisper_model"):
                self.transcriber.set_whisper_model(new_model)
                if not is_model_cached(new_model):
                    import threading

                    def bg_load() -> None:
                        def progress_cb(text: str, pct: float | None = None) -> None:
                            self.model_progress_signal.emit(text, pct)

                        self.transcriber.preload_model(on_progress=progress_cb)

                    threading.Thread(target=bg_load, name="SettingsModelLoader", daemon=True).start()

        # Update config fields
        self.config.transcription_engine = engine
        self.config.whisper_model = new_model
        self.config.overlay_theme = str(self.theme_combo.currentData())
        self.config.api_key = new_key
        self.config.hotkey = self.hotkey_recorder.get_hotkey()
        self.config.mic_device = self.mic_combo.currentData()
        self.config.language_hint = str(self.lang_combo.currentData())
        self.config.capitalize_english = str(self.capitalization_combo.currentData())
        self.config.max_duration_sec = float(self.duration_spin.value())
        self.config.custom_vocabulary = self.vocab_input.text().strip()
        self.config.start_stop_beeps = self.beeps_checkbox.isChecked()
        self.config.restore_clipboard = not self.clipboard_keep_checkbox.isChecked()
        self.config.start_with_windows = self.startup_checkbox.isChecked()
        self.config.paste_fallback_typing = self.typing_checkbox.isChecked()
        self.config.toggle_mode = self.toggle_checkbox.isChecked()

        # Save to disk, keyring, and registry
        self.config_manager.save_config(self.config)
        self._initial_theme = self.config.overlay_theme
        if self.overlay is not None:
            self.overlay.show_standby()
        logger.info(
            "Saved settings successfully (engine=%s, model=%s, hotkey=%s)",
            self.config.transcription_engine,
            self.config.whisper_model,
            self.config.hotkey,
        )
        self.accept()
