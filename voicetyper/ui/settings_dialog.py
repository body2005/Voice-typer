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
from PyQt6.QtGui import QColor, QFont, QIcon, QKeyEvent, QPainter, QPixmap
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

TEXTS: dict[str, dict[str, str]] = {
    "ar": {
        "title": "إعدادات VoiceTyper",
        "subtitle": "تخصيص تفضيلات الإملاء الصوتي بالضغط للتحدث",
        "engine_label": "محرك التفريغ الصوتي:",
        "engine_local": "محلي (faster-whisper) - 100% بدون إنترنت (GPU/CPU)",
        "engine_cloud": "سحابي (Google Gemini API)",
        "model_label": "نموذج Whisper المحلي:",
        "api_key_label_local": "مفتاح Gemini API (اختياري):",
        "api_key_label_cloud": "مفتاح Gemini API (مطلوب):",
        "api_key_placeholder_local": "مفتاح Gemini API (اختياري عند استخدام المحرك المحلي)",
        "api_key_placeholder_cloud": "الصق مفتاح Gemini API من Google AI Studio",
        "show": "إظهار",
        "hide": "إخفاء",
        "test": "تجربة الاتصال",
        "hotkey_label": "اختصار لوحة المفاتيح:",
        "record_hotkey": "تسجيل اختصار",
        "theme_label": "ثيم ومظهر الشريط:",
        "theme_classic": "الشكل الأصلي - Classic (الافتراضي)",
        "theme_neon": "نيون زجاجي - Neon Glass Pill",
        "theme_island": "الجزيرة التفاعلية - Matte Dynamic Island",
        "theme_mica": "بطاقة ميكا - Fluent Mica Card",
        "state_recording": "🔴 أثناء التسجيل",
        "state_processing": "🌊 أثناء المعالجة",
        "state_error": "⚠️ تنبيه أو خطأ",
        "state_standby": "⚪ وضع الاستعداد",
        "mic_label": "الميكروفون:",
        "mic_default": "ميكروفون النظام الافتراضي",
        "refresh": "تحديث",
        "lang_bias_label": "انحياز اللغة:",
        "lang_auto": "تلقائي (عربي، إنجليزي ومصطلحات مختلطة)",
        "lang_ar": "عربي (عامية مصرية وفصحى)",
        "lang_en": "الإنجليزية (English)",
        "duration_label": "أقصى مدة تسجيل:",
        "seconds": " ثانية",
        "casing_label": "تنسيق الحروف الإنجليزية:",
        "casing_word": "تكبير أول حرف في كل كلمة (Hello World)",
        "casing_sentence": "تكبير أول حرف في الجملة (Hello world)",
        "casing_none": "كما هي منطوقة دون تعديل",
        "vocab_label": "مصطلحات وقاموس مخصص:",
        "vocab_placeholder": "مصطلحات مفصولة بفواصل (مثل: PySide6, Docker, FastAPI)",
        "beeps": "تشغيل صوت تنبيه لطيف عند بدء وإيقاف التسجيل",
        "clipboard": "الاحتفاظ بالنص المفرغ في الحافظة وسجل ويندوز (Win + V)",
        "startup": "تشغيل VoiceTyper تلقائياً مع بدء تشغيل ويندوز",
        "typing": "الكتابة بمحاكاة الأحرف (Unicode) بدلاً من اللصق Ctrl+V",
        "toggle": "وضع التبديل (اضغط مرة لبدء التسجيل، ومرة أخرى للإيقاف)",
        "reset_pos": "استعادة موضع الشريط",
        "desktop_shortcut": "أيقونة سطح المكتب 🖥️",
        "cancel": "إلغاء",
        "save": "حفظ الإعدادات",
        "check_cache": "فحص الكاش",
        "download_model": "تحميل الموديل ⬇",
        "cancel_download": "إلغاء ⏹",
        "cached_disk": "الموديل مُحمّل وجاهز على القرص ✓",
        "not_cached": "غير مُحمّل على القرص ({size}). جاهز للتحميل.",
    },
    "en": {
        "title": "VoiceTyper Settings",
        "subtitle": "Configure your Push-to-Talk voice dictation preferences",
        "engine_label": "Transcription Engine:",
        "engine_local": "Local (faster-whisper) - 100% Offline GPU/CPU",
        "engine_cloud": "Google Gemini API - Cloud",
        "model_label": "Local Whisper Model:",
        "api_key_label_local": "Gemini API Key (Optional):",
        "api_key_label_cloud": "Gemini API Key (Required):",
        "api_key_placeholder_local": "Gemini API Key (Optional when using Local Engine)",
        "api_key_placeholder_cloud": "Paste Gemini API Key from Google AI Studio",
        "show": "Show",
        "hide": "Hide",
        "test": "Test",
        "hotkey_label": "PTT Hotkey:",
        "record_hotkey": "Record Hotkey",
        "theme_label": "Overlay Theme:",
        "theme_classic": "Classic (Default) - الشكل الأصلي",
        "theme_neon": "Neon Glass Pill",
        "theme_island": "Matte Dynamic Island",
        "theme_mica": "Fluent Mica Card",
        "state_recording": "🔴 Recording",
        "state_processing": "🌊 Processing",
        "state_error": "⚠️ Error",
        "state_standby": "⚪ Standby",
        "mic_label": "Microphone:",
        "mic_default": "Default System Microphone",
        "refresh": "Refresh",
        "lang_bias_label": "Language Bias:",
        "lang_auto": "Auto (Arabic, English & Mixed)",
        "lang_ar": "Arabic (Egyptian & Modern Standard)",
        "lang_en": "English",
        "duration_label": "Max Duration:",
        "seconds": " seconds",
        "casing_label": "English Casing:",
        "casing_word": "Capitalize Every Word (e.g. Hello World)",
        "casing_sentence": "Capitalize Sentences (e.g. Hello world)",
        "casing_none": "None (Keep as spoken)",
        "vocab_label": "Custom Vocabulary:",
        "vocab_placeholder": "Comma-separated terms (e.g. PySide6, Docker, FastAPI)",
        "beeps": "Play subtle chime sound on recording start and stop",
        "clipboard": "Keep transcript in Clipboard & Win+V History",
        "startup": "Start VoiceTyper automatically when Windows starts",
        "typing": "Type text via Unicode keystrokes instead of Ctrl+V paste",
        "toggle": "Toggle dictation mode (press once to start, press again to stop)",
        "reset_pos": "Reset Overlay Position",
        "desktop_shortcut": "Desktop Shortcut 🖥️",
        "cancel": "Cancel",
        "save": "Save Settings",
        "check_cache": "Check Cache",
        "download_model": "Download Model ⬇",
        "cancel_download": "Cancel ⏹",
        "cached_disk": "Model cached on disk [Ready to load]",
        "not_cached": "Not cached ({size}). Ready to download.",
    },
}


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

        # Presets Dropdown (shows selected preset or blank for custom combinations)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("", "")
        presets = [
            ("Ctrl + Shift", "ctrl+shift"),
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
        self.set_hotkey(current_hotkey)

    def get_hotkey(self) -> str:
        """Return the normalized hotkey string."""
        return self._current_hotkey

    def set_hotkey(self, hotkey_str: str) -> None:
        """Update hotkey display and stored value."""
        self._current_hotkey = hotkey_str.strip().lower()
        self.display_label.setText(format_hotkey_display(self._current_hotkey))

        matched_idx = 0
        for i in range(1, self.preset_combo.count()):
            if self.preset_combo.itemData(i) == self._current_hotkey:
                matched_idx = i
                break
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(matched_idx)
        self.preset_combo.blockSignals(False)
        self.hotkey_changed.emit(self._current_hotkey)

    def _on_preset_selected(self, index: int) -> None:
        val = str(self.preset_combo.itemData(index))
        if val:
            self.set_hotkey(val)

    def set_language(self, lang: str) -> None:
        self._lang = lang
        if self._is_recording:
            self.record_btn.setText("إلغاء (Esc)" if lang == "ar" else "Cancel (Esc)")
            self.display_label.setText("اضغط الأزرار... (Esc)" if lang == "ar" else "Press keys... (Esc)")
        else:
            self.record_btn.setText("تسجيل اختصار" if lang == "ar" else "Record Hotkey")

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
        self.record_btn.setText("تسجيل اختصار" if getattr(self, "_lang", "ar") == "ar" else "Record Hotkey")
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
                self.record_btn.setText("تسجيل اختصار" if getattr(self, "_lang", "ar") == "ar" else "Record Hotkey")
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


def create_dot_icon(is_cached: bool) -> QIcon:
    """Create a high-contrast glowing dot icon for model items."""
    pix = QPixmap(14, 14)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if is_cached:
        painter.setBrush(QColor("#FF4343"))
        painter.setPen(QColor("#FF8A80"))
    else:
        painter.setBrush(QColor("#5F6368"))
        painter.setPen(QColor("#3C4043"))
    painter.drawEllipse(1, 1, 11, 11)
    painter.end()
    return QIcon(pix)


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
        self._current_lang = getattr(self.config, "ui_language", "ar") or "ar"

        self.model_progress_signal.connect(self._on_model_progress_updated)
        if self.transcriber is not None and hasattr(self.transcriber, "add_download_listener"):
            self._dl_listener = lambda msg, pct=None, model="": self.model_progress_signal.emit(msg, pct)
            self.transcriber.add_download_listener(self._dl_listener)

        self.setWindowTitle("VoiceTyper Settings")
        self.setMinimumWidth(560)
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

        # Title / Header with Language Switcher opposite to subtitle
        header_layout = QHBoxLayout()
        header_text_vbox = QVBoxLayout()
        header_text_vbox.setSpacing(4)

        self.title_label = QLabel("VoiceTyper Settings")
        self.title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.subtitle_label = QLabel("Configure your Push-to-Talk voice dictation preferences")
        self.subtitle_label.setStyleSheet("color: #9AA0A6; font-size: 12px;")
        header_text_vbox.addWidget(self.title_label)
        header_text_vbox.addWidget(self.subtitle_label)
        header_layout.addLayout(header_text_vbox)

        header_layout.addStretch()

        # Language switcher toggle button placed opposite to subtitle in header
        self.lang_toggle_btn = QPushButton()
        self.lang_toggle_btn.setObjectName("langSwitchBtn")
        self.lang_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lang_toggle_btn.setFixedHeight(28)
        self.lang_toggle_btn.setStyleSheet(
            "QPushButton#langSwitchBtn {"
            "    background-color: #2D2F31;"
            "    color: #E8EAED;"
            "    border: 1px solid #3C4043;"
            "    border-radius: 6px;"
            "    padding: 3px 12px;"
            "    font-size: 12px;"
            "    font-weight: 600;"
            "}"
            "QPushButton#langSwitchBtn:hover {"
            "    background-color: #3C4043;"
            "    border-color: #5F6368;"
            "    color: #FFFFFF;"
            "}"
            "QPushButton#langSwitchBtn:pressed {"
            "    background-color: #1A73E8;"
            "    border-color: #4285F4;"
            "    color: #FFFFFF;"
            "}"
        )
        self.lang_toggle_btn.clicked.connect(self._toggle_language)
        header_layout.addWidget(self.lang_toggle_btn, alignment=Qt.AlignmentFlag.AlignVCenter)

        main_layout.addLayout(header_layout)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(sep)

        form_layout = QFormLayout()
        form_layout.setSpacing(14)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        # 1. Transcription Engine
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("محلي (faster-whisper) - 100% بدون إنترنت (GPU/CPU)", "local")
        self.engine_combo.addItem("سحابي (Google Gemini API)", "gemini")
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.engine_row_label = QLabel("محرك التفريغ الصوتي:")
        form_layout.addRow(self.engine_row_label, self.engine_combo)

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

        # Model cache indicator dot button (vibrant glowing red when cached, gray when not)
        self.model_cache_dot = QPushButton()
        self.model_cache_dot.setObjectName("modelCacheDot")
        self.model_cache_dot.setFixedSize(18, 18)
        self.model_cache_dot.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_cache_dot.clicked.connect(self._on_model_cache_dot_clicked)
        model_row.addWidget(self.model_cache_dot)

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

        self.download_now_btn = QPushButton("تحميل الموديل ⬇")
        self.download_now_btn.setFixedHeight(22)
        self.download_now_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 8px; background-color: #1A73E8; color: #FFFFFF; border-radius: 3px; border: none; font-weight: bold; }"
            "QPushButton:hover { background-color: #4285F4; }"
        )
        self.download_now_btn.clicked.connect(self._start_manual_download)
        self.download_now_btn.hide()
        status_row.addWidget(self.download_now_btn)

        self.cancel_download_btn = QPushButton("إلغاء ⏹")
        self.cancel_download_btn.setFixedHeight(22)
        self.cancel_download_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 8px; background-color: #C5221F; color: #FFFFFF; border-radius: 3px; border: none; font-weight: bold; }"
            "QPushButton:hover { background-color: #D93025; }"
            "QPushButton:disabled { background-color: #5F6368; color: #9AA0A6; }"
        )
        self.cancel_download_btn.clicked.connect(self._cancel_active_download)
        self.cancel_download_btn.hide()
        status_row.addWidget(self.cancel_download_btn)

        self.refresh_cache_btn = QPushButton("فحص الكاش")
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

        self.model_row_label = QLabel("نموذج Whisper المحلي:")
        form_layout.addRow(self.model_row_label, self.whisper_container)

        # 3. Gemini API Key (Required for Gemini, Optional for Local)
        self.key_container = QWidget()
        key_layout = QHBoxLayout(self.key_container)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(8)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("مفتاح Gemini API (اختياري عند استخدام المحرك المحلي)")

        self.toggle_key_btn = QPushButton("إظهار")
        self.toggle_key_btn.setFixedWidth(60)
        self.toggle_key_btn.clicked.connect(self._toggle_api_key_visibility)

        self.test_key_btn = QPushButton("تجربة الاتصال")
        self.test_key_btn.setFixedWidth(100)
        self.test_key_btn.clicked.connect(self._test_api_key)

        key_layout.addWidget(self.api_key_input)
        key_layout.addWidget(self.toggle_key_btn)
        key_layout.addWidget(self.test_key_btn)

        self.api_key_row_label = QLabel("مفتاح Gemini API:")
        form_layout.addRow(self.api_key_row_label, self.key_container)

        # 4. Global Hotkey (Interactive Keyboard Recorder)
        self.hotkey_recorder = HotkeyRecorderWidget(
            current_hotkey=self.config.hotkey,
            hotkey_manager=self.hotkey_manager,
        )
        self.hotkey_row_label = QLabel("اختصار لوحة المفاتيح:")
        form_layout.addRow(self.hotkey_row_label, self.hotkey_recorder)

        # 5. Overlay Theme (Interactive Live Preview)
        theme_container = QWidget()
        theme_vbox = QVBoxLayout(theme_container)
        theme_vbox.setContentsMargins(0, 0, 0, 0)
        theme_vbox.setSpacing(8)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("الشكل الأصلي - Classic (الافتراضي)", "classic")
        self.theme_combo.addItem("نيون زجاجي - Neon Glass Pill", "neon_glass")
        self.theme_combo.addItem("الجزيرة التفاعلية - Matte Dynamic Island", "matte_island")
        self.theme_combo.addItem("بطاقة ميكا - Fluent Mica Card", "fluent_mica")
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
            (OverlayState.RECORDING, "🔴 أثناء التسجيل"),
            (OverlayState.PROCESSING, "🌊 أثناء المعالجة"),
            (OverlayState.ERROR, "⚠️ تنبيه أو خطأ"),
            (OverlayState.STANDBY, "⚪ وضع الاستعداد"),
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

        self.theme_row_label = QLabel("ثيم ومظهر الشريط:")
        form_layout.addRow(self.theme_row_label, theme_container)

        # 6. Microphone Selector
        mic_container = QWidget()
        mic_layout = QHBoxLayout(mic_container)
        mic_layout.setContentsMargins(0, 0, 0, 0)
        mic_layout.setSpacing(8)

        self.mic_combo = QComboBox()
        self.refresh_mic_btn = QPushButton("تحديث")
        self.refresh_mic_btn.setFixedWidth(70)
        self.refresh_mic_btn.clicked.connect(self._refresh_microphones)

        mic_layout.addWidget(self.mic_combo)
        mic_layout.addWidget(self.refresh_mic_btn)

        self.mic_row_label = QLabel("الميكروفون:")
        form_layout.addRow(self.mic_row_label, mic_container)

        # 7. Language Hint
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("تلقائي (عربي، إنجليزي ومصطلحات مختلطة)", "Auto")
        self.lang_combo.addItem("عربي (عامية مصرية وفصحى)", "Arabic")
        self.lang_combo.addItem("الإنجليزية (English)", "English")
        self.lang_bias_row_label = QLabel("انحياز اللغة:")
        form_layout.addRow(self.lang_bias_row_label, self.lang_combo)

        # 8. Max Recording Duration
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 300)
        self.duration_spin.setSuffix(" ثانية")
        self.duration_row_label = QLabel("أقصى مدة تسجيل:")
        form_layout.addRow(self.duration_row_label, self.duration_spin)

        # 9. English Capitalization
        self.capitalization_combo = QComboBox()
        self.capitalization_combo.addItem("تكبير أول حرف في كل كلمة (Hello World)", "word")
        self.capitalization_combo.addItem("تكبير أول حرف في الجملة (Hello world)", "sentence")
        self.capitalization_combo.addItem("كما هي منطوقة دون تعديل", "none")
        self.casing_row_label = QLabel("تنسيق الحروف الإنجليزية:")
        form_layout.addRow(self.casing_row_label, self.capitalization_combo)

        # 10. Custom Vocabulary
        self.vocab_input = QLineEdit()
        self.vocab_input.setPlaceholderText("مصطلحات مفصولة بفواصل (مثل: PySide6, Docker, FastAPI)")
        self.vocab_row_label = QLabel("قاموس مخصص:")
        form_layout.addRow(self.vocab_row_label, self.vocab_input)

        main_layout.addLayout(form_layout)

        # Checkboxes
        check_layout = QVBoxLayout()
        check_layout.setSpacing(10)

        self.beeps_checkbox = QCheckBox("تشغيل صوت تنبيه لطيف عند بدء وإيقاف التسجيل")
        self.clipboard_keep_checkbox = QCheckBox("الاحتفاظ بالنص المفرغ في الحافظة وسجل ويندوز (Win + V)")
        self.startup_checkbox = QCheckBox("تشغيل VoiceTyper تلقائياً مع بدء تشغيل ويندوز")
        self.typing_checkbox = QCheckBox("الكتابة بمحاكاة الأحرف (Unicode) بدلاً من اللصق Ctrl+V")
        self.toggle_checkbox = QCheckBox("وضع التبديل (اضغط مرة لبدء التسجيل، ومرة أخرى للإيقاف)")

        check_layout.addWidget(self.beeps_checkbox)
        check_layout.addWidget(self.clipboard_keep_checkbox)
        check_layout.addWidget(self.startup_checkbox)
        check_layout.addWidget(self.typing_checkbox)
        check_layout.addWidget(self.toggle_checkbox)
        main_layout.addLayout(check_layout)

        # Action Buttons (Reset Position / Cancel / Save)
        btn_layout = QHBoxLayout()

        self.reset_pos_btn = QPushButton("استعادة موضع الشريط")
        self.reset_pos_btn.clicked.connect(self._reset_overlay_position)
        btn_layout.addWidget(self.reset_pos_btn)

        self.desktop_shortcut_btn = QPushButton("أيقونة سطح المكتب 🖥️")
        self.desktop_shortcut_btn.setToolTip("إنشاء أو تحديث اختصار VoiceTyper على سطح المكتب")
        self.desktop_shortcut_btn.clicked.connect(self._create_desktop_shortcut)
        btn_layout.addWidget(self.desktop_shortcut_btn)

        btn_layout.addStretch()

        self.cancel_btn = QPushButton("إلغاء")
        self.cancel_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton("حفظ الإعدادات")
        self.save_btn.setObjectName("primaryButton")
        self.save_btn.clicked.connect(self._save_and_close)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)
        main_layout.addLayout(btn_layout)

    def _toggle_language(self) -> None:
        """Toggle between Arabic and English on single click."""
        new_lang = "en" if self._current_lang == "ar" else "ar"
        self._apply_language(new_lang)

    def _apply_language(self, lang: str) -> None:
        """Dynamically update all UI text and layout direction for Arabic / English."""
        self._current_lang = lang
        t = TEXTS.get(lang, TEXTS["ar"])

        # Layout direction
        if lang == "ar":
            self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            self.lang_toggle_btn.setText("English 🌐")
            self.lang_toggle_btn.setToolTip("Switch UI to English")
        else:
            self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
            self.lang_toggle_btn.setText("العربية 🌐")
            self.lang_toggle_btn.setToolTip("تغيير الواجهة إلى اللغة العربية")

        # Window & Header
        self.setWindowTitle(t["title"])
        self.title_label.setText(t["title"])
        self.subtitle_label.setText(t["subtitle"])

        # Form Labels
        self.engine_row_label.setText(t["engine_label"])
        self.model_row_label.setText(t["model_label"])
        curr_engine = str(self.engine_combo.currentData())
        if curr_engine == "local":
            self.api_key_row_label.setText(t["api_key_label_local"])
            self.api_key_input.setPlaceholderText(t["api_key_placeholder_local"])
        else:
            self.api_key_row_label.setText(t["api_key_label_cloud"])
            self.api_key_input.setPlaceholderText(t["api_key_placeholder_cloud"])

        self.hotkey_row_label.setText(t["hotkey_label"])
        self.theme_row_label.setText(t["theme_label"])
        self.mic_row_label.setText(t["mic_label"])
        self.lang_bias_row_label.setText(t["lang_bias_label"])
        self.duration_row_label.setText(t["duration_label"])
        self.casing_row_label.setText(t["casing_label"])
        self.vocab_row_label.setText(t["vocab_label"])
        self.vocab_input.setPlaceholderText(t["vocab_placeholder"])

        # Combobox items
        self.engine_combo.setItemText(0, t["engine_local"])
        self.engine_combo.setItemText(1, t["engine_cloud"])

        self.theme_combo.setItemText(0, t["theme_classic"])
        self.theme_combo.setItemText(1, t["theme_neon"])
        self.theme_combo.setItemText(2, t["theme_island"])
        self.theme_combo.setItemText(3, t["theme_mica"])

        self.lang_combo.setItemText(0, t["lang_auto"])
        self.lang_combo.setItemText(1, t["lang_ar"])
        self.lang_combo.setItemText(2, t["lang_en"])

        self.capitalization_combo.setItemText(0, t["casing_word"])
        self.capitalization_combo.setItemText(1, t["casing_sentence"])
        self.capitalization_combo.setItemText(2, t["casing_none"])

        self.duration_spin.setSuffix(t["seconds"])

        # Buttons
        if self.api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.toggle_key_btn.setText(t["show"])
        else:
            self.toggle_key_btn.setText(t["hide"])
        self.test_key_btn.setText(t["test"])
        self.download_now_btn.setText(t["download_model"])
        self.cancel_download_btn.setText(t["cancel_download"])
        self.refresh_cache_btn.setText(t["check_cache"])
        self.refresh_mic_btn.setText(t["refresh"])
        self.reset_pos_btn.setText(t["reset_pos"])
        self.desktop_shortcut_btn.setText(t["desktop_shortcut"])
        self.cancel_btn.setText(t["cancel"])
        self.save_btn.setText(t["save"])

        # Checkboxes
        self.beeps_checkbox.setText(t["beeps"])
        self.clipboard_keep_checkbox.setText(t["clipboard"])
        self.startup_checkbox.setText(t["startup"])
        self.typing_checkbox.setText(t["typing"])
        self.toggle_checkbox.setText(t["toggle"])

        # Preview buttons
        if hasattr(self, "state_btns"):
            self.state_btns[OverlayState.RECORDING].setText(t["state_recording"])
            self.state_btns[OverlayState.PROCESSING].setText(t["state_processing"])
            self.state_btns[OverlayState.ERROR].setText(t["state_error"])
            self.state_btns[OverlayState.STANDBY].setText(t["state_standby"])

        # Hotkey recorder
        if hasattr(self, "hotkey_recorder"):
            self.hotkey_recorder.set_language(lang)

        # Refresh model items & cache dot
        self._refresh_model_cache_labels()

    def _on_model_cache_dot_clicked(self) -> None:
        """Clicking the cache dot starts download if un-cached or checks cache if cached."""
        sel_model = str(self.whisper_model_combo.currentData())
        if not is_model_cached(sel_model):
            self._start_manual_download()
        else:
            self._refresh_model_cache_labels()

    def _update_model_cache_dot(self, is_cached: bool) -> None:
        """Update cache dot indicator style (vibrant glowing red like recording dot when cached, gray when not)."""
        lang = getattr(self, "_current_lang", "ar")
        if is_cached:
            self.model_cache_dot.setStyleSheet(
                "QPushButton#modelCacheDot {"
                "    min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px;"
                "    border-radius: 9px;"
                "    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, "
                "        stop:0 #FF5252, stop:0.5 #EF4444, stop:1.0 #B91C1C);"
                "    border: 1.5px solid #FF8A80;"
                "}"
                "QPushButton#modelCacheDot:hover {"
                "    border-color: #FFFFFF;"
                "}"
            )
            self.model_cache_dot.setToolTip(
                "الموديل مُحمّل على الجهاز وجاهز للاستخدام الفوري ✓"
                if lang == "ar"
                else "Model is downloaded and ready for use ✓"
            )
        else:
            self.model_cache_dot.setStyleSheet(
                "QPushButton#modelCacheDot {"
                "    min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px;"
                "    border-radius: 9px;"
                "    background-color: #4B5563;"
                "    border: 1.5px solid #374151;"
                "}"
                "QPushButton#modelCacheDot:hover {"
                "    border-color: #9CA3AF;"
                "    background-color: #6B7280;"
                "}"
            )
            self.model_cache_dot.setToolTip(
                "الموديل غير مُحمّل - اضغط هنا لتنزيله"
                if lang == "ar"
                else "Model is not downloaded - click to download"
            )

    def _on_theme_changed(self, index: int) -> None:
        """Apply theme immediately for live preview on both widget and desktop overlay."""
        theme = str(self.theme_combo.itemData(index))
        self.config.overlay_theme = theme
        if hasattr(self, "theme_preview"):
            self.theme_preview.set_theme(theme)
        if self.overlay is not None:
            self.overlay.set_theme(theme)
            if not getattr(self.transcriber, "is_downloading", False):
                self.overlay.preview_state(self._current_preview_state)

    def _on_preview_state_selected(self, state: OverlayState) -> None:
        """Switch the preview state on both the embedded widget and desktop overlay."""
        self._current_preview_state = state
        if hasattr(self, "theme_preview"):
            self.theme_preview.set_preview_state(state)
        if self.overlay is not None:
            if not getattr(self.transcriber, "is_downloading", False):
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
        if hasattr(self, "_dl_listener") and self.transcriber is not None:
            try:
                self.transcriber.remove_download_listener(self._dl_listener)
            except Exception:
                pass
        if self.overlay is not None:
            self.overlay.set_theme(self._initial_theme)
            if not getattr(self.transcriber, "is_downloading", False):
                self.overlay.show_standby()
        super().reject()

    def closeEvent(self, event: Any) -> None:
        """Ensure overlay returns to standby when settings dialog is closed."""
        if hasattr(self, "_dl_listener") and self.transcriber is not None:
            try:
                self.transcriber.remove_download_listener(self._dl_listener)
            except Exception:
                pass
        if self.overlay is not None:
            self.overlay.set_theme(self._initial_theme)
            if not getattr(self.transcriber, "is_downloading", False):
                self.overlay.show_standby()
        super().closeEvent(event)

    def _on_engine_changed(self, index: int) -> None:
        """Update UI fields based on active transcription engine."""
        engine = self.engine_combo.itemData(index)
        t = TEXTS.get(getattr(self, "_current_lang", "ar"), TEXTS["ar"])
        if engine == "local":
            self.whisper_container.setEnabled(True)
            self.api_key_row_label.setText(t["api_key_label_local"])
            self.api_key_input.setPlaceholderText(t["api_key_placeholder_local"])
        else:
            self.whisper_container.setEnabled(False)
            self.api_key_row_label.setText(t["api_key_label_cloud"])
            self.api_key_input.setPlaceholderText(t["api_key_placeholder_cloud"])

    def _refresh_model_cache_labels(self) -> None:
        """Populate or update model items showing red/gray dot indicators and sizes without text tags."""
        current_data = (
            self.whisper_model_combo.currentData()
            if self.whisper_model_combo.count() > 0
            else self.config.whisper_model
        )
        self.whisper_model_combo.blockSignals(True)
        self.whisper_model_combo.clear()

        for model_id, title, size in MODELS_CATALOG:
            cached = is_model_cached(model_id)
            dot_icon = create_dot_icon(cached)
            display_text = title if cached else f"{title} ({size})"
            self.whisper_model_combo.addItem(dot_icon, display_text, model_id)

        idx = self.whisper_model_combo.findData(current_data)
        if idx >= 0:
            self.whisper_model_combo.setCurrentIndex(idx)
        else:
            self.whisper_model_combo.setCurrentIndex(0)
        self.whisper_model_combo.blockSignals(False)

        sel_model = str(self.whisper_model_combo.currentData())
        is_cached = is_model_cached(sel_model)
        self._update_model_cache_dot(is_cached)

        t = TEXTS.get(getattr(self, "_current_lang", "ar"), TEXTS["ar"])
        if is_cached:
            self.model_status_label.setText(t["cached_disk"])
            self.model_progress_bar.hide()
            self.download_now_btn.hide()
            self.cancel_download_btn.hide()
        else:
            sz = "~1.5 GB"
            for m, _, s in MODELS_CATALOG:
                if m == sel_model:
                    sz = s
                    break
            self.model_status_label.setText(t["not_cached"].format(size=sz))
            self.download_now_btn.show()
            self.download_now_btn.setEnabled(True)
            self.model_progress_bar.hide()
            self.cancel_download_btn.hide()

    def _on_whisper_model_changed(self, index: int) -> None:
        """Handle user selecting a different Whisper model with status label and cache dot update."""
        model_id = str(self.whisper_model_combo.itemData(index))
        if not model_id:
            return

        is_cached = is_model_cached(model_id)
        self._update_model_cache_dot(is_cached)

        # Check if a model is currently downloading
        is_dl = getattr(self.transcriber, "is_downloading", False)
        dl_text, dl_pct, dl_model = getattr(self.transcriber, "download_info", ("", None, ""))
        if is_dl and dl_model == model_id:
            self.model_status_label.setText(dl_text or f"Downloading '{dl_model}'...")
            self.download_now_btn.hide()
            self.cancel_download_btn.show()
            self.cancel_download_btn.setEnabled(True)
            self.model_progress_bar.show()
            if dl_pct is not None:
                self.model_progress_bar.setValue(int(dl_pct))
            return

        t = TEXTS.get(getattr(self, "_current_lang", "ar"), TEXTS["ar"])
        if is_cached:
            self.model_status_label.setText(t["cached_disk"])
            self.model_progress_bar.hide()
            self.download_now_btn.hide()
            self.cancel_download_btn.hide()
        else:
            sz = "~1.5 GB"
            for m, _, s in MODELS_CATALOG:
                if m == model_id:
                    sz = s
                    break
            self.model_status_label.setText(t["not_cached"].format(size=sz))
            self.download_now_btn.show()
            self.download_now_btn.setEnabled(True)
            self.model_progress_bar.hide()
            self.cancel_download_btn.hide()

    def _start_manual_download(self) -> None:
        """Trigger background download of the selected model with progress and cancellation."""
        sel_model = str(self.whisper_model_combo.currentData())
        t = TEXTS.get(getattr(self, "_current_lang", "ar"), TEXTS["ar"])
        if is_model_cached(sel_model):
            self.model_status_label.setText(t["cached_disk"])
            self.download_now_btn.hide()
            self._update_model_cache_dot(True)
            return

        if self.transcriber is not None:
            self.transcriber.set_whisper_model(sel_model)
            self.transcriber.reset_cancel_download()

        connecting_msg = (
            f"جاري الاتصال بـ Hugging Face لتحميل '{sel_model}'..."
            if getattr(self, "_current_lang", "ar") == "ar"
            else f"Connecting to Hugging Face to download '{sel_model}'..."
        )

        if self.overlay is not None:
            self.overlay.show_download(connecting_msg, 0.0, sel_model)

        self.download_now_btn.hide()
        self.cancel_download_btn.setEnabled(True)
        self.cancel_download_btn.setText(t["cancel_download"])
        self.cancel_download_btn.show()
        self.model_progress_bar.show()
        self.model_progress_bar.setRange(0, 0)
        self.model_status_label.setText(connecting_msg)

        import threading

        def bg_download() -> None:
            def progress_cb(text: str, pct: float | None = None) -> None:
                self.model_progress_signal.emit(text, pct)

            if self.transcriber is not None:
                self.transcriber.preload_model(on_progress=progress_cb, force_reload=True)

        threading.Thread(target=bg_download, name="ManualModelDownloader", daemon=True).start()

    def _cancel_active_download(self) -> None:
        """Cancel ongoing model download immediately."""
        if self.transcriber is not None:
            self.transcriber.cancel_download()
        if self.overlay is not None:
            self.overlay.cancel_active_download()
        is_ar = getattr(self, "_current_lang", "ar") == "ar"
        self.cancel_download_btn.setEnabled(False)
        self.cancel_download_btn.setText("جاري الإلغاء..." if is_ar else "Cancelling...")
        self.model_status_label.setText("جاري إلغاء التحميل..." if is_ar else "Cancelling download...")

    @pyqtSlot(str, object)
    def _on_model_progress_updated(self, status_text: str, progress_pct: float | None) -> None:
        """Update status label and progress bar without blocking GUI."""
        self.model_status_label.setText(status_text)
        is_cancelled = "cancelled" in status_text.lower() or "إلغاء" in status_text
        is_ready = "ready" in status_text.lower() or "✓" in status_text

        if self.overlay is not None and getattr(self.transcriber, "is_downloading", False):
            pct_val = float(progress_pct) if progress_pct is not None else 0.0
            self.overlay.update_download_progress(status_text, pct_val)

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

        if progress_pct is not None and progress_pct > 0:
            self.model_progress_bar.show()
            self.cancel_download_btn.show()
            self.download_now_btn.hide()
            self.model_progress_bar.setRange(0, 100)
            self.model_progress_bar.setValue(int(progress_pct))
            if progress_pct >= 100.0:
                self.model_progress_bar.hide()
                self.cancel_download_btn.hide()
                self._refresh_model_cache_labels()
        elif progress_pct is not None and progress_pct == 0.0:
            self.model_progress_bar.show()
            self.cancel_download_btn.show()
            self.download_now_btn.hide()
            self.model_progress_bar.setRange(0, 0)
        else:
            if "loading" in status_text.lower() or "connecting" in status_text.lower():
                self.model_progress_bar.show()
                self.model_progress_bar.setRange(0, 0)
                self.cancel_download_btn.hide()

    def _refresh_microphones(self) -> None:
        """Scan and populate available input devices with clean, deduplicated names."""
        t = TEXTS.get(getattr(self, "_current_lang", "ar"), TEXTS["ar"])
        self.mic_combo.clear()
        self.mic_combo.addItem(t["mic_default"], None)

        devices = AudioRecorder.get_input_devices()
        selected_index = 0

        for dev in devices:
            label = dev["name"]
            self.mic_combo.addItem(label, dev["index"])
            if self.config.mic_device == dev["index"] or (
                isinstance(self.config.mic_device, str) and self.config.mic_device == dev["name"]
            ):
                selected_index = self.mic_combo.count() - 1

        self.mic_combo.setCurrentIndex(selected_index)

    def _load_values(self) -> None:
        """Populate widgets with current Config values."""
        # UI Language (Default Arabic "ar")
        ui_lang = getattr(self.config, "ui_language", "ar") or "ar"
        self._apply_language(ui_lang)

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

        is_dl = getattr(self.transcriber, "is_downloading", False)
        if self.overlay is not None and not is_dl:
            self.overlay.preview_state(self._current_preview_state)

        if is_dl:
            dl_text, dl_pct, dl_model = getattr(self.transcriber, "download_info", ("", None, ""))
            self.model_status_label.setText(dl_text or f"Downloading '{dl_model}'...")
            self.download_now_btn.hide()
            self.cancel_download_btn.show()
            self.cancel_download_btn.setEnabled(True)
            self.model_progress_bar.show()
            if dl_pct is not None:
                self.model_progress_bar.setValue(int(dl_pct))

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
        t = TEXTS.get(getattr(self, "_current_lang", "ar"), TEXTS["ar"])
        if self.api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.toggle_key_btn.setText(t["hide"])
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.toggle_key_btn.setText(t["show"])

    def _test_api_key(self) -> None:
        """Quickly test API key connectivity with Gemini."""
        key = self.api_key_input.text().strip()
        is_ar = getattr(self, "_current_lang", "ar") == "ar"
        if not key:
            QMessageBox.warning(
                self,
                "مفتاح API مفقود" if is_ar else "API Key Missing",
                "يرجى إدخال مفتاح Gemini API أولاً." if is_ar else "Please enter an API key first.",
            )
            return

        self.test_key_btn.setEnabled(False)
        self.test_key_btn.setText("جاري الفحص..." if is_ar else "Testing...")

        try:
            transcriber = Transcriber(api_key=key, engine="gemini")
            client = transcriber._get_client()
            _ = client.models.get(model=transcriber.model_name)
            QMessageBox.information(
                self,
                "تم الاتصال بنجاح" if is_ar else "API Key Valid",
                "مفتاح Gemini API صالح وتم الاتصال بالسيرفر بنجاح!"
                if is_ar
                else "Gemini API key is valid and connected successfully!",
            )
        except Exception as err:
            QMessageBox.critical(
                self,
                "فشل الاتصال" if is_ar else "API Key Failed",
                f"فشل التحقق من المفتاح:\n{err}" if is_ar else f"Validation failed:\n{err}",
            )
        finally:
            self.test_key_btn.setEnabled(True)
            self.test_key_btn.setText("تجربة الاتصال" if is_ar else "Test")

    def _reset_overlay_position(self) -> None:
        """Reset pill position to bottom-right above taskbar clock."""
        self.config_manager.reset_overlay_position()
        self.config.overlay_pos = None
        is_ar = getattr(self, "_current_lang", "ar") == "ar"
        QMessageBox.information(
            self,
            "استعادة الموضع" if is_ar else "Position Reset",
            "تمت استعادة موضع الشريط إلى الوضع الافتراضي (أسفل يمين الشاشة فوق الساعة)."
            if is_ar
            else "Overlay position reset to default (bottom-right above the taskbar clock).",
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
                "اختصار سطح المكتب" if getattr(self, "_current_lang", "ar") == "ar" else "Desktop Shortcut",
                "تم إنشاء اختصار VoiceTyper على سطح المكتب بنجاح!\n\nDesktop shortcut created successfully.",
            )
        except Exception as err:
            QMessageBox.warning(
                self,
                "خطأ في الاختصار" if getattr(self, "_current_lang", "ar") == "ar" else "Desktop Shortcut Error",
                f"فشل إنشاء اختصار سطح المكتب:\n{err}",
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

        # Update config fields
        self.config.ui_language = self._current_lang
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
            "Saved settings successfully (lang=%s, engine=%s, model=%s, hotkey=%s)",
            self.config.ui_language,
            self.config.transcription_engine,
            self.config.whisper_model,
            self.config.hotkey,
        )
        self.accept()
