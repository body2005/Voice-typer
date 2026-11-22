"""Unit tests for UI overlay, drag positioning, smart opacity, and settings dialog."""

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent

from voicetyper.config import Config, ConfigManager, MockStartupManager
from voicetyper.ui.overlay import OverlayPill, OverlayState
from voicetyper.ui.settings_dialog import SettingsDialog


def test_overlay_pill_window_flags_and_states() -> None:
    """Verify OverlayPill flags: non-focus, NOT transparent for input (so it is draggable)."""
    cfg = Config()
    overlay = OverlayPill(audio_recorder=None, config=cfg)

    # Window flags verification
    flags = overlay.windowFlags()
    assert bool(flags & Qt.WindowType.Tool)
    assert bool(flags & Qt.WindowType.FramelessWindowHint)
    assert bool(flags & Qt.WindowType.WindowStaysOnTopHint)
    assert bool(flags & Qt.WindowType.WindowDoesNotAcceptFocus)
    # WindowTransparentForInput MUST NOT be set so mouse click-and-drag works
    assert not bool(flags & Qt.WindowType.WindowTransparentForInput)

    # State 1: Recording
    overlay.show_recording()
    assert overlay._state == OverlayState.RECORDING
    assert overlay._anim_timer.isActive()
    assert overlay.windowOpacity() > 0.9

    # Amplitude update
    overlay.set_amplitude(0.35)
    assert overlay._raw_amplitude == 0.35

    # State 2: Processing
    overlay.show_processing()
    assert overlay._state == OverlayState.PROCESSING
    assert overlay._anim_timer.isActive()

    # State 3: Error
    overlay.show_error("Network Timeout")
    assert overlay._state == OverlayState.ERROR
    assert overlay._error_text == "Network Timeout"
    assert overlay._error_timer.isActive()

    # State 4: Standby (ghost state, 60 FPS timer stopped to save CPU)
    overlay.show_standby()
    assert overlay._state == OverlayState.STANDBY
    assert not overlay._anim_timer.isActive()
    assert overlay._inactivity_timer.isActive()

    # State 5: Hidden
    overlay.hide_overlay()
    assert overlay._state == OverlayState.HIDDEN
    assert not overlay._anim_timer.isActive()


def test_overlay_pill_drag_and_save_position() -> None:
    """Verify dragging the pill updates its coordinates and saves them to config."""
    cfg = Config()
    cfg_mgr = ConfigManager()
    overlay = OverlayPill(audio_recorder=None, config=cfg, config_manager=cfg_mgr)
    overlay.show()

    initial_pos = overlay.pos()
    emitted_coords: list[tuple[int, int]] = []
    overlay.position_changed.connect(lambda x, y: emitted_coords.append((x, y)))

    # Simulate mouse press at global (200, 200), local (20, 20)
    press_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(20.0, 20.0),
        QPointF(200.0, 200.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mousePressEvent(press_event)
    assert overlay._is_dragging is True

    # Simulate mouse move to global (350, 450)
    move_event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(20.0, 20.0),
        QPointF(350.0, 450.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseMoveEvent(move_event)
    assert overlay.pos() != initial_pos

    # Simulate mouse release
    release_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(20.0, 20.0),
        QPointF(350.0, 450.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseReleaseEvent(release_event)
    assert overlay._is_dragging is False
    assert len(emitted_coords) == 1
    assert cfg.overlay_pos == [overlay.x(), overlay.y()]

    # Test reset position
    overlay.reset_position()
    assert cfg.overlay_pos is None
    overlay.hide()


def test_overlay_pill_multimonitor_fallback() -> None:
    """Verify that coordinates off all screens fall back to the primary screen default."""
    # Bogus coordinates off-screen
    cfg = Config(overlay_pos=[-9999, -9999])
    overlay = OverlayPill(audio_recorder=None, config=cfg)
    overlay.reposition(force_default=False)

    # Position should be valid on current screen, NOT at (-9999, -9999)
    assert overlay.x() > -1000
    assert overlay.y() > -1000


def test_settings_dialog_load_and_update() -> None:
    """Verify SettingsDialog loads config and writes values back."""
    cfg = Config(
        hotkey="Ctrl+Space",
        start_stop_beeps=True,
        language_hint="Arabic",
        max_duration_sec=90.0,
        custom_vocabulary="PyQt6, Python",
        api_key="SecretTestKey",
    )
    startup_mgr = MockStartupManager(initial_enabled=False)
    cfg_mgr = ConfigManager(startup_manager=startup_mgr)

    dialog = SettingsDialog(config=cfg, config_manager=cfg_mgr)
    assert dialog.api_key_input.text() == "SecretTestKey"
    assert dialog.duration_spin.value() == 90
    assert dialog.beeps_checkbox.isChecked() is True
    assert dialog.vocab_input.text() == "PyQt6, Python"

    # Mutate values in dialog
    dialog.api_key_input.setText("UpdatedKey123")
    dialog.duration_spin.setValue(150)
    dialog.vocab_input.setText("Docker, Linux")
    dialog.startup_checkbox.setChecked(True)

    dialog._save_and_close()

    assert cfg.api_key == "UpdatedKey123"
    assert cfg.max_duration_sec == 150.0
    assert cfg.custom_vocabulary == "Docker, Linux"
    assert cfg.start_with_windows is True
    assert startup_mgr.is_enabled() is True


def test_overlay_gear_button_click() -> None:
    """Verify clicking the gear icon emits settings_clicked without starting a drag."""
    cfg = Config()
    overlay = OverlayPill(audio_recorder=None, config=cfg)
    overlay.show()

    clicked_signals: list[bool] = []
    overlay.settings_clicked.connect(lambda: clicked_signals.append(True))

    # Click on the gear icon (center at 345, 28)
    press_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(345.0, 28.0),
        QPointF(345.0, 28.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mousePressEvent(press_event)
    assert overlay._is_dragging is False
    assert overlay._gear_pressed is True

    # Release on the gear icon
    release_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(345.0, 28.0),
        QPointF(345.0, 28.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseReleaseEvent(release_event)
    assert overlay._gear_pressed is False
    assert len(clicked_signals) == 1
    overlay.hide()


def test_settings_dialog_engine_and_hotkey_recorder() -> None:
    """Verify SettingsDialog handles local faster-whisper engine and hotkey recorder."""
    cfg = Config(
        hotkey="ctrl+shift",
        transcription_engine="local",
        whisper_model="large-v3-turbo",
    )
    startup_mgr = MockStartupManager(initial_enabled=False)
    cfg_mgr = ConfigManager(startup_manager=startup_mgr)

    dialog = SettingsDialog(config=cfg, config_manager=cfg_mgr)
    assert dialog.engine_combo.currentData() == "local"
    assert dialog.whisper_model_combo.currentData() == "large-v3-turbo"
    assert dialog.hotkey_recorder.get_hotkey() == "ctrl+shift"

    # Mutate values
    dialog.hotkey_recorder.set_hotkey("f8")
    idx_med = dialog.whisper_model_combo.findData("medium")
    dialog.whisper_model_combo.setCurrentIndex(idx_med)

    dialog._save_and_close()

    assert cfg.hotkey == "f8"
    assert cfg.whisper_model == "medium"
    assert cfg.transcription_engine == "local"


def test_settings_dialog_microphones_clean_names() -> None:
    """Verify microphone dropdown has 'Default System Microphone' first and no '(Device X)' suffixes."""
    from unittest.mock import patch

    mock_devs = [
        {"index": 17, "name": "USB Audio Microphone", "channels": 1, "default_samplerate": 44100.0, "hostapi": 2},
        {"index": 18, "name": "Webcam Microphone", "channels": 1, "default_samplerate": 44100.0, "hostapi": 2},
    ]

    cfg = Config()
    cfg_mgr = ConfigManager(startup_manager=MockStartupManager())

    with patch("voicetyper.audio_recorder.AudioRecorder.get_input_devices", return_value=mock_devs):
        dialog = SettingsDialog(config=cfg, config_manager=cfg_mgr)
        combo = dialog.mic_combo

        # Option 0 is Default System Microphone
        assert combo.count() == 3
        assert combo.itemText(0) in ("Default System Microphone", "ميكروفون النظام الافتراضي")
        assert combo.itemData(0) is None

        # Options 1 and 2 are clean names without (Device X)
        assert combo.itemText(1) == "USB Audio Microphone"
        assert combo.itemData(1) == 17
        assert "(Device" not in combo.itemText(1)

        assert combo.itemText(2) == "Webcam Microphone"
        assert combo.itemData(2) == 18
        assert "(Device" not in combo.itemText(2)


def test_overlay_themes_and_rendering() -> None:
    """Verify all 4 themes switch cleanly and render in all overlay states."""
    cfg = Config()
    overlay = OverlayPill(audio_recorder=None, config=cfg)
    assert overlay.theme == "classic"

    themes = ["classic", "neon_glass", "matte_island", "fluent_mica"]
    states = [
        OverlayState.RECORDING,
        OverlayState.PROCESSING,
        OverlayState.ERROR,
        OverlayState.STANDBY,
    ]

    for th in themes:
        overlay.set_theme(th)
        assert overlay.theme == th
        assert cfg.overlay_theme == th

        for st in states:
            overlay._state = st
            # Calling paintEvent triggers the custom QPainter paths for the active theme
            overlay.paintEvent(None)

    # Invalid theme falls back to classic
    overlay.set_theme("unknown_theme_xyz")
    assert overlay.theme == "classic"


def test_settings_dialog_theme_selector_and_deepdml_model() -> None:
    """Verify SettingsDialog has live Theme switching and deepdml whisper model option."""
    cfg = Config(overlay_theme="classic", whisper_model="large-v3-turbo")
    cfg_mgr = ConfigManager(startup_manager=MockStartupManager())
    overlay = OverlayPill(audio_recorder=None, config=cfg, config_manager=cfg_mgr)

    dialog = SettingsDialog(config=cfg, config_manager=cfg_mgr, overlay=overlay)

    # Verify theme dropdown
    assert dialog.theme_combo.count() == 4
    assert dialog.theme_combo.itemData(0) == "classic"
    assert dialog.theme_combo.itemData(1) == "neon_glass"
    assert dialog.theme_combo.itemData(2) == "matte_island"
    assert dialog.theme_combo.itemData(3) == "fluent_mica"

    # Test live theme preview
    idx_matte = dialog.theme_combo.findData("matte_island")
    dialog.theme_combo.setCurrentIndex(idx_matte)
    assert overlay.theme == "matte_island"

    # Verify deepdml whisper model is present
    idx_deepdml = dialog.whisper_model_combo.findData("deepdml/faster-whisper-large-v3-turbo-ct2")
    assert idx_deepdml >= 0
    dialog.whisper_model_combo.setCurrentIndex(idx_deepdml)

    # Save and verify config persistence
    dialog._save_and_close()
    assert cfg.overlay_theme == "matte_island"
    assert cfg.whisper_model == "deepdml/faster-whisper-large-v3-turbo-ct2"


def test_theme_preview_widget_and_dialog_states() -> None:
    """Verify ThemePreviewWidget renders, animates, and synchronizes with SettingsDialog."""
    from voicetyper.ui.overlay import ThemePreviewWidget

    widget = ThemePreviewWidget(theme="neon_glass")
    assert widget.theme == "neon_glass"
    assert widget._state == OverlayState.RECORDING
    assert widget._anim_timer.isActive()

    # Test state changes
    widget.set_preview_state(OverlayState.PROCESSING)
    assert widget._state == OverlayState.PROCESSING
    widget.set_preview_state(OverlayState.ERROR, "Sample Error")
    assert widget._state == OverlayState.ERROR
    assert widget._error_text == "Sample Error"

    # Test animation frame advance
    phase0 = widget._wave_phase
    widget._on_anim_frame()
    assert widget._wave_phase > phase0

    # Test paint event doesn't crash
    widget.paintEvent(None)

    # Test SettingsDialog state buttons and cancel revert
    cfg = Config(overlay_theme="classic")
    cfg_mgr = ConfigManager(startup_manager=MockStartupManager())
    overlay = OverlayPill(audio_recorder=None, config=cfg, config_manager=cfg_mgr)
    dialog = SettingsDialog(config=cfg, config_manager=cfg_mgr, overlay=overlay)

    # Click processing state button
    dialog.state_btns[OverlayState.PROCESSING].click()
    assert dialog._current_preview_state == OverlayState.PROCESSING
    assert dialog.theme_preview._state == OverlayState.PROCESSING
    assert overlay._state == OverlayState.PROCESSING

    # Change theme combo
    idx_mica = dialog.theme_combo.findData("fluent_mica")
    dialog.theme_combo.setCurrentIndex(idx_mica)
    assert dialog.theme_preview.theme == "fluent_mica"
    assert overlay.theme == "fluent_mica"

    # Reject / cancel reverts overlay to initial theme
    dialog.reject()
    assert overlay.theme == "classic"
    assert overlay._state == OverlayState.STANDBY


def test_settings_dialog_model_cache_labels_and_progress_feedback() -> None:
    """Verify SettingsDialog displays dot indicator, clean titles without [Downloaded ✓], and updates progress bar."""
    cfg = Config(transcription_engine="local", whisper_model="large-v3-turbo")
    cfg_mgr = ConfigManager(startup_manager=MockStartupManager())
    overlay = OverlayPill(audio_recorder=None, config=cfg, config_manager=cfg_mgr)

    dialog = SettingsDialog(config=cfg, config_manager=cfg_mgr, overlay=overlay)

    # Verify model cache dot indicator exists
    assert hasattr(dialog, "model_cache_dot")
    assert dialog.model_cache_dot is not None

    # Verify model combo items have icons and clean labels (no '[Downloaded ✓]')
    found_size = False
    for i in range(dialog.whisper_model_combo.count()):
        text = dialog.whisper_model_combo.itemText(i)
        assert "[Downloaded ✓]" not in text
        icon = dialog.whisper_model_combo.itemIcon(i)
        assert not icon.isNull()
        if "MB" in text or "GB" in text:
            found_size = True

    assert found_size is True

    # Test progress signal updates
    dialog.show()
    dialog._on_model_progress_updated("Downloading model: 50.0 MB / 100.0 MB (50%)", 50.0)
    assert "50%" in dialog.model_status_label.text()
    assert not dialog.model_progress_bar.isHidden()
    assert dialog.model_progress_bar.value() == 50

    # Test complete progress signal hides progress bar
    dialog._on_model_progress_updated("Model ready ✓", 100.0)
    assert "ready" in dialog.model_status_label.text().lower() or "✓" in dialog.model_status_label.text()
    assert dialog.model_progress_bar.isHidden()
    dialog.close()




