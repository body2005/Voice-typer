"""Unit tests for text injection, clipboard backup, and privacy flags."""

import sys
from unittest.mock import patch

from voicetyper.text_injector import MockTextInjector, WindowsTextInjector


def test_mock_text_injector() -> None:
    """Verify mock injector captures text and typing flags."""
    injector = MockTextInjector()
    assert injector.inject_text("Hello world", fallback_typing=False) is True
    assert injector.inject_text("Second text", fallback_typing=True) is True
    assert injector.injected_texts == ["Hello world", "Second text"]
    assert injector.fallback_used == [False, True]


def test_windows_injector_backup_and_restore_real() -> None:
    """Verify WindowsTextInjector preserves clipboard content on Windows."""
    if sys.platform != "win32":
        return

    import win32clipboard

    injector = WindowsTextInjector()

    # Save user's actual real clipboard before test so we leave zero pollution
    initial_backup = injector._backup_clipboard()

    try:
        # Pre-populate clipboard with a test string
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, "Test Clipboard Content 999")
        win32clipboard.CloseClipboard()

        # Backup
        backup = injector._backup_clipboard()
        assert len(backup) > 0

        # Overwrite
        assert injector._set_clipboard_payload("Temporary Payload") is True

        # Check clipboard currently has the payload
        win32clipboard.OpenClipboard()
        current_data = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
        win32clipboard.CloseClipboard()
        assert current_data == "Temporary Payload"

        # Restore
        injector._restore_clipboard(backup)

        # Check clipboard restored
        win32clipboard.OpenClipboard()
        restored_data = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
        win32clipboard.CloseClipboard()
        assert restored_data == "Test Clipboard Content 999"
    finally:
        # Always restore user's real clipboard content
        injector._restore_clipboard(initial_backup)


def test_windows_injector_empty_text_noop() -> None:
    """Verify empty text injection is a no-op."""
    injector = WindowsTextInjector()
    with patch.object(injector, "_send_ctrl_v") as mock_paste:
        assert injector.inject_text("") is True
        mock_paste.assert_not_called()


def test_windows_injector_clipboard_history_and_persistence() -> None:
    """Verify WindowsTextInjector keeps transcribed text in clipboard and enables Win+V history."""
    if sys.platform != "win32":
        return

    import win32clipboard

    injector = WindowsTextInjector()
    initial_backup = injector._backup_clipboard()

    try:
        # Inject text with restore_clipboard=False (default behavior)
        with patch.object(injector, "_send_ctrl_v"):
            success = injector.inject_text("Persistent Transcription Text", restore_clipboard=False)
            assert success is True

        # Verify clipboard still contains the transcribed text
        win32clipboard.OpenClipboard()
        current_data = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
        win32clipboard.CloseClipboard()
        assert current_data == "Persistent Transcription Text"
    finally:
        injector._restore_clipboard(initial_backup)


def test_windows_injector_focus_shifted_no_steal() -> None:
    """Verify focus is not stolen when target_hwnd != foreground window; toast is triggered."""
    if sys.platform != "win32":
        return

    from unittest.mock import patch

    injector = WindowsTextInjector()
    toast_msgs: list[str] = []

    with patch("voicetyper.text_injector.user32.GetForegroundWindow", return_value=12345), \
         patch("voicetyper.text_injector.user32.IsWindow", return_value=True), \
         patch.object(injector, "_set_clipboard_payload", return_value=True) as mock_set, \
         patch.object(injector, "_send_ctrl_v") as mock_paste:

        res = injector.inject_text(
            "Hello away",
            target_hwnd=99999,  # Focus shifted away! (99999 != 12345)
            on_copied_toast=lambda msg: toast_msgs.append(msg),
        )

        assert res is True
        mock_set.assert_called_once_with("Hello away")
        mock_paste.assert_not_called()  # Did NOT send Ctrl+V to random foreground window
        assert len(toast_msgs) == 1
        assert "Copied" in toast_msgs[0]


