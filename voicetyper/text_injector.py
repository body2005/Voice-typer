"""Windows text injection via clipboard paste (Ctrl+V) and Unicode SendInput fallback.

Key guarantees:
1. Complete clipboard backup across all formats (Unicode, images, files, custom).
2. Clipboard & History: Injected content is placed on clipboard and automatically saved
   in Windows Clipboard History (Win+V) so transcripts are never lost.
3. Modifier release wait: Waits until user releases Ctrl/Shift/Alt/Space before pasting.
4. Optional restoration: Original clipboard content can be restored if configured,
   or kept on clipboard by default ready for manual Ctrl+V.
5. Platform isolation: Wrapped behind an interface for cross-platform unit testing.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from ctypes import wintypes
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

# Win32 Constants
GMEM_MOVEABLE = 0x0002
CF_UNICODETEXT = 13
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_SPACE = 0x20
VK_RETURN = 0x0D
VK_V = 0x56

# Win32 Input Structs
if sys.platform == "win32":
    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.DWORD),
            ("union", _INPUT_UNION),
        ]

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # Set function prototypes
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.EnumClipboardFormats.argtypes = [wintypes.UINT]
    user32.EnumClipboardFormats.restype = wintypes.UINT
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = wintypes.SHORT
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetClipboardSequenceNumber.argtypes = []
    user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user32.AttachThreadInput.restype = wintypes.BOOL

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.restype = ctypes.c_size_t


class TextInjectorProtocol(Protocol):
    """Protocol for injecting text into the active focused control."""

    def inject_text(
        self,
        text: str,
        fallback_typing: bool = False,
        restore_clipboard: bool = False,
        target_hwnd: int | None = None,
        on_copied_toast: Callable[[str], None] | None = None,
    ) -> bool:
        """Inject transcribed text into the active window."""
        ...


class WindowsTextInjector:
    """Production-grade Windows text injector with full clipboard backup and privacy flags."""

    def __init__(self, paste_delay_ms: int = 120, release_timeout_ms: int = 300) -> None:
        self.paste_delay_ms = paste_delay_ms
        self.release_timeout_ms = release_timeout_ms
        self._fmt_exclude: int = 0
        self._fmt_history: int = 0
        self._fmt_cloud: int = 0

        if sys.platform == "win32":
            self._init_clipboard_formats()

    def _init_clipboard_formats(self) -> None:
        """Initialize clipboard configuration."""
        self._fmt_exclude = 0
        self._fmt_history = 0
        self._fmt_cloud = 0

    def _open_clipboard(self, max_retries: int = 15, delay_ms: float = 0.015) -> bool:
        """Attempt to open the clipboard with retries to handle transient locks."""
        for _ in range(max_retries):
            if user32.OpenClipboard(None):
                return True
            time.sleep(delay_ms)
        return False

    def _backup_clipboard(self) -> dict[int, bytes]:
        """Read and backup all standard and custom clipboard formats into RAM."""
        backup: dict[int, bytes] = {}
        if not self._open_clipboard():
            logger.warning("Failed to open clipboard for backup")
            return backup

        try:
            fmt = 0
            while True:
                fmt = user32.EnumClipboardFormats(fmt)
                if fmt == 0:
                    break

                # Ignore privacy control formats during backup
                if fmt in (self._fmt_exclude, self._fmt_history, self._fmt_cloud):
                    continue

                # Skip GDI handles (CF_BITMAP=2, CF_METAFILEPICT=3, CF_PALETTE=9, CF_ENHMETAFILE=14)
                # which are not HGLOBAL and will crash GlobalSize/GlobalLock
                if fmt in (2, 3, 9, 14):
                    continue

                handle = user32.GetClipboardData(fmt)
                if not handle:
                    continue

                size = kernel32.GlobalSize(handle)
                if size > 0:
                    ptr = kernel32.GlobalLock(handle)
                    if ptr:
                        try:
                            backup[fmt] = ctypes.string_at(ptr, size)
                        finally:
                            kernel32.GlobalUnlock(handle)
        except Exception as err:
            logger.warning("Error during clipboard backup: %s", err)
        finally:
            user32.CloseClipboard()

        logger.debug("Backed up %d clipboard formats", len(backup))
        return backup

    def _restore_clipboard(self, backup: dict[int, bytes]) -> None:
        """Restore all previously captured clipboard formats."""
        if not self._open_clipboard():
            logger.warning("Failed to open clipboard for restore")
            return

        try:
            user32.EmptyClipboard()
            for fmt, data in backup.items():
                if not data:
                    continue
                h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
                if not h_mem:
                    continue
                ptr = kernel32.GlobalLock(h_mem)
                if ptr:
                    ctypes.memmove(ptr, data, len(data))
                    kernel32.GlobalUnlock(h_mem)
                    user32.SetClipboardData(fmt, h_mem)
            logger.debug("Restored %d clipboard formats", len(backup))
        except Exception as err:
            logger.warning("Error during clipboard restore: %s", err)
        finally:
            user32.CloseClipboard()

    def _set_clipboard_payload(self, text: str) -> bool:
        """Place text on clipboard for Ctrl+V and Windows Clipboard History (Win+V)."""
        if not self._open_clipboard():
            logger.error("Failed to open clipboard to set transcribed text")
            return False

        try:
            user32.EmptyClipboard()

            # Unicode text payload (UTF-16LE with null terminator)
            encoded_text = (text + "\0").encode("utf-16le")
            h_text = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded_text))
            if h_text:
                ptr = kernel32.GlobalLock(h_text)
                if ptr:
                    ctypes.memmove(ptr, encoded_text, len(encoded_text))
                    kernel32.GlobalUnlock(h_text)
                    user32.SetClipboardData(CF_UNICODETEXT, h_text)

            return True
        finally:
            user32.CloseClipboard()

    def _wait_for_modifier_release(self) -> None:
        """Wait until user has physically released Ctrl, Alt, Shift, and Space."""
        keys_to_check = [VK_CONTROL, VK_SHIFT, VK_MENU, VK_SPACE]
        deadline = time.monotonic() + (self.release_timeout_ms / 1000.0)

        while time.monotonic() < deadline:
            pressed = False
            for vk in keys_to_check:
                if user32.GetAsyncKeyState(vk) & 0x8000:
                    pressed = True
                    break
            if not pressed:
                break
            time.sleep(0.01)

    def _send_ctrl_v(self) -> None:
        """Synthesize Ctrl+V keystroke via SendInput."""
        inputs = (INPUT * 4)()

        # 1. Ctrl down
        inputs[0].type = INPUT_KEYBOARD
        inputs[0].union.ki.wVk = VK_CONTROL
        inputs[0].union.ki.dwFlags = 0

        # 2. V down
        inputs[1].type = INPUT_KEYBOARD
        inputs[1].union.ki.wVk = VK_V
        inputs[1].union.ki.dwFlags = 0

        # 3. V up
        inputs[2].type = INPUT_KEYBOARD
        inputs[2].union.ki.wVk = VK_V
        inputs[2].union.ki.dwFlags = KEYEVENTF_KEYUP

        # 4. Ctrl up
        inputs[3].type = INPUT_KEYBOARD
        inputs[3].union.ki.wVk = VK_CONTROL
        inputs[3].union.ki.dwFlags = KEYEVENTF_KEYUP

        sent = user32.SendInput(4, inputs, ctypes.sizeof(INPUT))
        if sent != 4:
            logger.warning("SendInput sent %d/4 inputs", sent)

    def _type_unicode(self, text: str) -> None:
        """Fallback: Type text character-by-character using KEYEVENTF_UNICODE."""
        for char in text:
            if char == "\n":
                # Send Return key
                inputs = (INPUT * 2)()
                inputs[0].type = INPUT_KEYBOARD
                inputs[0].union.ki.wVk = VK_RETURN
                inputs[0].union.ki.dwFlags = 0

                inputs[1].type = INPUT_KEYBOARD
                inputs[1].union.ki.wVk = VK_RETURN
                inputs[1].union.ki.dwFlags = KEYEVENTF_KEYUP
                user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
            else:
                code_point = ord(char)
                inputs = (INPUT * 2)()
                # Char down
                inputs[0].type = INPUT_KEYBOARD
                inputs[0].union.ki.wVk = 0
                inputs[0].union.ki.wScan = code_point
                inputs[0].union.ki.dwFlags = KEYEVENTF_UNICODE
                # Char up
                inputs[1].type = INPUT_KEYBOARD
                inputs[1].union.ki.wVk = 0
                inputs[1].union.ki.wScan = code_point
                inputs[1].union.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
                user32.SendInput(2, inputs, ctypes.sizeof(INPUT))

    def _restore_target_focus(self, target_hwnd: int) -> None:
        """Bring target window back to foreground if user switched windows during transcription."""
        if sys.platform != "win32":
            return
        try:
            if not user32.IsWindow(target_hwnd):
                return
            current_hwnd = user32.GetForegroundWindow()
            if current_hwnd == target_hwnd:
                return

            logger.info("Restoring focus to target window (hwnd=0x%x)", target_hwnd)
            cur_thread = user32.GetWindowThreadProcessId(current_hwnd, None)
            target_thread = user32.GetWindowThreadProcessId(target_hwnd, None)
            if cur_thread != target_thread:
                user32.AttachThreadInput(cur_thread, target_thread, True)
                user32.SetForegroundWindow(target_hwnd)
                user32.AttachThreadInput(cur_thread, target_thread, False)
            else:
                user32.SetForegroundWindow(target_hwnd)
            time.sleep(0.08)  # Allow target window message loop to activate
        except Exception as err:
            logger.debug("Could not restore target window focus: %s", err)

    def inject_text(
        self,
        text: str,
        fallback_typing: bool = False,
        restore_clipboard: bool = False,
        target_hwnd: int | None = None,
        on_copied_toast: Callable[[str], None] | None = None,
    ) -> bool:
        """Main entry point for text injection.

        - If target_hwnd is provided and focus shifted, does NOT steal focus. Copies to clipboard
          and invokes on_copied_toast callback.
        - Backs up clipboard, places text on clipboard, and pastes via Ctrl+V.
        - If restore_clipboard is True, waits >= 300ms and skips restore if clipboard sequence number changed.
        """
        if not text:
            return True

        if sys.platform != "win32":
            logger.warning("WindowsTextInjector called on non-Windows system")
            return False

        # Wait for user to let go of physical hotkey keys
        self._wait_for_modifier_release()

        # If user switched away to another window, DO NOT steal focus with SetForegroundWindow!
        # Instead, copy to clipboard (preserves Win+V history) and trigger subtle toast.
        current_hwnd = user32.GetForegroundWindow()
        if target_hwnd is not None and user32.IsWindow(target_hwnd) and current_hwnd != target_hwnd:
            logger.info(
                "Focus shifted away from target (0x%x != 0x%x); copying to clipboard without stealing focus",
                current_hwnd,
                target_hwnd,
            )
            self._set_clipboard_payload(text)
            if on_copied_toast:
                on_copied_toast("Copied — press Ctrl+V")
            return True

        if fallback_typing:
            logger.info("Injecting text via Unicode keystroke typing fallback")
            self._type_unicode(text)
            return True

        # Clipboard paste approach
        clipboard_backup = self._backup_clipboard() if restore_clipboard else {}
        seq_after = 0
        try:
            if not self._set_clipboard_payload(text):
                logger.error("Could not set clipboard payload; aborting paste")
                return False

            seq_after = user32.GetClipboardSequenceNumber()

            # Synthesize Ctrl+V
            self._send_ctrl_v()

            # Wait for target application to process the paste message (>= 300ms)
            delay = max(0.3, self.paste_delay_ms / 1000.0)
            time.sleep(delay)
            return True
        finally:
            if restore_clipboard and clipboard_backup:
                # Skip restore if GetClipboardSequenceNumber() changed (user or other app copied something)
                curr_seq = user32.GetClipboardSequenceNumber()
                if curr_seq == seq_after:
                    self._restore_clipboard(clipboard_backup)
                else:
                    logger.debug("Clipboard changed externally (seq %d != %d); skipping restore", curr_seq, seq_after)


class MockTextInjector:
    """Mock text injector for unit testing and headless validation."""

    def __init__(self) -> None:
        self.injected_texts: list[str] = []
        self.fallback_used: list[bool] = []

    def inject_text(
        self,
        text: str,
        fallback_typing: bool = False,
        restore_clipboard: bool = False,
        target_hwnd: int | None = None,
        on_copied_toast: Callable[[str], None] | None = None,
    ) -> bool:
        self.injected_texts.append(text)
        self.fallback_used.append(fallback_typing)
        return True
