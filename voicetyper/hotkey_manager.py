"""Low-level Windows keyboard hook manager supporting 1-, 2-, and 3-key combinations.

Key guarantees:
1. Low-level WH_KEYBOARD_LL hook providing accurate key-down and key-up events.
2. Supports flexible 1-, 2-, or 3-key hotkey combinations:
   - Single keys: F1-F24, Right Alt (AltGr), CapsLock, Insert, etc.
   - Modifier-only combos: Ctrl+Shift, Ctrl+Alt, Alt+Shift, etc.
   - 2- or 3-key combos: Ctrl+Shift+V, Alt+Z, Ctrl+Space, etc.
3. Complete suppression of non-modifier trigger keys while the hotkey is active so
   they are never delivered to the focused application.
4. Auto-repeat suppression: ignores OS repeat events while held.
5. Push-to-Talk logic: start recording when all keys in the combination are held,
   and stop recording as soon as any key in the combination is released.
6. Esc key cancels recording immediately and suppresses Esc from reaching active apps.
7. Clean pause/resume support and zero hook leakage on application exit.
8. Thread isolation: emits Qt signals (queued) so the hook thread never touches UI widgets.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Callable, Protocol

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# Windows Virtual Key Codes
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_PAUSE = 0x13
VK_CAPITAL = 0x14  # Caps Lock
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_PRIOR = 0x21  # Page Up
VK_NEXT = 0x22  # Page Down
VK_END = 0x23
VK_HOME = 0x24
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_SNAPSHOT = 0x2C  # Print Screen
VK_INSERT = 0x2D
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5  # Right Alt / AltGr

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

LLKHF_INJECTED = 0x00000010

MODIFIER_VKS: set[int] = {
    VK_CONTROL,
    VK_LCONTROL,
    VK_RCONTROL,
    VK_SHIFT,
    VK_LSHIFT,
    VK_RSHIFT,
    VK_MENU,
    VK_LMENU,
    VK_RMENU,
    VK_LWIN,
    VK_RWIN,
}

KEY_NAME_TO_VK_SET: dict[str, set[int]] = {
    # Modifiers
    "ctrl": {VK_CONTROL, VK_LCONTROL, VK_RCONTROL},
    "control": {VK_CONTROL, VK_LCONTROL, VK_RCONTROL},
    "lctrl": {VK_LCONTROL, VK_CONTROL},
    "left ctrl": {VK_LCONTROL, VK_CONTROL},
    "rctrl": {VK_RCONTROL, VK_CONTROL},
    "right ctrl": {VK_RCONTROL, VK_CONTROL},
    "shift": {VK_SHIFT, VK_LSHIFT, VK_RSHIFT},
    "lshift": {VK_LSHIFT, VK_SHIFT},
    "left shift": {VK_LSHIFT, VK_SHIFT},
    "rshift": {VK_RSHIFT, VK_SHIFT},
    "right shift": {VK_RSHIFT, VK_SHIFT},
    "alt": {VK_MENU, VK_LMENU, VK_RMENU},
    "menu": {VK_MENU, VK_LMENU, VK_RMENU},
    "lalt": {VK_LMENU, VK_MENU},
    "left alt": {VK_LMENU, VK_MENU},
    "ralt": {VK_RMENU},
    "right alt": {VK_RMENU},
    "altgr": {VK_RMENU},
    "win": {VK_LWIN, VK_RWIN},
    "windows": {VK_LWIN, VK_RWIN},
    "cmd": {VK_LWIN, VK_RWIN},
    "super": {VK_LWIN, VK_RWIN},
    "lwin": {VK_LWIN},
    "rwin": {VK_RWIN},
    # Common keys
    "space": {VK_SPACE},
    "capslock": {VK_CAPITAL},
    "caps_lock": {VK_CAPITAL},
    "caps": {VK_CAPITAL},
    "insert": {VK_INSERT},
    "delete": {VK_DELETE},
    "del": {VK_DELETE},
    "tab": {VK_TAB},
    "enter": {VK_RETURN},
    "return": {VK_RETURN},
    "backspace": {VK_BACK},
    "esc": {VK_ESCAPE},
    "escape": {VK_ESCAPE},
    "pause": {VK_PAUSE},
    "pageup": {VK_PRIOR},
    "pgup": {VK_PRIOR},
    "pagedown": {VK_NEXT},
    "pgdn": {VK_NEXT},
    "home": {VK_HOME},
    "end": {VK_END},
    "left": {VK_LEFT},
    "up": {VK_UP},
    "right": {VK_RIGHT},
    "down": {VK_DOWN},
    "prtsc": {VK_SNAPSHOT},
    "printscreen": {VK_SNAPSHOT},
}

# Add F1-F24
for _i in range(1, 25):
    KEY_NAME_TO_VK_SET[f"f{_i}"] = {0x70 + (_i - 1)}

# Add letters A-Z
for _c in "abcdefghijklmnopqrstuvwxyz":
    KEY_NAME_TO_VK_SET[_c] = {ord(_c.upper())}

# Add digits 0-9
for _d in "0123456789":
    KEY_NAME_TO_VK_SET[_d] = {ord(_d)}


def parse_hotkey_slots(hotkey_str: str) -> tuple[list[set[int]], set[int]]:
    """Parse a 1-, 2-, or 3-key hotkey string into slots and non-modifier VK sets.

    Returns:
        (slots, non_modifier_vks)
        - slots: list of VK sets, where each slot must have at least one VK pressed.
        - non_modifier_vks: set of VK codes that should be suppressed from the OS.
    """
    clean_str = hotkey_str.strip().lower()
    if not clean_str:
        return [{VK_CONTROL, VK_LCONTROL, VK_RCONTROL}, {VK_SHIFT, VK_LSHIFT, VK_RSHIFT}], set()

    # Split on '+' or treat single tokens cleanly
    parts = [p.strip() for p in clean_str.split("+") if p.strip()]
    if not parts:
        parts = [clean_str]

    slots: list[set[int]] = []
    non_modifier_vks: set[int] = set()

    for part in parts:
        if part in KEY_NAME_TO_VK_SET:
            vks = set(KEY_NAME_TO_VK_SET[part])
            slots.append(vks)
            # Check if this slot contains non-modifiers
            if not vks.issubset(MODIFIER_VKS):
                non_modifier_vks.update(vks)
            elif len(parts) == 1 and part in ("ralt", "right alt", "altgr"):
                # If Right Alt is the sole PTT hotkey, suppress it so menu bars don't open
                non_modifier_vks.update(vks)
        elif len(part) == 1:
            code = ord(part.upper())
            slots.append({code})
            non_modifier_vks.add(code)
        else:
            logger.warning("Unrecognized hotkey token: '%s'", part)

    # Fallback to Ctrl+Shift if parsing produced no valid slots
    if not slots:
        slots = [{VK_CONTROL, VK_LCONTROL, VK_RCONTROL}, {VK_SHIFT, VK_LSHIFT, VK_RSHIFT}]

    return slots, non_modifier_vks


def parse_hotkey_string(hotkey_str: str) -> tuple[set[int], int]:
    """Backward-compatible hotkey string parser returning (modifier_vk_set, trigger_vk)."""
    slots, non_mods = parse_hotkey_slots(hotkey_str)
    if not slots:
        return ({VK_CONTROL, VK_LCONTROL, VK_RCONTROL}, VK_SPACE)

    if len(slots) >= 2:
        mod_vks = set().union(*slots[:-1])
        trigger_vk = next(iter(slots[-1]))
        return mod_vks, trigger_vk
    else:
        only_slot = slots[0]
        if only_slot.issubset(MODIFIER_VKS):
            return only_slot, VK_SPACE
        else:
            return {VK_CONTROL, VK_LCONTROL, VK_RCONTROL}, next(iter(only_slot))


def format_hotkey_display(hotkey_str: str) -> str:
    """Format hotkey string into a clean UI display representation."""
    parts = [p.strip() for p in hotkey_str.split("+") if p.strip()]
    if not parts:
        return "Ctrl + Shift"

    display_parts: list[str] = []
    for p in parts:
        lower = p.lower()
        if lower in ("ctrl", "control"):
            display_parts.append("Ctrl")
        elif lower in ("shift", "lshift", "rshift"):
            display_parts.append("Shift")
        elif lower in ("alt", "menu", "lalt"):
            display_parts.append("Alt")
        elif lower in ("ralt", "right alt", "altgr"):
            display_parts.append("Right Alt")
        elif lower in ("win", "cmd", "super", "windows"):
            display_parts.append("Win")
        elif lower == "space":
            display_parts.append("Space")
        elif lower.startswith("f") and lower[1:].isdigit():
            display_parts.append(lower.upper())
        elif len(lower) == 1:
            display_parts.append(lower.upper())
        else:
            display_parts.append(p.title())

    return " + ".join(display_parts)


class HotkeyManagerProtocol(Protocol):
    """Protocol for global hotkey management."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def pause(self) -> None:
        ...

    def resume(self) -> None:
        ...

    def set_recording_active(self, active: bool) -> None:
        ...

    def set_toggle_mode(self, enabled: bool) -> None:
        ...

    def is_paused(self) -> bool:
        ...

    def set_hotkey(self, hotkey_str: str) -> None:
        ...


class Win32HotkeyManager(QObject):
    """Low-level Windows keyboard hook manager supporting 1-, 2-, and 3-key hotkeys."""

    # Qt Signals for controller communication (queued thread-safe execution)
    hotkey_down = pyqtSignal()  # type: ignore[call-arg]
    hotkey_up = pyqtSignal()  # type: ignore[call-arg]
    hotkey_cancel = pyqtSignal()  # type: ignore[call-arg]

    def __init__(self, hotkey_str: str = "ctrl+shift", parent: QObject | None = None) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.hotkey_str = hotkey_str
        self._slots, self._non_modifier_vks = parse_hotkey_slots(hotkey_str)

        self._lock = threading.Lock()
        self._paused = False
        self._is_active = False
        self._recording_active = False
        self._toggle_mode = False
        self._pressed_vks: set[int] = set()
        self._listener: Any = None

        # Callbacks for headless testing
        self.on_down_callbacks: list[Callable[[], None]] = []
        self.on_up_callbacks: list[Callable[[], None]] = []
        self.on_cancel_callbacks: list[Callable[[], None]] = []

    def set_hotkey(self, hotkey_str: str) -> None:
        """Update active hotkey combination."""
        with self._lock:
            self.hotkey_str = hotkey_str
            self._slots, self._non_modifier_vks = parse_hotkey_slots(hotkey_str)
            self._is_active = False
            self._pressed_vks.clear()
            slots_count = len(self._slots)
            non_mod_hex = [hex(k) for k in self._non_modifier_vks]

        logger.info(
            "Updated hotkey to '%s' (slots=%d, non_modifiers=%s)",
            hotkey_str,
            slots_count,
            non_mod_hex,
        )

    def set_recording_active(self, active: bool) -> None:
        """Inform hotkey manager whether dictation recording is active.

        Enables global Esc key cancellation and suppression in both Push-to-Talk
        and Toggle dictation modes.
        """
        with self._lock:
            self._recording_active = active
            if not active:
                self._is_active = False

    def set_toggle_mode(self, enabled: bool) -> None:
        """Configure whether Toggle dictation mode is enabled."""
        with self._lock:
            self._toggle_mode = enabled

    def is_paused(self) -> bool:
        """Return True if hotkey listening is paused."""
        with self._lock:
            return self._paused

    def pause(self) -> None:
        """Temporarily pause hotkey processing without destroying the hook."""
        with self._lock:
            self._paused = True
            self._is_active = False
            self._pressed_vks.clear()
        logger.info("Hotkey manager paused")

    def resume(self) -> None:
        """Resume hotkey processing."""
        with self._lock:
            self._paused = False
            self._is_active = False
            self._pressed_vks.clear()
        logger.info("Hotkey manager resumed")

    def start(self) -> None:
        """Install low-level Windows keyboard hook using pynput with win32_event_filter."""
        if sys.platform != "win32":
            logger.warning("Win32HotkeyManager started on non-Windows platform")
            return

        if self._listener is not None:
            return

        try:
            from pynput import keyboard

            self._listener = keyboard.Listener(win32_event_filter=self._win32_filter)
            self._listener.daemon = True
            self._listener.start()
            logger.info("Low-level keyboard hook installed successfully for '%s'", self.hotkey_str)
        except Exception as err:
            logger.error("Failed to install low-level keyboard hook: %s", err)

    def stop(self) -> None:
        """Safely unhook and terminate listener."""
        if self._listener is not None:
            try:
                self._listener.stop()
                logger.info("Low-level keyboard hook stopped")
            except Exception as err:
                logger.warning("Error stopping keyboard listener: %s", err)
            finally:
                self._listener = None
        with self._lock:
            self._is_active = False
            self._pressed_vks.clear()

    def _suppress(self) -> bool:
        """Suppress the current event using pynput suppress_event() if listener is active."""
        if self._listener is not None and hasattr(self._listener, "suppress_event"):
            self._listener.suppress_event()
        return False

    def _win32_filter(self, msg: int, data: Any) -> bool:
        """Windows low-level keyboard hook callback.

        Calls self._suppress() to suppress the key event from reaching other apps.
        Returning True lets the key event pass through normally.
        """
        with self._lock:
            if self._paused:
                return True

        try:
            target = getattr(data, "contents", data)
            flags = getattr(target, "flags", 0)
            # Never intercept our own synthetic SendInput events
            if flags & LLKHF_INJECTED:
                return True

            vk_code = getattr(target, "vkCode", 0)

            # Cancel on Esc while hotkey or recording is active
            if msg in (WM_KEYDOWN, WM_SYSKEYDOWN) and vk_code == VK_ESCAPE:
                with self._lock:
                    should_cancel = self._is_active or self._recording_active
                    if should_cancel:
                        self._is_active = False
                        self._recording_active = False
                        self._pressed_vks.clear()

                if should_cancel:
                    logger.info("Hotkey CANCEL (Esc) detected")
                    self.hotkey_cancel.emit()
                    for cb in self.on_cancel_callbacks:
                        cb()
                    return self._suppress()  # Suppress Esc so it doesn't close active tabs/windows

            # Map specific left/right modifiers to generic counterparts as well
            mapped_vks = {vk_code}
            if vk_code in (VK_LCONTROL, VK_RCONTROL):
                mapped_vks.add(VK_CONTROL)
            elif vk_code in (VK_LSHIFT, VK_RSHIFT):
                mapped_vks.add(VK_SHIFT)
            elif vk_code in (VK_LMENU, VK_RMENU):
                mapped_vks.add(VK_MENU)

            # Key Down Events
            if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
                with self._lock:
                    self._pressed_vks.update(mapped_vks)

                    # Check if all slots in the hotkey set are now satisfied
                    all_satisfied = bool(self._slots) and all(
                        bool(slot & self._pressed_vks) for slot in self._slots
                    )

                    fire_down = False
                    if all_satisfied and not self._is_active:
                        self._is_active = True
                        fire_down = True

                    is_suppressed = all_satisfied and (vk_code in self._non_modifier_vks)

                if fire_down:
                    logger.info("Hotkey DOWN detected for '%s'", self.hotkey_str)
                    self.hotkey_down.emit()
                    for cb in self.on_down_callbacks:
                        cb()

                if is_suppressed:
                    return self._suppress()
                return True

            # Key Up Events
            elif msg in (WM_KEYUP, WM_SYSKEYUP):
                with self._lock:
                    self._pressed_vks.discard(vk_code)
                    if vk_code in (VK_LCONTROL, VK_RCONTROL) and not (self._pressed_vks & {VK_LCONTROL, VK_RCONTROL}):
                        self._pressed_vks.discard(VK_CONTROL)
                    if vk_code in (VK_LSHIFT, VK_RSHIFT) and not (self._pressed_vks & {VK_LSHIFT, VK_RSHIFT}):
                        self._pressed_vks.discard(VK_SHIFT)
                    if vk_code in (VK_LMENU, VK_RMENU) and not (self._pressed_vks & {VK_LMENU, VK_RMENU}):
                        self._pressed_vks.discard(VK_MENU)

                    all_satisfied = bool(self._slots) and all(
                        bool(slot & self._pressed_vks) for slot in self._slots
                    )

                    was_active = self._is_active
                    fire_up = False
                    if was_active and not all_satisfied:
                        self._is_active = False
                        fire_up = True

                    suppress_up = (vk_code in self._non_modifier_vks) and (was_active or fire_up)

                if fire_up:
                    logger.info("Hotkey UP detected for '%s'", self.hotkey_str)
                    self.hotkey_up.emit()
                    for cb in self.on_up_callbacks:
                        cb()

                if suppress_up:
                    return self._suppress()
                return True

            return True
        except Exception as err:
            if type(err).__name__ == "SuppressException":
                raise
            logger.error("Error in hotkey filter: %s", err, exc_info=True)
            return True
            return True


class MockHotkeyManager(QObject):
    """Mock hotkey manager for cross-platform unit testing."""

    hotkey_down = pyqtSignal()  # type: ignore[call-arg]
    hotkey_up = pyqtSignal()  # type: ignore[call-arg]
    hotkey_cancel = pyqtSignal()  # type: ignore[call-arg]

    def __init__(self, hotkey_str: str = "ctrl+shift", parent: QObject | None = None) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.hotkey_str = hotkey_str
        self._paused = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def set_toggle_mode(self, enabled: bool) -> None:
        self._toggle_mode = enabled

    def set_recording_active(self, active: bool) -> None:
        """Inform hotkey manager whether dictation recording is active.

        Enables global Esc key cancellation and suppression in both Push-to-Talk
        and Toggle dictation modes.
        """
        self._recording_active = active
        if not active:
            self._is_active = False

    def is_paused(self) -> bool:
        return self._paused

    def set_hotkey(self, hotkey_str: str) -> None:
        self.hotkey_str = hotkey_str

    def trigger_down(self) -> None:
        if not self._paused:
            self.hotkey_down.emit()

    def trigger_up(self) -> None:
        if not self._paused:
            self.hotkey_up.emit()

    def trigger_cancel(self) -> None:
        if not self._paused:
            self.hotkey_cancel.emit()
