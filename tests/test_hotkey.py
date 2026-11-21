"""Unit tests for hotkey parsing and low-level hook filter logic."""

from types import SimpleNamespace

from voicetyper.hotkey_manager import (
    VK_CONTROL,
    VK_ESCAPE,
    VK_MENU,
    VK_SHIFT,
    VK_SPACE,
    WM_KEYDOWN,
    WM_KEYUP,
    Win32HotkeyManager,
    parse_hotkey_string,
)


def test_parse_hotkey_string() -> None:
    """Test hotkey string parsing for modifiers and trigger keys."""
    # Default Ctrl+Space
    mods, trig = parse_hotkey_string("ctrl+space")
    assert VK_CONTROL in mods
    assert trig == VK_SPACE

    # Alt+Space
    mods, trig = parse_hotkey_string("alt+space")
    assert VK_MENU in mods
    assert trig == VK_SPACE

    # Shift+F5
    mods, trig = parse_hotkey_string("shift+f5")
    assert VK_SHIFT in mods
    assert trig == 0x70 + 4  # VK_F5 = 0x74

    # Single letter trigger Ctrl+V
    mods, trig = parse_hotkey_string("ctrl+v")
    assert VK_CONTROL in mods
    assert trig == ord("V")


def test_win32_filter_down_up_and_suppression() -> None:
    """Test that trigger key is suppressed while modifier is held and signals are emitted."""
    mgr = Win32HotkeyManager("ctrl+space")

    down_count = 0
    up_count = 0
    mgr.on_down_callbacks.append(lambda: nonlocal_down())
    mgr.on_up_callbacks.append(lambda: nonlocal_up())

    def nonlocal_down() -> None:
        nonlocal down_count
        down_count += 1

    def nonlocal_up() -> None:
        nonlocal up_count
        up_count += 1

    # 1. Ctrl down -> pass through (True)
    data_ctrl = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_CONTROL, flags=0))
    res = mgr._win32_filter(WM_KEYDOWN, data_ctrl)
    assert res is True
    assert down_count == 0

    # 2. Space down while Ctrl is down -> SUPPRESSED (False) and emits hotkey_down
    data_space = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_SPACE, flags=0))
    res = mgr._win32_filter(WM_KEYDOWN, data_space)
    assert res is False
    assert down_count == 1

    # 3. Space repeat down (OS auto-repeat) -> SUPPRESSED (False) and NO duplicate signal
    res = mgr._win32_filter(WM_KEYDOWN, data_space)
    assert res is False
    assert down_count == 1  # Still 1!

    # 4. Space up -> SUPPRESSED (False) and emits hotkey_up
    res = mgr._win32_filter(WM_KEYUP, data_space)
    assert res is False
    assert up_count == 1

    # 5. Space up again when not held -> passes through (True)
    res = mgr._win32_filter(WM_KEYUP, data_space)
    assert res is True
    assert up_count == 1


def test_win32_filter_cancel_on_esc() -> None:
    """Test pressing Esc while hotkey is held emits cancel and suppresses Esc."""
    mgr = Win32HotkeyManager("ctrl+space")
    cancel_count = 0
    mgr.on_cancel_callbacks.append(lambda: nonlocal_cancel())

    def nonlocal_cancel() -> None:
        nonlocal cancel_count
        cancel_count += 1

    # Hold Ctrl and Space
    data_ctrl = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_CONTROL, flags=0))
    data_space = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_SPACE, flags=0))
    mgr._win32_filter(WM_KEYDOWN, data_ctrl)
    mgr._win32_filter(WM_KEYDOWN, data_space)

    # Press Esc -> SUPPRESSED (False) and emits cancel
    data_esc = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_ESCAPE, flags=0))
    res = mgr._win32_filter(WM_KEYDOWN, data_esc)
    assert res is False
    assert cancel_count == 1


def test_win32_filter_pause_mode() -> None:
    """Test that pausing lets all keys pass through without emitting signals."""
    mgr = Win32HotkeyManager("ctrl+space")
    mgr.pause()
    assert mgr.is_paused() is True

    down_count = 0
    mgr.on_down_callbacks.append(lambda: nonlocal_down())

    def nonlocal_down() -> None:
        nonlocal down_count
        down_count += 1

    data_ctrl = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_CONTROL, flags=0))
    data_space = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_SPACE, flags=0))

    assert mgr._win32_filter(WM_KEYDOWN, data_ctrl) is True
    assert mgr._win32_filter(WM_KEYDOWN, data_space) is True
    assert down_count == 0

    mgr.resume()
    assert mgr.is_paused() is False


def test_parse_hotkey_slots_1_2_3_keys() -> None:
    """Test flexible parsing of 1-, 2-, and 3-key combinations."""
    from voicetyper.hotkey_manager import VK_RMENU, parse_hotkey_slots

    # 1-key: F8
    slots, non_mods = parse_hotkey_slots("f8")
    assert len(slots) == 1
    assert 0x77 in slots[0]  # VK_F8
    assert 0x77 in non_mods

    # 1-key: Right Alt
    slots, non_mods = parse_hotkey_slots("right alt")
    assert len(slots) == 1
    assert VK_RMENU in slots[0]

    # 2-key modifier-only: Ctrl + Shift
    slots, non_mods = parse_hotkey_slots("ctrl+shift")
    assert len(slots) == 2
    assert len(non_mods) == 0  # Modifiers pass through

    # 3-key: Ctrl + Shift + V
    slots, non_mods = parse_hotkey_slots("ctrl+shift+v")
    assert len(slots) == 3
    assert ord("V") in slots[2]
    assert ord("V") in non_mods


def test_modifier_only_combo_ctrl_shift() -> None:
    """Test modifier-only combo (Ctrl+Shift): triggers on all held, releases on any up."""
    mgr = Win32HotkeyManager("ctrl+shift")

    down_count = 0
    up_count = 0
    mgr.on_down_callbacks.append(lambda: nonlocal_down())
    mgr.on_up_callbacks.append(lambda: nonlocal_up())

    def nonlocal_down() -> None:
        nonlocal down_count
        down_count += 1

    def nonlocal_up() -> None:
        nonlocal up_count
        up_count += 1

    data_ctrl = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_CONTROL, flags=0))
    data_shift = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_SHIFT, flags=0))

    # 1. Ctrl down -> passes through, hotkey not down yet
    assert mgr._win32_filter(WM_KEYDOWN, data_ctrl) is True
    assert down_count == 0

    # 2. Shift down while Ctrl down -> passes through, hotkey DOWN emitted
    assert mgr._win32_filter(WM_KEYDOWN, data_shift) is True
    assert down_count == 1

    # 3. Auto-repeat Shift -> passes through, no duplicate signal
    assert mgr._win32_filter(WM_KEYDOWN, data_shift) is True
    assert down_count == 1

    # 4. Release Shift -> passes through, hotkey UP emitted
    assert mgr._win32_filter(WM_KEYUP, data_shift) is True
    assert up_count == 1


def test_single_key_f8_suppression() -> None:
    """Test single key (F8): triggers on press and release, and is suppressed from OS."""
    mgr = Win32HotkeyManager("f8")

    down_count = 0
    up_count = 0
    mgr.on_down_callbacks.append(lambda: nonlocal_down())
    mgr.on_up_callbacks.append(lambda: nonlocal_up())

    def nonlocal_down() -> None:
        nonlocal down_count
        down_count += 1

    def nonlocal_up() -> None:
        nonlocal up_count
        up_count += 1

    data_f8 = SimpleNamespace(contents=SimpleNamespace(vkCode=0x77, flags=0))

    # 1. F8 down -> SUPPRESSED (False), hotkey DOWN emitted
    assert mgr._win32_filter(WM_KEYDOWN, data_f8) is False
    assert down_count == 1

    # 2. Auto-repeat F8 -> SUPPRESSED (False), no duplicate
    assert mgr._win32_filter(WM_KEYDOWN, data_f8) is False
    assert down_count == 1

    # 3. F8 up -> SUPPRESSED (False), hotkey UP emitted
    assert mgr._win32_filter(WM_KEYUP, data_f8) is False
    assert up_count == 1


def test_3_key_combo_ctrl_shift_v() -> None:
    """Test 3-key combo (Ctrl+Shift+V): triggers when all 3 held, suppresses V."""
    mgr = Win32HotkeyManager("ctrl+shift+v")

    down_count = 0
    up_count = 0
    mgr.on_down_callbacks.append(lambda: nonlocal_down())
    mgr.on_up_callbacks.append(lambda: nonlocal_up())

    def nonlocal_down() -> None:
        nonlocal down_count
        down_count += 1

    def nonlocal_up() -> None:
        nonlocal up_count
        up_count += 1

    data_ctrl = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_CONTROL, flags=0))
    data_shift = SimpleNamespace(contents=SimpleNamespace(vkCode=VK_SHIFT, flags=0))
    data_v = SimpleNamespace(contents=SimpleNamespace(vkCode=ord("V"), flags=0))

    # Press Ctrl then Shift -> pass through
    assert mgr._win32_filter(WM_KEYDOWN, data_ctrl) is True
    assert mgr._win32_filter(WM_KEYDOWN, data_shift) is True
    assert down_count == 0

    # Press V -> SUPPRESSED (False), emits DOWN
    assert mgr._win32_filter(WM_KEYDOWN, data_v) is False
    assert down_count == 1

    # Release V -> SUPPRESSED (False), emits UP
    assert mgr._win32_filter(WM_KEYUP, data_v) is False
    assert up_count == 1


def test_win32_filter_with_direct_struct_no_contents() -> None:
    """Test that _win32_filter works when data is directly the struct without .contents attribute (real pynput)."""
    mgr = Win32HotkeyManager("ctrl+shift")

    down_count = 0
    up_count = 0
    mgr.on_down_callbacks.append(lambda: nonlocal_down())
    mgr.on_up_callbacks.append(lambda: nonlocal_up())

    def nonlocal_down() -> None:
        nonlocal down_count
        down_count += 1

    def nonlocal_up() -> None:
        nonlocal up_count
        up_count += 1

    # Struct passed directly (no .contents attribute, matching pynput's _KBDLLHOOKSTRUCT)
    data_ctrl = SimpleNamespace(vkCode=VK_CONTROL, flags=0)
    data_shift = SimpleNamespace(vkCode=VK_SHIFT, flags=0)

    # Press Ctrl
    assert mgr._win32_filter(WM_KEYDOWN, data_ctrl) is True
    assert down_count == 0

    # Press Shift -> combo satisfied -> emits DOWN
    assert mgr._win32_filter(WM_KEYDOWN, data_shift) is True
    assert down_count == 1

    # Release Shift -> combo broken -> emits UP
    assert mgr._win32_filter(WM_KEYUP, data_shift) is True
    assert up_count == 1


def test_win32_filter_esc_in_toggle_mode() -> None:
    """Verify Esc suppresses and cancels when recording_active is True even if hotkey is not held."""
    mgr = Win32HotkeyManager("ctrl+shift")

    cancel_count = 0
    mgr.on_cancel_callbacks.append(lambda: nonlocal_cancel())

    def nonlocal_cancel() -> None:
        nonlocal cancel_count
        cancel_count += 1

    # In toggle mode, keys have been released, but recording is active
    mgr.set_recording_active(True)
    assert mgr._recording_active is True

    # Press Esc -> must be suppressed (return False) and emit cancel
    data_esc = SimpleNamespace(vkCode=VK_ESCAPE, flags=0)
    res = mgr._win32_filter(WM_KEYDOWN, data_esc)
    assert res is False
    assert cancel_count == 1
    assert mgr._recording_active is False


def test_win32_filter_true_suppression_with_listener() -> None:
    """Verify that when listener.suppress_event() raises SuppressException, it is re-raised."""
    from unittest.mock import MagicMock

    import pytest

    class SuppressException(Exception):
        pass

    mgr = Win32HotkeyManager("f8")
    mock_listener = MagicMock()
    mock_listener.suppress_event.side_effect = SuppressException()
    mgr._listener = mock_listener

    data_f8 = SimpleNamespace(vkCode=0x77, flags=0)  # F8 = 0x77

    with pytest.raises(Exception) as exc_info:
        mgr._win32_filter(WM_KEYDOWN, data_f8)
    assert exc_info.type.__name__ == "SuppressException"
    mock_listener.suppress_event.assert_called_once()


