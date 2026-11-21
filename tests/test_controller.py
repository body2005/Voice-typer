"""Unit tests for the controller state machine and coordination."""

from unittest.mock import MagicMock

from voicetyper.audio_recorder import AudioRecorder
from voicetyper.config import Config
from voicetyper.controller import AppState, VoiceTyperController
from voicetyper.transcriber import Transcriber, TranscriptionResult


def test_controller_initial_state() -> None:
    """Verify controller starts in IDLE."""
    cfg = Config(api_key="test_key")
    ctrl = VoiceTyperController(config=cfg)
    assert ctrl.state == AppState.IDLE


def test_controller_press_and_short_release() -> None:
    """Verify release under min_duration_sec discards audio and returns to IDLE."""
    cfg = Config(api_key="test_key", min_duration_sec=0.3)
    mock_recorder = MagicMock(spec=AudioRecorder)
    # Return 0.1s (too short)
    mock_recorder.stop_recording.return_value = (b"", 0.1, False)

    ctrl = VoiceTyperController(config=cfg, audio_recorder=mock_recorder)
    state_history: list[tuple[AppState, AppState]] = []
    ctrl.add_state_listener(lambda old_s, new_s: state_history.append((old_s, new_s)))

    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING

    ctrl.on_hotkey_up()
    assert ctrl.state == AppState.IDLE
    assert (AppState.IDLE, AppState.RECORDING) in state_history
    assert (AppState.RECORDING, AppState.PROCESSING) in state_history
    assert (AppState.PROCESSING, AppState.IDLE) in state_history


def test_controller_cancel_recording() -> None:
    """Verify Esc cancels recording and returns to IDLE without transcribing."""
    cfg = Config(api_key="test_key")
    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_transcriber = MagicMock(spec=Transcriber)

    ctrl = VoiceTyperController(config=cfg, audio_recorder=mock_recorder, transcriber=mock_transcriber)
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING

    ctrl.on_cancel()
    assert ctrl.state == AppState.IDLE
    mock_recorder.cancel_recording.assert_called_once()
    mock_transcriber.transcribe.assert_not_called()


def test_controller_full_workflow_success() -> None:
    """Verify IDLE -> RECORDING -> PROCESSING -> INJECTING -> IDLE flow."""
    cfg = Config(api_key="test_key", min_duration_sec=0.3)
    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.stop_recording.return_value = (b"RIFFaudio", 1.5, True)

    mock_transcriber = MagicMock(spec=Transcriber)
    mock_transcriber.transcribe.return_value = TranscriptionResult(
        success=True,
        text="Hello world transcription",
    )

    mock_injector = MagicMock()

    ctrl = VoiceTyperController(
        config=cfg,
        audio_recorder=mock_recorder,
        transcriber=mock_transcriber,
        text_injector=mock_injector,
    )

    state_history: list[tuple[AppState, AppState]] = []
    ctrl.add_state_listener(lambda old_s, new_s: state_history.append((old_s, new_s)))

    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING

    ctrl.on_hotkey_up()
    # Wait for the background worker thread to finish
    if ctrl._active_worker_thread:
        ctrl._active_worker_thread.join(timeout=2.0)

    assert ctrl.state == AppState.IDLE
    mock_injector.inject_text.assert_called_once()
    call_args, call_kwargs = mock_injector.inject_text.call_args
    assert call_args[0] == "Hello World Transcription"
    assert call_kwargs.get("fallback_typing") is False
    assert call_kwargs.get("restore_clipboard") is False
    assert call_kwargs.get("target_hwnd") is None

    states_visited = [new_s for _, new_s in state_history]
    assert AppState.RECORDING in states_visited
    assert AppState.PROCESSING in states_visited
    assert AppState.INJECTING in states_visited
    assert AppState.IDLE in states_visited


def test_controller_silence_detected() -> None:
    """Verify silence / empty clip transitions to ERROR then auto-resets to IDLE."""
    cfg = Config(api_key="test_key", min_duration_sec=0.3)
    mock_recorder = MagicMock(spec=AudioRecorder)
    # Stop recording returns duration 1.0s but is_speech=False
    mock_recorder.stop_recording.return_value = (b"", 1.0, False)

    ctrl = VoiceTyperController(config=cfg, audio_recorder=mock_recorder)
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING

    ctrl.on_hotkey_up()
    assert ctrl.state == AppState.ERROR

    # Check reset timer
    if ctrl._error_reset_timer:
        ctrl._error_reset_timer.join(timeout=3.0)
    assert ctrl.state == AppState.IDLE


def test_controller_toggle_mode() -> None:
    """Verify toggle mode: press starts, releasing stays recording, second press stops."""
    import time

    cfg = Config(api_key="test_key", toggle_mode=True, min_duration_sec=0.3)
    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.stop_recording.return_value = (b"RIFFdata", 1.0, True)

    mock_transcriber = MagicMock(spec=Transcriber)
    mock_transcriber.transcribe.return_value = TranscriptionResult(success=True, text="Toggle test")

    ctrl = VoiceTyperController(config=cfg, audio_recorder=mock_recorder, transcriber=mock_transcriber)

    # 1. First press starts recording
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING

    # 2. Releasing the hotkey keys MUST NOT stop recording in toggle mode
    ctrl.on_hotkey_up()
    assert ctrl.state == AppState.RECORDING
    mock_recorder.stop_recording.assert_not_called()

    # 3. Fast repeat/bounce within 250ms cooldown is ignored
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING
    mock_recorder.stop_recording.assert_not_called()

    # 4. Advance time past 250ms cooldown and press again -> stops and processes
    ctrl._record_start_time = time.monotonic() - 0.5
    ctrl.on_hotkey_down()
    if ctrl._active_worker_thread:
        ctrl._active_worker_thread.join(timeout=2.0)

    assert ctrl.state == AppState.IDLE
    mock_recorder.stop_recording.assert_called_once()


def test_controller_toggle_mode_esc_cancels() -> None:
    """Verify pressing Esc while recording in Toggle mode cancels cleanly."""
    cfg = Config(api_key="test_key", toggle_mode=True, min_duration_sec=0.3)
    mock_recorder = MagicMock(spec=AudioRecorder)

    ctrl = VoiceTyperController(config=cfg, audio_recorder=mock_recorder)

    # First press starts
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING

    # Releasing keys stays recording
    ctrl.on_hotkey_up()
    assert ctrl.state == AppState.RECORDING

    # Esc cancels
    ctrl.on_cancel()
    assert ctrl.state == AppState.IDLE
    mock_recorder.cancel_recording.assert_called_once()


def test_controller_live_toggle_mode_switching() -> None:
    """Verify switching toggle_mode live updates behavior without restart."""
    cfg = Config(api_key="test_key", toggle_mode=False, min_duration_sec=0.3)
    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.stop_recording.return_value = (b"RIFFdata", 1.0, True)

    ctrl = VoiceTyperController(config=cfg, audio_recorder=mock_recorder)

    # PTT mode: release stops recording
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING
    ctrl.on_hotkey_up()
    assert ctrl.state != AppState.RECORDING

    # Switch to toggle mode live
    new_cfg = Config(api_key="test_key", toggle_mode=True, min_duration_sec=0.3)
    ctrl.update_config(new_cfg)
    assert ctrl.config.toggle_mode is True

    # Now in toggle mode: release does NOT stop recording
    ctrl._transition_to(AppState.IDLE)
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING
    ctrl.on_hotkey_up()
    assert ctrl.state == AppState.RECORDING
    ctrl.on_cancel()
    assert ctrl.state == AppState.IDLE


def test_controller_error_to_recording_transition() -> None:
    """Verify pressing hotkey while in ERROR immediately transitions to RECORDING."""
    cfg = Config(api_key="test_key")
    mock_recorder = MagicMock(spec=AudioRecorder)
    ctrl = VoiceTyperController(config=cfg, audio_recorder=mock_recorder)

    # Put controller in ERROR state
    ctrl._show_error("Test error")
    assert ctrl.state == AppState.ERROR

    # Press hotkey -> transitions to RECORDING immediately
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING
    ctrl.on_cancel()
    assert ctrl.state == AppState.IDLE


def test_controller_job_id_cancel_safety() -> None:
    """Verify old in-flight worker cannot inject stale text into a new recording."""
    import time

    cfg = Config(api_key="test_key", min_duration_sec=0.1)
    mock_recorder = MagicMock(spec=AudioRecorder)
    mock_recorder.stop_recording.return_value = (b"RIFFdata", 1.0, True)

    def slow_transcribe(*args, **kwargs):
        time.sleep(0.3)
        return TranscriptionResult(success=True, text="Stale Text")

    mock_transcriber = MagicMock(spec=Transcriber)
    mock_transcriber.transcribe.side_effect = slow_transcribe
    mock_injector = MagicMock()

    ctrl = VoiceTyperController(
        config=cfg,
        audio_recorder=mock_recorder,
        transcriber=mock_transcriber,
        text_injector=mock_injector,
    )

    # 1. Start recording and stop -> starts worker for job 1
    ctrl.on_hotkey_down()
    first_job = ctrl._job_id
    ctrl.on_hotkey_up()
    assert ctrl.state == AppState.PROCESSING

    # 2. Cancel on Esc -> increments job_id
    ctrl.on_cancel()
    assert ctrl.state == AppState.IDLE
    assert ctrl._job_id > first_job

    # 3. Wait for background worker to complete
    if ctrl._active_worker_thread:
        ctrl._active_worker_thread.join(timeout=1.0)

    # 4. Injector MUST NOT have been called with stale text!
    mock_injector.inject_text.assert_not_called()


def test_controller_toggle_mode_key_latch() -> None:
    """Verify key repeat before releasing keys does not trigger stop."""
    cfg = Config(api_key="test_key", toggle_mode=True, min_duration_sec=0.1)
    mock_recorder = MagicMock(spec=AudioRecorder)
    ctrl = VoiceTyperController(config=cfg, audio_recorder=mock_recorder)

    # 1. First press starts
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING
    assert ctrl._toggle_keys_released is False

    # 2. Key repeat events while still holding down must NOT stop recording
    ctrl.on_hotkey_down()
    ctrl.on_hotkey_down()
    assert ctrl.state == AppState.RECORDING
    mock_recorder.stop_recording.assert_not_called()

    # 3. Release keys -> latch cleared
    ctrl.on_hotkey_up()
    assert ctrl.state == AppState.RECORDING
    assert ctrl._toggle_keys_released is True

