"""Core application controller and thread-safe state machine.

Manages transitions between:
IDLE -> RECORDING -> PROCESSING -> INJECTING -> ERROR -> IDLE.
Ensures zero race conditions, strict state validation, background worker isolation,
and support for both Push-to-Talk (press & hold) and optional Toggle modes.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from enum import Enum
from typing import Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal

from voicetyper.audio_recorder import AudioRecorder
from voicetyper.config import Config, ConfigManager
from voicetyper.transcriber import Transcriber, TranscriptionResult

logger = logging.getLogger(__name__)


class AppState(str, Enum):
    """VoiceTyper state machine states."""

    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    INJECTING = "INJECTING"
    ERROR = "ERROR"


class VoiceTyperController(QObject):
    """Coordinates hotkeys, audio capture, transcription, overlay, and text injection."""

    # Qt Signals for UI decoupling
    state_changed = pyqtSignal(str, str)  # type: ignore[call-arg] # (old_state, new_state)
    overlay_show_recording = pyqtSignal()  # type: ignore[call-arg]
    overlay_show_processing = pyqtSignal(str)  # type: ignore[call-arg] # (message)
    overlay_show_error = pyqtSignal(str)  # type: ignore[call-arg] # error text
    overlay_hide = pyqtSignal()  # type: ignore[call-arg]
    play_beep_requested = pyqtSignal(str)  # type: ignore[call-arg] # "start" or "stop"
    text_injection_requested = pyqtSignal(str)  # type: ignore[call-arg] # text to inject on GUI thread

    def __init__(
        self,
        config: Config,
        config_manager: ConfigManager | None = None,
        audio_recorder: AudioRecorder | None = None,
        transcriber: Transcriber | None = None,
        text_injector: Any = None,
        hotkey_manager: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.config = config
        self.config_manager = config_manager
        self.hotkey_manager = hotkey_manager
        self.recorder = audio_recorder or AudioRecorder(
            device=config.mic_device,
            min_duration_sec=config.min_duration_sec,
        )
        self.transcriber = transcriber or Transcriber(
            api_key=config.api_key,
            engine=config.transcription_engine,
            whisper_model=config.whisper_model,
            whisper_device=config.whisper_device,
            whisper_compute_type=config.whisper_compute_type,
        )
        self.injector = text_injector

        self._state = AppState.IDLE
        self._state_lock = threading.Lock()
        self._record_start_time = 0.0
        self._target_hwnd: int | None = None
        self._job_id: int = 0
        self._toggle_keys_released: bool = True

        # Background worker management
        self._active_worker_thread: threading.Thread | None = None

        # Listeners for headless unit testing without Qt event loop
        self._state_listeners: list[Callable[[AppState, AppState], None]] = []

        # Automatic max duration watchdog
        self._max_duration_timer: Any = None

        # Auto-reset error timer
        self._error_reset_timer: threading.Timer | None = None

    @property
    def state(self) -> AppState:
        """Current state of the controller."""
        with self._state_lock:
            return self._state

    def add_state_listener(self, listener: Callable[[AppState, AppState], None]) -> None:
        """Register a callback for state changes (useful for unit tests)."""
        self._state_listeners.append(listener)

    def _transition_to(self, new_state: AppState) -> bool:
        """Thread-safe state transition with validation.

        Returns True if transition succeeded, False if rejected.
        """
        with self._state_lock:
            old_state = self._state
            if old_state == new_state:
                return True

            # State transition rules
            valid = False
            if old_state == AppState.IDLE and new_state in (AppState.RECORDING, AppState.ERROR):
                valid = True
            elif old_state == AppState.RECORDING and new_state in (AppState.PROCESSING, AppState.IDLE, AppState.ERROR):
                valid = True
            elif old_state == AppState.PROCESSING and new_state in (AppState.INJECTING, AppState.ERROR, AppState.IDLE):
                valid = True
            elif old_state == AppState.INJECTING and new_state in (AppState.IDLE, AppState.ERROR):
                valid = True
            elif old_state == AppState.ERROR and new_state in (AppState.IDLE, AppState.RECORDING):
                valid = True

            if not valid:
                logger.warning("Invalid state transition rejected: %s -> %s", old_state.value, new_state.value)
                return False

            self._state = new_state
            logger.info("State transition: %s -> %s", old_state.value, new_state.value)

        # Emit Qt signal
        self.state_changed.emit(old_state.value, new_state.value)

        # Keep hotkey manager informed of recording state for Esc suppression/cancel
        if self.hotkey_manager is not None and hasattr(self.hotkey_manager, "set_recording_active"):
            try:
                self.hotkey_manager.set_recording_active(new_state == AppState.RECORDING)
            except Exception as h_err:
                logger.debug("Error setting hotkey recording active: %s", h_err)

        # Notify pure Python listeners
        for listener in self._state_listeners:
            try:
                listener(old_state, new_state)
            except Exception as err:
                logger.error("Error in state listener: %s", err)

        return True

    def on_hotkey_down(self) -> None:
        """Invoked when the PTT hotkey is pressed."""
        with self._state_lock:
            current = self._state

        # Handle Toggle mode (press once to start, press again to stop)
        if self.config.toggle_mode and current == AppState.RECORDING:
            # Check key release latch: keys must have been released after start before stop can trigger
            if not self._toggle_keys_released:
                logger.debug("Toggle stop ignored: hotkey has not been released since start")
                return

            now = time.monotonic()
            # Debounce cooldown: require at least 250ms of recording before toggle-stop
            if now - self._record_start_time < 0.25:
                logger.debug("Toggle stop ignored: within 250ms debounce cooldown")
                return

            logger.info("Toggle mode: second hotkey press detected; stopping recording")
            self._stop_and_process()
            return

        if current == AppState.ERROR:
            # Allow immediate restart from error state
            if self._error_reset_timer:
                self._error_reset_timer.cancel()
                self._error_reset_timer = None
            if not self._transition_to(AppState.RECORDING):
                return
        elif current != AppState.IDLE:
            logger.debug("Hotkey press ignored: state is %s", current.value)
            return
        else:
            if not self._transition_to(AppState.RECORDING):
                return

        # Check for API key presence ONLY if using Gemini engine
        if self.config.transcription_engine == "gemini" and not self.config.api_key:
            logger.warning("No API key configured for Gemini")
            self._show_error("Gemini API key not configured")
            return

        with self._state_lock:
            self._job_id += 1
            self._record_start_time = time.monotonic()
            self._toggle_keys_released = False

        # Capture target window handle for focus restoration before paste
        if sys.platform == "win32":
            try:
                import ctypes

                self._target_hwnd = int(ctypes.windll.user32.GetForegroundWindow())
            except Exception:
                self._target_hwnd = None

        # Start audio recording
        self.recorder.start_recording()

        # Play start beep if enabled
        if self.config.start_stop_beeps:
            self.play_beep_requested.emit("start")

        # Show overlay
        self.overlay_show_recording.emit()

        # Schedule hard cap max recording duration using QTimer on GUI thread
        from PyQt6.QtCore import QTimer

        if self._max_duration_timer:
            try:
                self._max_duration_timer.stop()
            except Exception:
                pass
            self._max_duration_timer = None

        max_dur_ms = int(max(5.0, self.config.max_duration_sec) * 1000)
        self._max_duration_timer = QTimer(self)
        self._max_duration_timer.setSingleShot(True)
        self._max_duration_timer.timeout.connect(self._on_max_duration_reached)
        self._max_duration_timer.start(max_dur_ms)

    def on_hotkey_up(self) -> None:
        """Invoked when the PTT hotkey is released."""
        # In Toggle mode, releasing the keys must NOT stop recording!
        if self.config.toggle_mode:
            with self._state_lock:
                self._toggle_keys_released = True
            return

        if self.state != AppState.RECORDING:
            return

        self._stop_and_process()

    def _stop_and_process(self) -> None:
        """Stop audio capture and trigger transcription if speech detected.

        Atomic transition from RECORDING -> PROCESSING under state lock prevents race conditions.
        """
        with self._state_lock:
            if self._state != AppState.RECORDING:
                return
            old_state = self._state
            self._state = AppState.PROCESSING
            current_job = self._job_id

        if self._max_duration_timer:
            try:
                self._max_duration_timer.stop()
            except Exception:
                pass
            self._max_duration_timer = None

        # Emit state change signal
        self.state_changed.emit(old_state.value, AppState.PROCESSING.value)
        if self.hotkey_manager is not None and hasattr(self.hotkey_manager, "set_recording_active"):
            try:
                self.hotkey_manager.set_recording_active(False)
            except Exception:
                pass
        for listener in self._state_listeners:
            try:
                listener(old_state, AppState.PROCESSING)
            except Exception as err:
                logger.error("Error in state listener: %s", err)

        # Stop audio capture
        wav_bytes, duration, is_speech = self.recorder.stop_recording()

        # Play stop beep if enabled (strictly once when recording stops)
        if self.config.start_stop_beeps:
            self.play_beep_requested.emit("stop")

        # Check min duration (ignore presses < 0.3s)
        if duration < self.config.min_duration_sec:
            logger.debug("Recording too short (%.2fs < %.2fs); ignoring", duration, self.config.min_duration_sec)
            self._transition_to(AppState.IDLE)
            self.overlay_hide.emit()
            return

        # Check energy VAD silence gate
        if not is_speech:
            logger.info("VAD detected silence or empty clip")
            self._show_error("No speech detected")
            return

        is_loaded = getattr(self.transcriber, "is_model_loaded", True)
        if self.config.transcription_engine == "local" and not is_loaded:
            info_text, _, _ = getattr(self.transcriber, "download_info", ("", None, ""))
            self.overlay_show_processing.emit(info_text or "Downloading model...")
        else:
            self.overlay_show_processing.emit("Processing speech...")

        # Launch transcription in background thread with current job_id
        self._start_transcription_worker(wav_bytes, current_job)

    def on_cancel(self) -> None:
        """Invoked when user presses Esc to abort recording/processing or model download."""
        # Check if model download is active
        is_dl = getattr(self.transcriber, "is_downloading", False) is True
        if is_dl:
            logger.info("Esc pressed; cancelling model download")
            try:
                self.transcriber.cancel_download()
            except Exception as err:
                logger.error("Failed to cancel download: %s", err)
            self._transition_to(AppState.IDLE)
            self.overlay_hide.emit()
            return

        current = self.state
        if current not in (AppState.RECORDING, AppState.PROCESSING):
            return

        logger.info("Recording canceled by user (Esc)")
        with self._state_lock:
            self._job_id += 1
            self._toggle_keys_released = True

        if self._max_duration_timer:
            try:
                self._max_duration_timer.stop()
            except Exception:
                pass
            self._max_duration_timer = None

        self.recorder.cancel_recording()
        self._transition_to(AppState.IDLE)
        self.overlay_hide.emit()

    def _on_max_duration_reached(self) -> None:
        """Callback when hard cap recording duration is reached."""
        logger.warning("Max recording duration reached (%.1fs). Auto-stopping.", self.config.max_duration_sec)
        if self.state == AppState.RECORDING:
            self._stop_and_process()

    def _start_transcription_worker(self, wav_bytes: bytes, job_id: int) -> None:
        """Execute transcription on a dedicated background thread."""
        def is_job_canceled() -> bool:
            with self._state_lock:
                return self._job_id != job_id

        def worker() -> None:
            try:
                res = self.transcriber.transcribe(
                    audio_wav_bytes=wav_bytes,
                    language_hint=self.config.language_hint,
                    custom_vocabulary=self.config.custom_vocabulary,
                    cancel_check=is_job_canceled,
                )
            except Exception as err:
                logger.error("Unexpected worker exception during transcription: %s", err)
                res = TranscriptionResult(
                    success=False,
                    error_message="Unexpected error",
                    error_type="unknown",
                )

            # Check if user canceled or started a new recording while worker was running
            with self._state_lock:
                if self._job_id != job_id:
                    logger.info("Worker completed for job %d, but superseded by job %d", job_id, self._job_id)
                    return

            self._on_transcription_completed(res, job_id)

        self._active_worker_thread = threading.Thread(
            target=worker,
            name="TranscriptionWorker",
            daemon=True,
        )
        self._active_worker_thread.start()

    def _on_transcription_completed(self, result: TranscriptionResult, job_id: int) -> None:
        """Handle completion of transcription worker."""
        with self._state_lock:
            if self._job_id != job_id or self._state != AppState.PROCESSING:
                logger.debug(
                    "Discarding transcription result; job %d (current %d) state %s",
                    job_id,
                    self._job_id,
                    self._state.value,
                )
                return

        if not result.success:
            if result.error_type == "canceled":
                self._transition_to(AppState.IDLE)
                self.overlay_hide.emit()
                return
            logger.warning("Transcription failed: %s (%s)", result.error_message, result.error_type)
            self._show_error(result.error_message)
            return

        if not result.text:
            logger.info("Transcription returned empty text (no intelligible speech)")
            self._show_error("No speech detected")
            return

        # Transition to INJECTING
        if not self._transition_to(AppState.INJECTING):
            return

        logger.info("Injecting text (length=%d chars)", len(result.text))

        # Format English casing according to user preferences
        from voicetyper.transcriber import format_english_casing

        text_to_inject = format_english_casing(result.text, mode=self.config.capitalize_english)

        pasted = True
        if self.injector:
            try:
                from PyQt6.QtCore import QTimer

                def on_toast(msg: str) -> None:
                    nonlocal pasted
                    pasted = False
                    self.overlay_show_processing.emit(msg)
                    QTimer.singleShot(2200, self._on_toast_timeout)

                self.injector.inject_text(
                    text_to_inject,
                    fallback_typing=self.config.paste_fallback_typing,
                    restore_clipboard=self.config.restore_clipboard,
                    target_hwnd=self._target_hwnd,
                    on_copied_toast=on_toast,
                )
            except Exception as err:
                logger.error("Text injection failed: %s", err)
        else:
            self.text_injection_requested.emit(text_to_inject)

        if pasted:
            self.overlay_hide.emit()
        self._transition_to(AppState.IDLE)

    def _on_toast_timeout(self) -> None:
        """Hide overlay after toast timeout if controller has returned to IDLE."""
        if self.state == AppState.IDLE:
            self.overlay_hide.emit()

    def _show_error(self, message: str, auto_hide_sec: float = 1.8) -> None:
        """Display an error message in overlay toast and return to IDLE."""
        self._transition_to(AppState.ERROR)
        self.overlay_show_error.emit(message)

        if self._error_reset_timer:
            self._error_reset_timer.cancel()

        def reset_to_idle() -> None:
            if self.state == AppState.ERROR:
                self.overlay_hide.emit()
                self._transition_to(AppState.IDLE)

        self._error_reset_timer = threading.Timer(auto_hide_sec, reset_to_idle)
        self._error_reset_timer.daemon = True
        self._error_reset_timer.start()

    def update_config(self, new_config: Config) -> None:
        """Update runtime configuration dynamically."""
        old_device = self.config.mic_device
        self.config = new_config
        self.transcriber.set_api_key(new_config.api_key)
        self.transcriber.set_engine(new_config.transcription_engine)
        self.transcriber.set_whisper_model(new_config.whisper_model)
        self.recorder.min_duration_sec = new_config.min_duration_sec
        if old_device != new_config.mic_device:
            self.recorder.set_device(new_config.mic_device)

    def shutdown(self) -> None:
        """Clean shutdown of timers, streams, and threads on exit."""
        logger.info("Shutting down controller...")
        with self._state_lock:
            self._job_id += 1

        if self._max_duration_timer:
            try:
                self._max_duration_timer.stop()
            except Exception:
                pass
            self._max_duration_timer = None

        if self._error_reset_timer:
            self._error_reset_timer.cancel()
            self._error_reset_timer = None

        self.recorder.stop_stream()
        self._transition_to(AppState.IDLE)
