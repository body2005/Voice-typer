"""Audio capture engine with continuous pre-roll ring buffer and silence gating.

Privacy Notice:
To prevent losing the first syllables of speech due to PortAudio stream start-up
latency, the audio stream remains open continuously while the hotkey is active.
Audio outside of active Push-to-Talk (PTT) recording is kept strictly in a tiny
~300 ms circular RAM buffer and immediately overwritten. It is NEVER written to
disk, persisted, or transmitted over any network.
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
import wave
from collections import deque
from typing import Any

import numpy as np
import scipy.signal
import sounddevice as sd

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000  # 16 kHz mono 16-bit PCM target
TARGET_CHANNELS = 1
PREROLL_DURATION_SEC = 0.30  # ~300 ms pre-roll
DEFAULT_MIN_DURATION_SEC = 0.30
DEFAULT_RMS_SILENCE_THRESHOLD = 0.004  # Below this RMS is considered silence
DEFAULT_PEAK_SILENCE_THRESHOLD = 0.012  # Below this peak is considered silence

EXCLUDED_INPUT_DEVICE_NAMES: set[str] = {
    "microsoft sound mapper - input",
    "primary sound capture driver",
}


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample 1D float32 audio array to target sample rate using polyphase filtering."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = scipy.signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Encode 1D float32 audio to in-memory 16-bit mono WAV bytes."""
    if len(audio) == 0:
        clipped = np.zeros(0, dtype=np.int16)
    else:
        clipped = np.clip(audio, -1.0, 1.0)
        clipped = (clipped * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())

    return buf.getvalue()


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 0.9,
    min_peak: float = 0.01,
    max_gain: float = 4.0,
) -> np.ndarray:
    """Apply gentle peak normalization to 1D float32 audio capped at max_gain (4x).

    Scales peak amplitude to `target_peak` (0.9) if peak exceeds `min_peak` (0.01),
    strictly capping amplification gain at `max_gain` (4.0x) so keyboard thumps or
    pre-roll background noise do not get amplified into hallucinations.
    """
    if len(audio) == 0:
        return audio.astype(np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak > min_peak:
        gain = min(target_peak / peak, max_gain)
        normalized = audio * gain
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    return audio.astype(np.float32)


def is_audio_silent(
    audio: np.ndarray,
    rms_threshold: float = DEFAULT_RMS_SILENCE_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_SILENCE_THRESHOLD,
) -> bool:
    """Energy-based VAD silence detector.

    Returns True if the clip is empty, completely silent, or below noise thresholds.
    """
    if len(audio) == 0:
        return True

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    logger.debug("VAD check: audio length=%d, RMS=%.5f (thresh=%.5f), Peak=%.5f (thresh=%.5f)", len(audio), rms, rms_threshold, peak, peak_threshold)

    return rms < rms_threshold and peak < peak_threshold


class AudioRecorder:
    """Captures microphone audio with continuous pre-roll ring buffer and in-memory WAV export."""

    def __init__(
        self,
        device: int | str | None = None,
        min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
        rms_threshold: float = DEFAULT_RMS_SILENCE_THRESHOLD,
        peak_threshold: float = DEFAULT_PEAK_SILENCE_THRESHOLD,
    ) -> None:
        self.device = device
        self.min_duration_sec = min_duration_sec
        self.rms_threshold = rms_threshold
        self.peak_threshold = peak_threshold

        self._lock = threading.Lock()
        self._is_recording = False
        self._record_start_time = 0.0

        # Pre-roll and active recording buffers
        self._preroll_chunks: deque[np.ndarray] = deque()
        self._preroll_samples = 0
        self._max_preroll_samples = int(TARGET_SAMPLE_RATE * PREROLL_DURATION_SEC)
        self._recorded_chunks: list[np.ndarray] = []

        # Real-time amplitude for 60 fps UI polling (no Qt signals in PortAudio callback)
        self._latest_rms = 0.0

        # Stream management
        self._stream: sd.InputStream | None = None
        self._device_sample_rate = TARGET_SAMPLE_RATE
        self._stream_running = False
        self._device_error_flag = False

    @staticmethod
    def get_input_devices() -> list[dict[str, Any]]:
        """List available microphone input devices, filtered to a single preferred Host API.

        Guarantees:
        1. Filters to a single Windows Host API (Windows WASAPI prioritized, MME fallback)
           to eliminate duplicate entries across MME, DirectSound, and WDM-KS.
        2. Filters out generic Windows mapper entries (e.g. 'Microsoft Sound Mapper - Input',
           'Primary Sound Capture Driver').
        3. Deduplicates remaining devices by name and strips whitespace.
        """
        devices: list[dict[str, Any]] = []
        try:
            raw_hostapis = sd.query_hostapis()
            hostapis = list(raw_hostapis) if isinstance(raw_hostapis, (list, tuple)) else [raw_hostapis]
            raw_devices = sd.query_devices()
            all_devices = list(raw_devices) if isinstance(raw_devices, (list, tuple)) else [raw_devices]

            # Find preferred host API index with at least one input device
            preferred_api_idx: int | None = None
            for pref in ("wasapi", "mme"):
                for i, h in enumerate(hostapis):
                    if pref in h.get("name", "").lower():
                        api_devices = h.get("devices", [])
                        has_input = False
                        for d_idx in api_devices:
                            if 0 <= d_idx < len(all_devices):
                                if all_devices[d_idx].get("max_input_channels", 0) > 0:
                                    has_input = True
                                    break
                        if has_input:
                            preferred_api_idx = i
                            break
                if preferred_api_idx is not None:
                    break

            seen_names: set[str] = set()

            for idx, dev in enumerate(all_devices):
                if dev.get("max_input_channels", 0) <= 0:
                    continue

                # Filter by preferred Host API if determined
                if preferred_api_idx is not None and dev.get("hostapi") != preferred_api_idx:
                    continue

                name = dev.get("name", f"Device {idx}").strip()
                name_lower = name.lower()

                # Exclude virtual mapper entries
                if (
                    name_lower in EXCLUDED_INPUT_DEVICE_NAMES
                    or "sound mapper" in name_lower
                    or "primary sound capture" in name_lower
                ):
                    continue

                # Deduplicate by name
                if name_lower in seen_names:
                    continue

                seen_names.add(name_lower)
                devices.append(
                    {
                        "index": idx,
                        "name": name,
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": dev.get("default_samplerate", 44100.0),
                        "hostapi": dev.get("hostapi", 0),
                    }
                )
        except Exception as err:
            logger.error("Failed to query audio devices: %s", err)

        return devices

    def set_device(self, device: int | str | None) -> None:
        """Switch audio input device and restart stream if active."""
        with self._lock:
            self.device = device
        if self._stream_running:
            self.stop_stream()
            self.start_stream()

    def start_stream(self) -> bool:
        """Open and start the continuous PortAudio input stream."""
        with self._lock:
            if self._stream_running and self._stream is not None:
                return True

        # Determine device sample rate
        sample_rate = TARGET_SAMPLE_RATE
        try:
            # Query device info
            dev_info = sd.query_devices(self.device, "input")
            native_sr = int(dev_info.get("default_samplerate", 44100))

            # Test if 16 kHz is supported directly
            try:
                sd.check_input_settings(
                    device=self.device,
                    channels=TARGET_CHANNELS,
                    samplerate=TARGET_SAMPLE_RATE,
                )
                sample_rate = TARGET_SAMPLE_RATE
            except Exception:
                sample_rate = native_sr
                logger.info(
                    "Device does not support 16 kHz directly. Using native %d Hz and resampling.",
                    sample_rate,
                )
        except Exception as err:
            logger.warning("Could not probe device settings (%s). Defaulting to 16 kHz.", err)
            sample_rate = TARGET_SAMPLE_RATE

        self._device_sample_rate = sample_rate
        self._max_preroll_samples = int(self._device_sample_rate * PREROLL_DURATION_SEC)
        block_size = max(256, int(self._device_sample_rate * 0.02))  # ~20ms blocks

        try:
            self._stream = sd.InputStream(
                device=self.device,
                channels=TARGET_CHANNELS,
                samplerate=self._device_sample_rate,
                dtype="float32",
                callback=self._audio_callback,
                finished_callback=self._stream_finished_callback,
                blocksize=block_size,
            )
            self._stream.start()
            with self._lock:
                self._stream_running = True
                self._device_error_flag = False
            logger.info(
                "Audio stream started (device=%s, rate=%d Hz, blocksize=%d)",
                self.device,
                self._device_sample_rate,
                block_size,
            )
            return True
        except Exception as err:
            logger.error("Failed to start audio input stream: %s", err)
            with self._lock:
                self._device_error_flag = True
                self._stream_running = False
                self._stream = None
            return False

    def _stream_finished_callback(self) -> None:
        """Invoked by PortAudio when the audio stream terminates unexpectedly."""
        logger.warning("Audio input stream terminated by PortAudio (device disconnected or stopped)")
        with self._lock:
            self._stream_running = False
            self._device_error_flag = True

    def stop_stream(self) -> None:
        """Stop and close the continuous input stream."""
        with self._lock:
            self._stream_running = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
                logger.info("Audio input stream stopped")
            except Exception as err:
                logger.warning("Error closing audio stream: %s", err)

    def is_stream_active(self) -> bool:
        """Return True if the background audio stream is currently running and healthy."""
        with self._lock:
            if not self._stream_running or self._stream is None:
                return False
            # Watchdog: ensure underlying PortAudio stream is still marked active
            try:
                if not self._stream.active:
                    self._stream_running = False
                    return False
            except Exception:
                self._stream_running = False
                return False
            return True

    def has_device_error(self) -> bool:
        """Return True if a device disconnection or error occurred."""
        with self._lock:
            return self._device_error_flag

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio stream callback.

        Executed on audio thread. Minimal overhead: captures raw mono chunks at
        native device sample rate with zero resampling in the callback loop.
        """
        if status:
            logger.debug("PortAudio callback status flag: %s", status)

        # Extract mono 1D array
        if indata.ndim > 1:
            chunk = indata[:, 0].copy()
        else:
            chunk = indata.flatten().copy()

        # Calculate fast RMS for UI meter on raw float32 chunk
        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0

        with self._lock:
            self._latest_rms = rms

            if self._is_recording:
                self._recorded_chunks.append(chunk)
            else:
                # Store in pre-roll ring buffer at native sample rate
                self._preroll_chunks.append(chunk)
                self._preroll_samples += len(chunk)
                while self._preroll_samples > self._max_preroll_samples and self._preroll_chunks:
                    dropped = self._preroll_chunks.popleft()
                    self._preroll_samples -= len(dropped)

    def get_latest_amplitude(self) -> float:
        """Thread-safe readout of current audio RMS amplitude (0.0 to ~1.0) for 60 fps UI polling."""
        with self._lock:
            return self._latest_rms

    def __del__(self) -> None:
        """Ensure PortAudio background stream is terminated when object is destroyed."""
        try:
            self.stop_stream()
        except Exception:
            pass

    def start_recording(self, auto_start_stream: bool = True) -> None:
        """Begin PTT recording. Transfers pre-roll buffer into active recording."""
        # Ensure stream is alive if auto_start_stream is True
        if auto_start_stream and not self.is_stream_active():
            self.start_stream()

        with self._lock:
            self._is_recording = True
            self._record_start_time = time.monotonic()
            # Seed active recording with pre-roll chunks
            self._recorded_chunks = list(self._preroll_chunks)
            self._preroll_chunks.clear()
            self._preroll_samples = 0
        logger.debug("Recording started (included pre-roll chunks)")

    def stop_recording(self) -> tuple[bytes, float, bool]:
        """Stop PTT recording and extract WAV audio.

        Returns:
            (wav_bytes, duration_sec, is_valid_speech)
            - wav_bytes: In-memory WAV format bytes
            - duration_sec: Duration in seconds
            - is_valid_speech: False if too short or silent (VAD rejected)
        """
        with self._lock:
            if not self._is_recording:
                return b"", 0.0, False

            self._is_recording = False
            duration = max(0.0, time.monotonic() - self._record_start_time)
            chunks = list(self._recorded_chunks)
            self._recorded_chunks = []
            self._preroll_chunks.clear()
            self._preroll_samples = 0
            native_rate = self._device_sample_rate

        if not chunks:
            logger.debug("Recording stopped: no audio chunks captured")
            return b"", duration, False

        # Concatenate raw chunks into single 1D array
        audio_data = np.concatenate(chunks)

        # Duration check
        if duration < self.min_duration_sec:
            logger.info("Recording too short (%.2fs < %.2fs); discarded", duration, self.min_duration_sec)
            return b"", duration, False

        # Resample ONCE here after concatenation if native rate differs from 16 kHz
        if native_rate != TARGET_SAMPLE_RATE:
            try:
                audio_data = resample_audio(audio_data, native_rate, TARGET_SAMPLE_RATE)
            except Exception as err:
                logger.error("Audio resampling error in stop_recording: %s", err)

        # Silence check (Energy VAD)
        if is_audio_silent(audio_data, self.rms_threshold, self.peak_threshold):
            logger.info("No speech detected (audio below energy thresholds); skipping API call")
            return b"", duration, False

        # Apply gentle peak normalization (capped at max 4x gain)
        audio_data = normalize_audio(audio_data, target_peak=0.9, min_peak=0.01, max_gain=4.0)

        wav_bytes = audio_to_wav_bytes(audio_data, TARGET_SAMPLE_RATE)
        logger.info("Captured %.2fs of speech audio (%d bytes WAV)", duration, len(wav_bytes))
        return wav_bytes, duration, True

    def cancel_recording(self) -> None:
        """Cancel current recording without processing or saving audio."""
        with self._lock:
            self._is_recording = False
            self._recorded_chunks.clear()
        logger.debug("Recording canceled by user")
