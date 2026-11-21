"""Unit tests for audio capture, resampling, VAD silence gating, and WAV encoding."""

import io
import time
import wave

import numpy as np

from voicetyper.audio_recorder import (
    AudioRecorder,
    audio_to_wav_bytes,
    is_audio_silent,
    normalize_audio,
    resample_audio,
)


def test_audio_to_wav_bytes_format() -> None:
    """Verify in-memory WAV encoding creates a valid 16 kHz 16-bit mono WAV."""
    sample_rate = 16000
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    wav_bytes = audio_to_wav_bytes(sine_wave, sample_rate)
    assert len(wav_bytes) > 0
    assert wav_bytes.startswith(b"RIFF")

    # Read back with wave module
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 16000
        frames = wf.readframes(wf.getnframes())
        assert len(frames) == len(sine_wave) * 2


def test_resample_audio() -> None:
    """Verify resampling from 48 kHz to 16 kHz."""
    orig_sr = 48000
    target_sr = 16000
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(orig_sr * duration_sec), endpoint=False, dtype=np.float32)
    orig_audio = np.sin(2 * np.pi * 300.0 * t)

    resampled = resample_audio(orig_audio, orig_sr, target_sr)
    assert len(resampled) == int(target_sr * duration_sec)
    assert resampled.dtype == np.float32


def test_vad_silence_detection() -> None:
    """Verify energy-based VAD correctly distinguishes silence from speech."""
    # 1. Complete silence
    zeros = np.zeros(16000, dtype=np.float32)
    assert is_audio_silent(zeros) is True

    # 2. Quiet background noise
    quiet_noise = np.random.normal(0, 0.001, 16000).astype(np.float32)
    assert is_audio_silent(quiet_noise) is True

    # 3. Speech-level signal (loud voice amplitude)
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    speech_signal = 0.4 * np.sin(2 * np.pi * 350.0 * t)
    assert is_audio_silent(speech_signal) is False


def test_audio_recorder_preroll_and_recording() -> None:
    """Verify pre-roll ring buffer is included when recording starts."""
    recorder = AudioRecorder(min_duration_sec=0.01)

    try:
        # Simulate callback delivering pre-roll blocks before PTT press
        dummy_block = np.ascontiguousarray(0.2 * np.ones((512, 1), dtype=np.float32))
        for _ in range(5):
            recorder._audio_callback(dummy_block, 512, None, 0)

        assert recorder._preroll_samples > 0

        # Start recording without spinning up physical hardware audio stream in tests
        recorder.start_recording(auto_start_stream=False)
        recorder._record_start_time = time.monotonic() - 1.0  # Simulate 1s duration
        assert recorder._is_recording is True
        assert recorder._preroll_samples == 0  # transferred to recording

        # Add more blocks during recording
        speech_block = np.ascontiguousarray(0.3 * np.ones((1024, 1), dtype=np.float32))
        for _ in range(5):
            recorder._audio_callback(speech_block, 1024, None, 0)

        # Stop recording
        wav_bytes, duration, is_speech = recorder.stop_recording()
        assert recorder._is_recording is False
        assert len(wav_bytes) > 0
        assert is_speech is True
    finally:
        recorder.stop_stream()


def test_audio_recorder_cancel() -> None:
    """Verify cancel clears recorded chunks."""
    recorder = AudioRecorder()
    recorder.start_recording()

    dummy_block = (0.3 * np.ones(1024, dtype=np.float32)).reshape(-1, 1)
    recorder._audio_callback(dummy_block, 1024, None, 0)

    recorder.cancel_recording()
    assert recorder._is_recording is False
    assert len(recorder._recorded_chunks) == 0


def test_get_input_devices_wasapi_priority_and_mapper_exclusion() -> None:
    """Verify input devices are filtered to Windows WASAPI, mappers are excluded, and names are deduplicated."""
    from unittest.mock import patch

    mock_hostapis = (
        {"name": "MME", "devices": [0, 1, 2]},
        {"name": "Windows DirectSound", "devices": [3, 4]},
        {"name": "Windows WASAPI", "devices": [5, 6, 7, 8]},
    )
    mock_devices = [
        # MME
        {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2},
        {"name": "Microphone (Realtek)", "hostapi": 0, "max_input_channels": 1},
        {"name": "Speakers (Realtek)", "hostapi": 0, "max_input_channels": 0},
        # DirectSound
        {"name": "Primary Sound Capture Driver", "hostapi": 1, "max_input_channels": 2},
        {"name": "Microphone (Realtek)", "hostapi": 1, "max_input_channels": 1},
        # WASAPI
        {"name": "Microphone (Realtek)", "hostapi": 2, "max_input_channels": 1},
        {"name": "Microphone (Realtek)", "hostapi": 2, "max_input_channels": 1},  # duplicate
        {"name": "USB Audio Mic", "hostapi": 2, "max_input_channels": 1},
        {"name": "Speakers (Realtek)", "hostapi": 2, "max_input_channels": 0},  # output only
    ]

    with patch("sounddevice.query_hostapis", return_value=mock_hostapis), \
         patch("sounddevice.query_devices", return_value=mock_devices):
        devs = AudioRecorder.get_input_devices()

        # Only devices from WASAPI should be present
        assert len(devs) == 2
        names = [d["name"] for d in devs]
        assert "Microphone (Realtek)" in names
        assert "USB Audio Mic" in names
        assert "Microsoft Sound Mapper - Input" not in names
        assert "Primary Sound Capture Driver" not in names
        assert "Speakers (Realtek)" not in names
        # Device indices should be from WASAPI (5 and 7)
        indices = [d["index"] for d in devs]
        assert 5 in indices
        assert 7 in indices


def test_get_input_devices_mme_fallback() -> None:
    """Verify fallback to MME when WASAPI is unavailable or has no input devices."""
    from unittest.mock import patch

    mock_hostapis = (
        {"name": "MME", "devices": [0, 1]},
        {"name": "Windows DirectSound", "devices": [2]},
    )
    mock_devices = [
        {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2},
        {"name": "Microphone (MME Mic)", "hostapi": 0, "max_input_channels": 1},
        {"name": "Primary Sound Capture Driver", "hostapi": 1, "max_input_channels": 2},
    ]

    with patch("sounddevice.query_hostapis", return_value=mock_hostapis), \
         patch("sounddevice.query_devices", return_value=mock_devices):
        devs = AudioRecorder.get_input_devices()

        assert len(devs) == 1
        assert devs[0]["name"] == "Microphone (MME Mic)"
        assert devs[0]["index"] == 1


def test_audio_normalization() -> None:
    """Verify gentle peak normalization scales quiet audio to 0.9 and preserves silence."""
    # 1. Quiet audio with peak 0.3 boosted by 3.0x (<= max_gain 4.0x) to peak 0.9
    quiet = np.array([-0.15, 0.3, -0.24, 0.05], dtype=np.float32)
    normalized = normalize_audio(quiet, target_peak=0.9, min_peak=0.01)
    assert np.isclose(float(np.max(np.abs(normalized))), 0.9, atol=1e-5)
    assert normalized.dtype == np.float32

    # 2. Pure silence/micro-noise below min_peak (0.01) is not amplified
    silence = np.array([-0.002, 0.005, -0.001], dtype=np.float32)
    norm_silence = normalize_audio(silence, target_peak=0.9, min_peak=0.01)
    assert np.allclose(silence, norm_silence)

    # 3. Empty audio
    empty = np.zeros(0, dtype=np.float32)
    assert len(normalize_audio(empty)) == 0

    # 4. Loud audio with peak > 0.9 is gently scaled to 0.9
    loud = np.array([-0.98, 0.5, 0.2], dtype=np.float32)
    norm_loud = normalize_audio(loud, target_peak=0.9, min_peak=0.01)
    assert np.isclose(float(np.max(np.abs(norm_loud))), 0.9, atol=1e-5)

    # 5. Very quiet audio with peak 0.05 (which would otherwise need 18x gain) is capped at max 4x gain (0.20)
    tiny = np.array([-0.02, 0.05, -0.03], dtype=np.float32)
    norm_tiny = normalize_audio(tiny, target_peak=0.9, min_peak=0.01, max_gain=4.0)
    assert np.isclose(float(np.max(np.abs(norm_tiny))), 0.20, atol=1e-5)

