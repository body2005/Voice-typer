# VoiceTyper

**Production-grade, lightweight Windows 10/11 Push-to-Talk (PTT) Voice-to-Text desktop utility in Python 3.11+.**

VoiceTyper allows you to hold a global hotkey (default: `Ctrl + Shift`), speak in **Arabic (Egyptian & Modern Standard), English, or mixed code-switching**, and have your speech instantly transcribed **100% locally on your GPU/CPU** via **faster-whisper** (or optionally via Google Gemini 2.5 Flash) and typed or pasted directly into any active text field across Windows (VS Code, Chrome, WhatsApp Desktop/Web, Microsoft Word, Slack, Terminal, etc.).

---

## Key Features

- **100% Offline Local GPU/CPU Engine (`faster-whisper` + CTranslate2):**
  - **Zero cloud API requirement:** transcribes speech locally with near-zero latency (~0.5–1.0s).
  - **Automatic Hardware Detection:** auto-detects NVIDIA CUDA GPU with `float16` or `int8_float16`; gracefully falls back to CPU with `int8`.
  - **Model Flexibility:** default model `large-v3-turbo` (recommended for Arabic + English code-switching), with choices for `medium`, `small`, and `base`.
  - **Background Startup Loading:** loads the model once in a background worker at app launch so recording begins instantaneously.
- **Flexible 1-, 2-, or 3-Key Global Hotkey Support:**
  - New default hotkey: **`Ctrl + Shift`**.
  - Supports **Single Keys** (`F8`, `F9`, `Right Alt / AltGr`, `CapsLock`, `Insert`), **Modifier-Only Combos** (`Ctrl + Shift`, `Ctrl + Alt`), and **2- or 3-Key Combos** (`Ctrl + Shift + V`, `Alt + Z`, `Ctrl + Space`).
  - **Push-to-Talk Logic:** start recording as soon as all keys in the combination are held; stop recording as soon as any key is released.
  - **Non-Modifier Suppression:** keys like Space, letters, or F-keys are intercepted and suppressed so they never type into focused applications.
- **Interactive Keyboard Hotkey Recorder:**
  - In Settings, click **Record Hotkey**, press any 1, 2, or 3 keys simultaneously, and save immediately to `config.json` with live dynamic reload.
- **Quick Settings Gear Icon (⚙) on Floating Overlay:**
  - Clean circular gear button on the right edge of the glassmorphic pill overlay.
  - Click to open Settings directly without interfering with drag-to-move repositioning.
- **Glassmorphic Floating Pill Overlay:**
  - Frameless, non-activating (`WA_ShowWithoutActivating`, `WS_EX_NOACTIVATE`) - **never steals keyboard focus** from editors or browsers.
  - Left-click draggable with persistent coordinates saved to `config.json`.
  - 60 FPS real-time audio visualizer with equalizer bars, edge-tapered dual sine waves, and animated flowing wave orbs.
  - Smart opacity fade with 60-second inactivity auto-hide.
- **Zero Startup Latency (Pre-roll Ring Buffer):**
  - Keeps microphone stream open in a tiny ~300 ms circular RAM buffer so the first syllables are never lost.
- **Complete Clipboard Preservation & Win+V Privacy Guard:**
  - Backs up all clipboard formats (images, files, rich text, Unicode), pastes text, and restores original clipboard contents.
  - Injected text is marked with privacy flags to exclude it from Windows Clipboard History (`Win + V`) and cloud sync.

---

## Architecture & Project Structure

```
voice2text/
├── main.py                     # Root launcher
├── voicetyper/
│   ├── __init__.py
│   ├── main.py                 # System tray lifecycle & signal wiring
│   ├── controller.py           # State machine (Idle -> Recording -> Processing -> Injecting -> Error -> Idle)
│   ├── audio_recorder.py       # Sounddevice PortAudio capture, pre-roll ring buffer, resampling, energy VAD
│   ├── transcriber.py          # Unified transcription engine (Local faster-whisper + Google Gemini)
│   ├── hotkey_manager.py       # Win32 low-level keyboard hook supporting 1-, 2-, 3-key combos & suppression
│   ├── text_injector.py        # Win32 clipboard backup/restore, privacy flags, modifier wait, SendInput Ctrl+V
│   ├── config.py               # Settings model, JSON persistence, keyring credential vault, startup toggle
│   ├── paths.py                # sys._MEIPASS resource resolution, APPDATA paths, rotating file logging
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── overlay.py          # Non-activating floating pill with 60 FPS visualizer & gear icon (⚙)
│   │   └── settings_dialog.py  # Settings window with interactive Hotkey Recorder & model selector
│   └── assets/
│       ├── icon.ico            # Windows application icon
│       ├── icon.png            # High-DPI tray icon
│       ├── start.wav           # Ascending recording chime
│       └── stop.wav            # Descending completion chime
├── tests/                      # Full pytest test suite with mocks (46 tests)
├── requirements.txt            # Pinned dependencies (including faster-whisper and ctranslate2)
├── pyproject.toml              # Ruff and Mypy configuration
├── build.bat                   # PyInstaller packaging script (onedir + onefile)
└── README.md
```

---

## Concurrency & Thread-Safety Rules

| Component | Execution Thread | Concurrency & Thread-Safety Rules |
| :--- | :--- | :--- |
| **PortAudio Audio Callback** | PortAudio C Thread | Extracts float32 samples, resamples if needed, updates the pre-roll RAM buffer, and writes RMS amplitude under a `threading.Lock`. **Never calls Qt widget methods or emits per-block Qt signals.** |
| **Keyboard Hook** | Pynput Win32 Hook Thread | Intercepts keyboard events, suppresses trigger keys, and emits Qt signals (`hotkey_down`, `hotkey_up`, `hotkey_cancel`). Handled via Qt queued connections on GUI thread. |
| **State Machine (`controller.py`)** | Controller / Multiple Threads | Explicit state transitions (`IDLE -> RECORDING -> PROCESSING -> INJECTING -> ERROR -> IDLE`) are guarded by `threading.Lock`. |
| **Transcription Worker** | Background Daemon Thread | Model inference is executed in a dedicated background worker (`TranscriptionWorker`), keeping the UI at a fluid 60 FPS. |
| **Text Injector (`text_injector.py`)** | GUI Main Thread | Clipboard access (`OpenClipboard`, `SetClipboardData`, etc.) and `SendInput` run on the main thread, always restoring clipboard in a `finally` block. |
| **Single Instance Mutex** | App Launch Thread | Named Win32 mutex `Global\VoiceTyper_SingleInstance_Mutex` prevents duplicate running instances. |

---

## Installation & Setup

### Prerequisites
- Windows 10 or Windows 11 (64-bit)
- Python 3.11+
- Optional: NVIDIA CUDA GPU for fastest local transcription (CPU fallback is included automatically)

### Setup Virtual Environment
```powershell
cd e:\voice2text
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running from Source
```powershell
python main.py
```

---

## Testing & Static Verification

Run static typing, code style checks, and the automated unit test suite:

```powershell
# Run Ruff linting
.\.venv\Scripts\ruff check voicetyper tests

# Run Mypy static type checking
.\.venv\Scripts\mypy voicetyper

# Run Pytest unit tests (46 unit tests)
.\.venv\Scripts\pytest tests/ -v
```

---

## Packaging into Standalone Executables

Run the automated PyInstaller build script:

```cmd
build.bat
```

This generates **both** deployment variants in the `dist/` directory:
1. **`dist\VoiceTyper\VoiceTyper.exe` (`--onedir`, Recommended):**
   - Faster application startup time.
   - Bundles all DLLs, CTranslate2 binaries, and assets in `dist\VoiceTyper\_internal\`.
2. **`dist\VoiceTyper-Standalone.exe` (`--onefile`, Optional):**
   - Single portable executable.

---

## Configuration Settings

Settings are stored in `%APPDATA%\VoiceTyper\config.json`. Any Gemini API key is securely encrypted in Windows Credential Manager:

```json
{
  "hotkey": "ctrl+shift",
  "transcription_engine": "local",
  "whisper_model": "large-v3-turbo",
  "whisper_device": "auto",
  "whisper_compute_type": "auto",
  "mic_device": null,
  "start_stop_beeps": true,
  "language_hint": "Auto",
  "max_duration_sec": 120.0,
  "min_duration_sec": 0.3,
  "start_with_windows": false,
  "paste_fallback_typing": false,
  "toggle_mode": false,
  "custom_vocabulary": ""
}
```

---

## License

MIT License. Designed and built with Google DeepMind technologies.
