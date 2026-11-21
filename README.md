# VoiceTyper 🎙️

**Production-grade, lightweight Windows 10/11 Push-to-Talk (PTT) & Toggle Voice-to-Text desktop utility in Python 3.11+.**

VoiceTyper allows you to press or hold a global hotkey (default: `Ctrl + Shift`), speak naturally in **Arabic (Egyptian & Modern Standard), English, or mixed code-switching**, and have your speech instantly transcribed **100% locally on your GPU/CPU** via **faster-whisper** (or optionally via Google Gemini 2.5 Flash) and automatically typed or pasted directly into any active window across Windows (VS Code, Chrome, WhatsApp Desktop/Web, Microsoft Word, Slack, Terminal, etc.).

---

## 📸 Visual Tour & Interface Overview

### 1. Unified Settings & Customization Dialog (`VoiceTyper Settings`)
Configure all audio, engine, hotkey, and theme preferences from one unified control panel:

![VoiceTyper Settings](docs/images/settings_dialog.png)

#### Detailed Settings Breakdown:
| Option / Setting | Description & Functionality |
| :--- | :--- |
| **Transcription Engine** | Choose between **Local (`faster-whisper`)** for 100% offline, zero-latency transcription or **Cloud (Google Gemini 2.5 Flash)** for ultra-fast cloud processing. |
| **Local Whisper Model** | Select your model: `large-v3-turbo` (Recommended for mixed Arabic/English), `large-v3` (High Accuracy), `medium`, or `small`. Shows real-time cache status (`[Downloaded ✓]`) and **Active Hardware** (`CPU` / `NVIDIA CUDA GPU`). |
| **Check Cache** | Instantly probes your local Hugging Face cache to check which models are stored on disk. |
| **Gemini API Key (Optional)** | Store an optional Gemini API key encrypted in the Windows Credential Manager. Includes **Show** and **Test** connection buttons. |
| **PTT Hotkey & Recorder** | Configured trigger hotkey (default: `Ctrl + Shift`). Click **Record Hotkey** to capture any 1-, 2-, or 3-key combination on the fly. Includes quick presets (`Presets...`). |
| **Overlay Theme & Live Preview** | Select between 4 themes: **Classic (Default)**, **Neon Glass Pill**, **Matte Dynamic Island**, and **Fluent Mica Card**. An interactive **Live Preview Box** below renders the theme instantly, with interactive state preview buttons (`Recording`, `Processing`, `Error`, `Standby`). |
| **Microphone Selection** | Choose your active recording microphone or leave it on `Default System Microphone`. Click **Refresh** to detect newly connected devices. |
| **Language Bias** | Set transcription mode: `Auto (Arabic, English & Mixed)` (with Egyptian dialect prompt tuning), `Arabic (ar)`, or `English (en)`. |
| **Max Duration** | Maximum speech recording duration limit in seconds (default: 120s) with auto-cutoff. |
| **English Casing** | Formats English words: `Capitalize Every Word (e.g. Hello World)`, `Sentence case`, `lowercase`, or `UPPERCASE`. |
| **Custom Vocabulary** | Comma-separated list of technical terms, company names, or programming identifiers (e.g., `PySide6, Docker, FastAPI, GitHub, Windows`) injected into Whisper's prompt. |
| **Audio Chimes** | Plays gentle ascending chime on recording start and descending chime on stop. |
| **Keep in Clipboard & Win+V** | Keeps every transcription in the Windows Clipboard and Windows History (`Win + V`) so you never lose spoken thoughts even if you switch windows. |
| **Start with Windows** | Automatically launches VoiceTyper minimized to the system tray on Windows boot. |
| **Unicode Keystroke Fallback** | Optional toggle to simulate typing character-by-character instead of `Ctrl + V` paste for special terminal emulators. |
| **Toggle Dictation Mode** | **Press once to start recording, press again to stop.** Eliminates the need to hold down keys during long dictation sessions. |
| **Reset Overlay Position** | Restores the floating pill position to above the Windows taskbar clock. |

---

### 2. Intelligent Floating Overlay States

The floating capsule overlay is fully frameless, draggable, and **non-activating** (`WS_EX_NOACTIVATE`) — it never steals focus or drops your text cursor from active editors or browsers.

#### 🔴 Recording State (Live Audio Visualizer)
Active when holding the hotkey or during toggle dictation:
![Recording State (LIVE)](docs/images/overlay_recording_live.png)
- **Features:**
  - Frosted glassmorphic obsidian pill with an animated coral-to-cyan gradient border.
  - Soft glowing red pulsing dot indicating active microphone capture.
  - Distinct **`● LIVE`** pill badge.
  - Real-time 60 FPS hybrid audio visualizer combining dynamic RMS micro-equalizer bars and fluid dual sine waves that react to your vocal volume.
  - Quick-access **Settings Gear Icon (⚙)** on the right edge to open settings immediately without moving the overlay.

---

#### ⚪ Standby State (Discreet & Non-Intrusive)
Appears when the overlay is idle or waiting for dictation:
![Standby State](docs/images/overlay_standby.png)
- **Features:**
  - Minimalist semi-translucent dark capsule displaying `VoiceTyper Standby`.
  - Stays cleanly on screen without distracting from code or reading.
  - Left-click draggable to reposition anywhere across single- or multi-monitor setups.
  - Automatically fades out when configured for inactivity auto-hide.

---

#### ⚠️ Error & Notice State (Clear Diagnostics & Retry)
Appears if microphone capture failed, no speech was spoken, or an engine timeout occurred:
![Error State](docs/images/overlay_error.png)
- **Features:**
  - Ambient warm amber warning glow border.
  - Amber circular warning icon `!`.
  - Descriptive status message (e.g., `No speech detected`, `Local engine error`, or `Microphone disconnected`).
  - One-click circular **Retry (⟳)** button to re-trigger recording immediately.
  - Settings gear icon (⚙) for instant troubleshooting.

---

## 🌟 Key Features & Innovations

- **100% Offline Local GPU/CPU Engine (`faster-whisper` + CTranslate2):**
  - **Zero Cloud API Requirement:** Transcribes speech locally with near-instantaneous turnaround.
  - **Auto Hardware Detection:** Automatically detects NVIDIA CUDA GPUs with `float16` / `int8_float16`; seamlessly falls back to CPU multi-threading with `int8`.
  - **Model Flexibility:** Default model `deepdml/faster-whisper-large-v3-turbo-ct2` (best accuracy and speed for Arabic + English code-switching), with choices for `large-v3`, `medium`, and `small`.
  - **Instant Download Speed:** Integrates `hf_transfer` for maximum bandwidth during model downloads.
- **Bilingual Egyptian Arabic + English Code-Switching:**
  - Proprietary prompt conditioning guides Whisper to transcribe English technical terms (e.g. `Python, Windows, GitHub, API, React`) in Latin script while transcribing Arabic speech in natural Arabic letters, avoiding awkward phonetic transliterations.
- **Dual Dictation Modes:**
  - **Push-to-Talk (Default):** Hold `Ctrl + Shift`, speak, release to type.
  - **Toggle Mode:** Press once to start, speak freely, press again to finish and type. Equipped with hardware key-repeat suppression and debounce protection.
- **Non-Activating Glassmorphic Overlay:**
  - Windows API level non-activating window (`WA_ShowWithoutActivating`, `WS_EX_NOACTIVATE`) ensures your target text field **never loses focus**.
  - Left-click draggable with persistent coordinates saved to `config.json`.
  - Live Theme Engine with 4 selectable styles: Classic, Neon Glass, Matte Dynamic Island, and Fluent Mica.
- **Zero Startup Latency (Pre-roll Ring Buffer):**
  - Maintains a continuous ~300 ms circular RAM buffer so the first syllable you speak is captured completely without clipping.
- **Clipboard History & Focus-Stealing Guard:**
  - Transcriptions are automatically added to **Windows Clipboard History (`Win + V`)** so your text is preserved even if you clicked away before dictation finished.
  - Detects if the active window changed during speech and displays a non-intrusive toast: *"Copied — press Ctrl+V"*.

---

## 📁 Architecture & Project Structure

```
Voice typer/
├── main.py                     # Application entry point & single-instance lock
├── voicetyper/
│   ├── __init__.py
│   ├── main.py                 # System tray lifecycle, singletons, Qt event wiring
│   ├── controller.py           # Thread-safe state machine (Idle -> Recording -> Processing -> Injecting -> Error)
│   ├── audio_recorder.py       # PortAudio stream capture, pre-roll ring buffer, peak normalization, RMS metering
│   ├── transcriber.py          # Unified transcription engine (Local faster-whisper + Gemini API)
│   ├── hotkey_manager.py       # Win32 low-level keyboard hook supporting 1-, 2-, 3-key combos & suppression
│   ├── text_injector.py        # Win32 focus verification, clipboard Win+V preservation, SendInput keystrokes
│   ├── config.py               # Atomic configuration persistence, Credential Manager vault
│   ├── paths.py                # sys._MEIPASS resource resolution, APPDATA paths, rotating logs
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── overlay.py          # Frameless, non-activating floating pill with 60 FPS multi-theme visualizer
│   │   └── settings_dialog.py  # Settings window with live theme preview box & interactive Hotkey Recorder
│   └── assets/
│       ├── icon.ico            # Windows application icon
│       ├── icon.png            # High-DPI tray icon
│       ├── start.wav           # Ascending recording chime
│       └── stop.wav            # Descending completion chime
├── docs/
│   └── images/                 # Screenshot assets for documentation
│       ├── settings_dialog.png
│       ├── overlay_recording_live.png
│       ├── overlay_standby.png
│       └── overlay_error.png
├── tests/                      # Full pytest test suite (70 unit tests)
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Ruff and Mypy lint/type configurations
├── build.bat                   # Automated PyInstaller build script (onedir + onefile)
├── README.md                   # English & Overview documentation
└── README_AR.md                # Full Arabic User & Architecture Guide (دليل المستخدم العربي)
```

---

## ⚙️ Concurrency & Thread-Safety Rules

| Component | Execution Thread | Concurrency & Safety Guarantee |
| :--- | :--- | :--- |
| **PortAudio Audio Callback** | PortAudio C Thread | Captures native float32 chunks into ring buffer. Computes RMS amplitude under `threading.Lock`. **Never invokes Qt UI methods directly.** |
| **Keyboard Hook** | Pynput Win32 Hook Thread | Intercepts keyboard events, suppresses trigger keys, and emits Qt signals (`hotkey_down`, `hotkey_up`, `hotkey_cancel`) handled via Qt queued connections on GUI thread. |
| **State Machine (`controller.py`)** | Controller / Multiple Threads | Explicit transitions (`IDLE -> RECORDING -> PROCESSING -> IDLE`) guarded by thread lock; job cancellation tokens prevent stale speech injections. |
| **Transcription Worker** | Background Daemon Thread | Heavy model inference runs off-screen, maintaining a fluid 60 FPS UI overlay. |
| **Text Injector (`text_injector.py`)** | GUI Main Thread | Clipboard access (`OpenClipboard`, `SetClipboardData`) and `SendInput` run on the main thread, always restoring original clipboard contents in a `finally` block. |
| **Single Instance Mutex** | App Launch Thread | Named Win32 mutex `Local\VoiceTyper_SingleInstance_Mutex` prevents duplicate running instances. |

---

## 🚀 Installation & Getting Started

### Prerequisites
- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Python:** Python 3.11 or Python 3.12
- **Optional:** NVIDIA GPU with CUDA support for maximum local performance (CPU fallback works out-of-the-box).

### Setup Environment
```powershell
# Clone or navigate to the directory
cd "D:\Git Hub\Voice typer"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running from Source
```powershell
python main.py
```

---

## 🧪 Testing & Code Verification

Run static typing, code linting, and the automated unit test suite:

```powershell
# 1. Run Ruff linter
ruff check .

# 2. Run Mypy static type checker
mypy voicetyper

# 3. Run Pytest unit tests (70 passed tests)
pytest tests/ -v
```

---

## 📦 Building Executables (`build.bat`)

To build standalone Windows executables without requiring a Python installation:

```cmd
build.bat
```

This compiles both variants into `dist/`:
1. **`dist\VoiceTyper\VoiceTyper.exe` (`--onedir`, Recommended):**
   - Fastest startup time, bundles all CTranslate2 and PySide6 runtimes cleanly.
2. **`dist\VoiceTyper-Standalone.exe` (`--onefile`):**
   - Single portable `.exe` that runs anywhere on Windows 10/11.

---

## 🌐 Arabic Documentation / الدليل العربي

للحصول على شرح تفصيلي كامل لكافة الإعدادات وميزات البرنامج باللغة العربية، يرجى الاطلاع على:
👉 **[README_AR.md (دليل المستخدم العربي الكامل)](README_AR.md)**

---

## License

MIT License. Designed and built with Google DeepMind technologies.
