"""VoiceTyper launcher entry point."""

import importlib.util
import os
import sys
from pathlib import Path

# Enable high-speed downloads via hf-transfer and disable Windows symlink warnings
if importlib.util.find_spec("hf_transfer") is not None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from voicetyper.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
