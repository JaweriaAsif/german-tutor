"""Text-to-speech for pronunciation help.

Dependency-free: uses the macOS built-in `say` command (which ships native German
voices) to synthesize audio, and `afplay` to play it. No API key, works offline.
On non-macOS systems `tts_available()` returns False and the tool degrades
gracefully.

Cloud TTS (ElevenLabs/Azure/etc.) could be slotted in behind `synthesize()` later
without changing the tool interface.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

# Anna is the classic de_DE voice present on macOS.
DEFAULT_VOICE = os.getenv("TUTOR_TTS_VOICE", "Anna")
DEFAULT_DIR = Path(os.getenv("TUTOR_TTS_DIR", ".german_tutor/audio"))


def tts_available() -> bool:
    """True if a local TTS engine (macOS `say`) is available."""
    return shutil.which("say") is not None


def _slug(text: str, voice: str) -> str:
    return f"{voice}-{hashlib.sha1((voice + '|' + text).encode('utf-8')).hexdigest()[:12]}"


def synthesize(text: str, voice: str | None = None, out_dir: str | Path | None = None) -> Path | None:
    """Synthesize `text` to an AIFF file and return its path (cached by content)."""
    if not tts_available() or not text.strip():
        return None
    voice = voice or DEFAULT_VOICE
    out_dir = Path(out_dir or DEFAULT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_slug(text, voice)}.wav"
    if path.exists():
        return path
    # `say` only writes a WAV with explicit format flags (a bare .wav fails).
    fmt = ["--file-format=WAVE", "--data-format=LEI16@22050"]
    try:
        subprocess.run(["say", "-v", voice, *fmt, "-o", str(path), text], check=True)
    except subprocess.CalledProcessError:
        # Requested voice unavailable: fall back to the system default voice.
        subprocess.run(["say", *fmt, "-o", str(path), text], check=True)
    return path


def play(path: str | Path) -> bool:
    """Play an audio file without blocking. Returns False if no player is found."""
    player = shutil.which("afplay")
    if not player:
        return False
    subprocess.Popen([player, str(path)])  # non-blocking
    return True
