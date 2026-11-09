"""
Speech-to-text engine selection shared by the transcribe_*.py scripts.

Qwen3-ASR runs on the Apple GPU through mlx-audio and is the default on Apple
Silicon. Whisper (faster-whisper on the CPU) is the fallback everywhere else.
QwenEngine.transcribe() mirrors WhisperModel.transcribe(): it returns lazy
segments with .start/.end/.text plus an info object with .language/.duration.
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

QWEN_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
WHISPER_MODEL = "large-v3-turbo"
SAMPLE_RATE = 16_000

_QWEN_LANGUAGES = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ko": "Korean",
    "ja": "Japanese",
}

# Qwen invents short text ("Okay.") on chunks with no speech. A permissive
# threshold skips those without dropping chunks that only have quiet speech.
_VAD_THRESHOLD = 0.3


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionInfo:
    language: str
    duration: float


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=("qwen", "whisper"),
        default=None,
        help="Speech-to-text engine (default: qwen on Apple Silicon, otherwise whisper).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name or path (default: {QWEN_MODEL} or {WHISPER_MODEL}, per engine).",
    )


def resolve_engine(engine: str | None, model: str | None) -> str:
    if engine:
        return engine
    if model and "qwen" not in model.lower():
        return "whisper"
    on_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    return "qwen" if on_apple_silicon else "whisper"


def resolve_model(engine: str, model: str | None) -> str:
    return model or (QWEN_MODEL if engine == "qwen" else WHISPER_MODEL)


def split_at_quiet_points(audio: np.ndarray, chunk_seconds: float) -> list[tuple[int, int]]:
    """Cut audio into (start, end) sample ranges about chunk_seconds long.

    Each cut is the quietest moment just after a multiple of chunk_seconds, so
    words stay whole and a chunk always starts inside its own time slot.
    """
    step = int(chunk_seconds * SAMPLE_RATE)
    search = int(min(5.0, chunk_seconds / 4) * SAMPLE_RATE)
    window = int(0.1 * SAMPLE_RATE)
    cuts = [0]
    boundary = step
    while boundary + search + SAMPLE_RATE < len(audio):
        region = audio[boundary : boundary + search]
        energy = np.convolve(region**2, np.ones(window) / window, mode="valid")
        cuts.append(boundary + int(np.argmin(energy)) + window // 2)
        boundary += step
    cuts.append(len(audio))
    return list(zip(cuts, cuts[1:]))


class QwenEngine:
    def __init__(self, model_name: str = QWEN_MODEL) -> None:
        # Imported here because mlx only installs on Apple Silicon.
        from mlx_audio.stt import load

        self._model = load(model_name)

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        *,
        language: str | None = "en",
        vad: bool = True,
        chunk_seconds: float = 60.0,
    ) -> tuple[Iterator[Segment], TranscriptionInfo]:
        if language is not None and language.lower() not in _QWEN_LANGUAGES:
            supported = ", ".join(sorted(_QWEN_LANGUAGES))
            raise ValueError(f"Qwen3-ASR doesn't support {language!r}; use one of: {supported}, auto")
        if not isinstance(audio, np.ndarray):
            audio = decode_audio(str(audio), sampling_rate=SAMPLE_RATE)
        name = _QWEN_LANGUAGES[language.lower()] if language else None
        info = TranscriptionInfo(language or "auto", len(audio) / SAMPLE_RATE)
        return self._segments(audio, name, vad, chunk_seconds), info

    def _segments(
        self, audio: np.ndarray, language: str | None, vad: bool, chunk_seconds: float
    ) -> Iterator[Segment]:
        options = VadOptions(threshold=_VAD_THRESHOLD)
        for start, end in split_at_quiet_points(audio, chunk_seconds):
            chunk = audio[start:end]
            if len(chunk) == 0 or (vad and not get_speech_timestamps(chunk, options)):
                continue
            text = self._model.generate(chunk, language=language).text.strip()
            if text:
                yield Segment(start / SAMPLE_RATE, end / SAMPLE_RATE, text)
