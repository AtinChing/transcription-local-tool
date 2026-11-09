#!/usr/bin/env python3
"""Transcribe an audio or video file with Qwen3-ASR (Apple Silicon) or faster-whisper.

Writes a timestamped, append-friendly text transcript to ``transcripts/`` by
default.  FFmpeg is used to decode common media formats.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from faster_whisper import WhisperModel

from asr_engine import QwenEngine, add_engine_arguments, resolve_engine, resolve_model


SCRIPT_DIR = Path(__file__).resolve().parent


def timestamp(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def run() -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio or video file.")
    parser.add_argument("input", type=Path, help="Audio or video file to transcribe")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Transcript destination (default: transcripts/<input stem>.txt)",
    )
    add_engine_arguments(parser)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu", help="Whisper only")
    parser.add_argument("--compute-type", default="int8", help="Whisper only")
    parser.add_argument("--no-vad", action="store_true")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    if not source.is_file():
        parser.error(f"input file not found: {source}")
    output = args.output or SCRIPT_DIR / "transcripts" / f"{source.stem}.txt"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    language = None if args.language.lower() == "auto" else args.language
    engine = resolve_engine(args.engine, args.model)
    model_name = resolve_model(engine, args.model)
    started = time.perf_counter()
    if engine == "qwen":
        print(f"Loading {model_name!r} (MLX, Apple GPU)…", flush=True)
        model = QwenEngine(model_name)
    else:
        print(f"Loading {model_name!r} ({args.device}, {args.compute_type})…", flush=True)
        model = WhisperModel(model_name, device=args.device, compute_type=args.compute_type)
    print(f"Model ready in {time.perf_counter() - started:.1f}s", flush=True)
    print(f"Transcribing: {source}", flush=True)

    if engine == "qwen":
        segments, info = model.transcribe(str(source), language=language, vad=not args.no_vad)
    else:
        segments, info = model.transcribe(
            str(source),
            language=language,
            vad_filter=not args.no_vad,
            vad_parameters={"min_silence_duration_ms": 1000, "speech_pad_ms": 400},
            condition_on_previous_text=False,
        )
    recorded = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with output.open("w", encoding="utf-8") as transcript:
        transcript.write("=" * 80 + "\n")
        transcript.write(f"Source: {source}\n")
        transcript.write(f"Transcribed (local): {recorded}\n")
        transcript.write(f"Model: {model_name} | Language: {info.language}\n")
        transcript.write("=" * 80 + "\n\n")
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcript.write(f"[{timestamp(segment.start)} - {timestamp(segment.end)}] {text}\n")
    print(f"Saved transcript: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
