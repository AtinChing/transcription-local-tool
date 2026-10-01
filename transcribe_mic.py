#!/usr/bin/env python3
"""
Record from the default microphone and transcribe with faster-whisper.
Transcripts are always appended to a per-class, per-session file under
<School>/<class>/transcripts/<session>.txt

faster-whisper uses CTranslate2 (CPU/CUDA). On Apple Silicon there is no
PyTorch MPS path; use device=auto or cpu — inference is still fast with int8.
"""

from __future__ import annotations

import argparse
import queue
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Default input sample rate for Whisper
SAMPLE_RATE = 16_000
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCHOOL_ROOT = SCRIPT_DIR.parent
SKIP_DIR_NAMES = frozenset({"mic-faster-whisper"})

# Whisper often emits these on silence/room noise when it should output nothing.
_HALLUCINATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:thank you[.!?,;\s]*)+$", re.IGNORECASE),
    re.compile(r"^(thanks?( for (watching|listening))?[.!,\s]*)+$", re.IGNORECASE),
    re.compile(r"^(\s*you[.!,\s]*)+$", re.IGNORECASE),
    re.compile(r"^(uh+,?\s*)+$", re.IGNORECASE),
    re.compile(r"^(mm+-?hm+[.!,\s]*)+$", re.IGNORECASE),
    re.compile(r"^(subtitle(s|d)?\s*(by\b.*)?)[.!,\s]*$", re.IGNORECASE),
    re.compile(r"^\[?\s*music\s*\]?\s*$", re.IGNORECASE),
    re.compile(r"^\.+$"),
)


def looks_like_silence_hallucination(text: str, max_chars: int = 96) -> bool:
    t = " ".join(text.split()).strip()
    if len(t) > max_chars:
        return False
    return any(p.match(t) for p in _HALLUCINATION_PATTERNS)


def default_school_root() -> Path:
    return DEFAULT_SCHOOL_ROOT


def list_course_dirs(school: Path) -> list[str]:
    if not school.is_dir():
        return []
    names: list[str] = []
    for p in school.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if p.name in SKIP_DIR_NAMES:
            continue
        names.append(p.name)
    return sorted(names, key=str.lower)


def sanitize_session(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("session name is empty")
    # No path components or nulls
    cleaned = re.sub(r'[/\\\0]', "_", name)
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError("session name is invalid after sanitization")
    return cleaned


def resolve_course_dir(school: Path, course: str) -> Path:
    course = course.strip()
    if not course:
        raise ValueError("class/course name is empty")
    # Single segment only (must match a top-level folder)
    if "/" in course or course in (".", ".."):
        raise ValueError(
            'use a single folder name for --class (e.g. "ECS171"), not a path'
        )
    d = (school / course).resolve()
    try:
        d.relative_to(school.resolve())
    except ValueError as e:
        raise ValueError("class folder must be inside the school root") from e
    if not d.is_dir():
        raise FileNotFoundError(
            f'no folder "{course}" under {school}. '
            f'Try: python transcribe_mic.py --list-classes'
        )
    return d


def output_path_for_session(course_dir: Path, session: str) -> Path:
    safe = sanitize_session(session)
    out_dir = course_dir / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe}.txt"


def pick_device_and_compute(user_device: str | None) -> tuple[str, str]:
    if user_device:
        return user_device, "default"
    # Apple Silicon: fast CPU + int8; avoid assuming CUDA
    if sys.platform == "darwin":
        return "cpu", "int8"
    return "auto", "default"


_MIN_TRANSCRIBE_SEC = 0.5
_POLL_INTERVAL = 0.15


class MicRecorder:
    """Keeps one InputStream open for the whole session so the OS mic stays on."""

    def __init__(self) -> None:
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._leftover = np.array([], dtype=np.float32)

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[sounddevice] {status}", file=sys.stderr)
        self._q.put(indata[:, 0].copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        self._drain()
        self._leftover = np.array([], dtype=np.float32)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def record(
        self,
        duration: float,
        interrupt: threading.Event | None = None,
    ) -> np.ndarray:
        # Don't drain here: audio queued while the previous chunk was being
        # transcribed belongs at the start of this chunk.
        target_frames = int(round(duration * SAMPLE_RATE))
        parts: list[np.ndarray] = []
        n = 0
        if len(self._leftover):
            parts.append(self._leftover)
            n += len(self._leftover)
            self._leftover = np.array([], dtype=np.float32)
        while n < target_frames:
            if interrupt is not None and interrupt.is_set():
                break
            try:
                chunk = self._q.get(timeout=_POLL_INTERVAL)
            except queue.Empty:
                continue
            parts.append(chunk)
            n += len(chunk)
        if not parts:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(parts)
        if len(audio) > target_frames and (interrupt is None or not interrupt.is_set()):
            # Carry the overshoot into the next chunk instead of dropping it.
            self._leftover = audio[target_frames:]
            audio = audio[:target_frames]
        return audio

    def take_remaining(self) -> np.ndarray:
        """Return leftover + everything still queued. Call after stop()."""
        parts = [self._leftover]
        while True:
            try:
                parts.append(self._q.get_nowait())
            except queue.Empty:
                break
        self._leftover = np.array([], dtype=np.float32)
        return np.concatenate(parts)

    def _drain(self) -> None:
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break


def append_transcription(
    path: Path,
    *,
    course: str,
    session: str,
    model_name: str,
    loop_index: int,
    loops_total: int | None,
    text: str,
) -> None:
    now = datetime.now(timezone.utc).astimezone()
    if loops_total is None:
        chunk_line = f"Chunk: {loop_index + 1} (continuous)"
    else:
        chunk_line = f"Chunk: {loop_index + 1}/{loops_total}"
    sep = f"{'=' * 80}\n"
    block = (
        f"{sep}"
        f"Recorded (local): {now.isoformat(timespec='seconds')}\n"
        f"Class: {course} | Session: {session}\n"
        f"Model: {model_name} | {chunk_line}\n"
        f"{sep}"
    )
    prefix = "\n" if path.exists() and path.stat().st_size > 0 else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + block)
        f.write(text.rstrip() + "\n")


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe Mac microphone audio with faster-whisper; append per class/session.",
        epilog=(
            "Examples:\n"
            '  %(prog)s --class ECS171 --session "week5-lecture" --seconds 120\n'
            "  %(prog)s --class STA106 --session lecture --continuous --seconds 90\n"
            "    (record 90s chunks until you press Ctrl+C in the terminal)\n"
            "Output: <School>/<class>/transcripts/<session>.txt (always append).\n"
            "Silero VAD is ON by default (--vad-min-silence-ms 5200 for long pauses). "
            "Use --no-vad if breaks still get cut up or the first word after a pause drops."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--class",
        "--course",
        dest="course",
        metavar="NAME",
        help='Top-level course folder under your School directory (e.g. ECS171, STA106).',
    )
    parser.add_argument(
        "--session",
        metavar="NAME",
        help='Label for this transcript file (same session reuses one file; always append).',
    )
    parser.add_argument(
        "--school-root",
        type=Path,
        default=None,
        help=f"Parent folder containing class folders (default: {DEFAULT_SCHOOL_ROOT})",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="Print course folder names under the school root and exit.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="Length of each recording segment in seconds (default: 60).",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of back-to-back record+transcribe cycles (default: 1).",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "Repeat record+transcribe until you press Ctrl+C in the terminal. "
            "Each cycle uses --seconds of audio (or less if you interrupt mid-chunk)."
        ),
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model size or path (default: large-v3-turbo).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='CTranslate2 device override: cpu, cuda, or auto (default: cpu+int8 on macOS).',
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Spoken language code (default: en). Use 'auto' for detection on each run.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.82,
        metavar="X",
        help=(
            "Whisper no-speech gate; higher reduces fake text on silence (default: 0.82)."
        ),
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Silero VAD (default is ON: silence is removed before Whisper).",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=5200,
        metavar="MS",
        help=(
            "Silero VAD: how long silence (ms) must last before a speech segment ends. "
            "Default 5200 (~5 s) fits instructors who pause a lot; use 2000 for tighter cuts."
        ),
    )
    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=600,
        metavar="MS",
        help="Extra audio kept before/after each speech segment (default: 600).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Append to this file instead of <class>/transcripts/<session>.txt",
    )

    args = parser.parse_args()
    if args.continuous and args.loops != 1:
        parser.error("use --continuous alone, not with --loops (leave --loops at default 1)")

    school = (args.school_root or default_school_root()).resolve()

    if args.list_classes:
        for name in list_course_dirs(school):
            print(name)
        return 0

    if not args.session:
        parser.error("--session is required unless using --list-classes")

    if not args.course:
        parser.error("--class / --course is required (or use --list-classes)")

    try:
        course_dir = resolve_course_dir(school, args.course)
    except (OSError, ValueError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1

    out_path = args.output if args.output else output_path_for_session(
        course_dir, args.session
    )
    out_path = out_path.resolve()
    if args.output:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    device, compute_type = pick_device_and_compute(args.device)
    if args.device:
        compute_type = "default"

    print(f"Loading model {args.model!r} (device={device!r}, compute_type={compute_type!r})…")
    t0 = time.perf_counter()
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
    )
    print(f"Model ready in {time.perf_counter() - t0:.1f}s")
    print(f"Appending to: {out_path}")

    mic = MicRecorder()
    mic.start()
    print("Microphone stream opened (stays open until done).")

    def transcribe_and_append(audio: np.ndarray, i: int, loops_total: int | None) -> None:
        print("Transcribing…")
        lang = None if args.language and args.language.lower() == "auto" else args.language
        use_vad = not args.no_vad
        vad_parameters = None
        if use_vad:
            vad_parameters = {
                "min_silence_duration_ms": max(0, args.vad_min_silence_ms),
                "speech_pad_ms": max(0, args.vad_speech_pad_ms),
            }
        segments, info = model.transcribe(
            audio,
            language=lang,
            vad_filter=use_vad,
            vad_parameters=vad_parameters,
            log_progress=False,
            no_speech_threshold=args.no_speech_threshold,
            condition_on_previous_text=False,
        )
        parts: list[str] = []
        for seg in segments:
            parts.append(seg.text.strip())
        text = " ".join(p for p in parts if p).strip()
        if not text:
            text = "[no speech detected]"
        elif looks_like_silence_hallucination(text):
            print(
                "Ignored short phrase typical of silence/noise hallucination; "
                f"raw model output was: {text!r}",
                file=sys.stderr,
            )
            text = "[no speech detected — silence/noise; raise mic or wait for speech]"

        append_transcription(
            out_path,
            course=args.course.strip(),
            session=sanitize_session(args.session),
            model_name=args.model,
            loop_index=i,
            loops_total=loops_total,
            text=text,
        )
        print(f"Appended ({info.language}, duration ~{info.duration:.1f}s audio).")

    try:
        if args.continuous:
            interrupt = threading.Event()
            prev = signal.signal(signal.SIGINT, lambda _s, _f: interrupt.set())
            print(
                f"\nContinuous mode: {args.seconds}s chunks, Ctrl+C to stop.\n"
            )
            i = 0
            pending = np.array([], dtype=np.float32)
            try:
                while not interrupt.is_set():
                    print(f"Recording segment {i + 1} for up to {args.seconds}s…")
                    audio = mic.record(args.seconds, interrupt)
                    if interrupt.is_set():
                        # Partial slice goes into the final flush below.
                        pending = audio
                        break
                    transcribe_and_append(audio, i, None)
                    i += 1
                # Stopped: close the mic, then transcribe everything still
                # buffered (partial slice + backlog queued during transcription).
                mic.stop()
                audio = np.concatenate([pending, mic.take_remaining()])
                dur_sec = len(audio) / SAMPLE_RATE
                if dur_sec >= _MIN_TRANSCRIBE_SEC:
                    print(f"Stopped. Transcribing remaining {dur_sec:.1f}s of audio…")
                    transcribe_and_append(audio, i, None)
                else:
                    print("Stopped (remaining audio too short to transcribe).")
            finally:
                signal.signal(signal.SIGINT, prev)
        else:
            loops = max(1, args.loops)
            for i in range(loops):
                print(f"\nRecording segment {i + 1}/{loops} for {args.seconds}s…")
                audio = mic.record(args.seconds)
                transcribe_and_append(audio, i, loops)
    finally:
        mic.stop()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
