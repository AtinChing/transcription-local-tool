#!/usr/bin/env python3
"""
Download the audio of an online lecture video and transcribe it with faster-whisper.

Built for UC Davis Kaltura MediaSpace links (https://video.ucdavis.edu/media/...),
but any site yt-dlp supports works too.

Audio quality: for Kaltura videos the original uploaded file ("source" flavor) is
located through Kaltura's API and its audio track is extracted without
re-encoding. Every other Kaltura rendition is a lossy re-encode of that file.
Other sites use yt-dlp's best audio stream.

Output matches transcribe_mic.py: <School>/<class>/transcripts/<session>.txt,
written in the same block format (one block per --seconds of video time).
Transcription settings (large-v3-turbo, cpu/int8 on macOS, VAD, no-speech gate,
silence-hallucination filter) are the same defaults as transcribe_mic.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from faster_whisper import WhisperModel
import yt_dlp

from transcribe_mic import (
    default_school_root,
    list_course_dirs,
    looks_like_silence_hallucination,
    output_path_for_session,
    pick_device_and_compute,
    resolve_course_dir,
    sanitize_session,
)

KALTURA_API = "https://www.kaltura.com/api_v3/service"


def timestamp(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def _kaltura_call(path: str, params: dict[str, str]) -> object:
    query = urllib.parse.urlencode({**params, "format": "1"})
    with urllib.request.urlopen(f"{KALTURA_API}/{path}?{query}", timeout=30) as resp:
        data = json.load(resp)
    if isinstance(data, dict) and data.get("objectType") == "KalturaAPIException":
        raise RuntimeError(data.get("message", "Kaltura API error"))
    return data


def kaltura_source_flavor(partner_id: str, entry_id: str, ks: str | None = None) -> str | None:
    """Return the flavorId of the original upload, or None if unavailable."""
    if ks is None:
        session = _kaltura_call(
            "session/action/startWidgetSession", {"widgetId": f"_{partner_id}"}
        )
        ks = session["ks"]
    assets = _kaltura_call(
        "flavorasset/action/getByEntryId", {"entryId": entry_id, "ks": ks}
    )
    for asset in assets:
        # status 2 = READY
        if asset.get("isOriginal") and asset.get("status") == 2:
            return asset["id"]
    return None


def choose_format(info: dict) -> tuple[str, str]:
    """Pick a yt-dlp format selector and a human description of the choice."""
    if info.get("extractor_key") == "Kaltura":
        partner_id = None
        for f in info.get("formats", []):
            m = re.search(r"/p/(\d+)/", f.get("url", ""))
            if m:
                partner_id = m.group(1)
                break
        if partner_id:
            try:
                flavor = kaltura_source_flavor(partner_id, info["id"])
            except Exception as e:  # API hiccup: fall back to yt-dlp's choice
                print(f"Kaltura source lookup failed ({e}); using best available.",
                      file=sys.stderr)
                flavor = None
            if flavor:
                for f in info["formats"]:
                    if f"/flavorId/{flavor}" in f.get("url", ""):
                        return (
                            f"{f['format_id']}/bestaudio/best",
                            f"Kaltura original upload ({f['format_id']}, flavor {flavor})",
                        )
    return "bestaudio/best", "best audio stream reported by yt-dlp"


def guess_course(title: str, school: Path) -> str | None:
    """Match a course folder name (e.g. ECS154A) inside the video title."""
    squashed = re.sub(r"\s+", "", title).lower()
    hits = [c for c in list_course_dirs(school) if c.replace(" ", "").lower() in squashed]
    if not hits:
        return None
    # Prefer the most specific match (ECS154A over ECS154, if both exist).
    return max(hits, key=len)


def download_audio(url: str, dest_dir: Path, ydl_opts: dict) -> tuple[Path, dict, str]:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") in ("playlist", "multi_video"):
        raise SystemExit("That link is a playlist/channel; pass a single video link.")

    fmt, why = choose_format(info)
    print(f"Audio source: {why}", flush=True)
    opts = {
        **ydl_opts,
        "format": fmt,
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        # Pull the audio track out as-is (no re-encode) and drop the video.
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "best"}],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.process_ie_result(info, download=True)
    audio = next(
        (p for p in dest_dir.iterdir() if p.is_file() and not p.name.endswith(".part")),
        None,
    )
    if audio is None:
        raise SystemExit("Download finished but no audio file was produced.")
    return audio, info, why


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe an online lecture video (UC Davis Kaltura or any yt-dlp site).",
        epilog=(
            "Examples:\n"
            "  %(prog)s https://video.ucdavis.edu/media/<video-name>/<entry-id>\n"
            "    (class guessed from the title, session = video title)\n"
            '  %(prog)s URL --class ECS154A --session "lecture3"\n'
            "  %(prog)s URL --cookies-from-browser brave   (for login-only videos)\n"
            "Output: <School>/<class>/transcripts/<session>.txt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", help="Video page URL")
    parser.add_argument("--class", "--course", dest="course", metavar="NAME",
                        help="Course folder under School (default: guessed from video title)")
    parser.add_argument("--session", metavar="NAME",
                        help="Transcript file name (default: the video title)")
    parser.add_argument("--school-root", type=Path, default=None,
                        help=f"Parent folder containing class folders (default: {default_school_root()})")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write to this file instead of <class>/transcripts/<session>.txt")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing transcript (default: refuse)")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Save the downloaded audio next to the transcript")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                        help="Use your browser's login (e.g. brave, chrome) for restricted videos")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="Video time covered by each transcript block (default: 60)")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default=None,
                        help="CTranslate2 device override: cpu, cuda, or auto (default: cpu+int8 on macOS)")
    parser.add_argument("--language", default="en", help="Language code, or 'auto'")
    parser.add_argument("--no-speech-threshold", type=float, default=0.82, metavar="X")
    parser.add_argument("--no-vad", action="store_true", help="Disable Silero VAD")
    parser.add_argument("--vad-min-silence-ms", type=int, default=5200, metavar="MS")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=600, metavar="MS")
    args = parser.parse_args()

    school = (args.school_root or default_school_root()).resolve()
    ydl_opts: dict = {"quiet": True, "no_warnings": True, "noprogress": False}
    if args.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (args.cookies_from_browser,)

    print("Looking up video…", flush=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        meta = ydl.extract_info(args.url, download=False, process=False)
    title = meta.get("title") or meta.get("id") or "video"
    print(f"Title: {title}", flush=True)

    course = args.course or guess_course(title, school)
    session = sanitize_session(args.session or title)
    if args.output:
        out_path = args.output.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if not course:
            print(
                f"Couldn't tell the class from the title {title!r}; pass --class "
                "(see: python transcribe_mic.py --list-classes).",
                file=sys.stderr,
            )
            return 1
        try:
            course_dir = resolve_course_dir(school, course)
        except (OSError, ValueError, FileNotFoundError) as e:
            print(e, file=sys.stderr)
            return 1
        out_path = output_path_for_session(course_dir, session).resolve()
    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        print(f"{out_path} already exists; use --overwrite or a different --session.",
              file=sys.stderr)
        return 1
    print(f"Class: {course or '-'} | Session: {session}\nWriting to: {out_path}", flush=True)

    with tempfile.TemporaryDirectory(prefix="lecture-audio-") as tmp:
        print("Downloading audio…", flush=True)
        audio, info, audio_note = download_audio(args.url, Path(tmp), ydl_opts)
        print(f"Audio ready: {audio.name} ({audio.stat().st_size / 1e6:.1f} MB)", flush=True)
        if args.keep_audio:
            kept = out_path.with_suffix(audio.suffix)
            kept.write_bytes(audio.read_bytes())
            print(f"Saved audio: {kept}", flush=True)

        device, compute_type = pick_device_and_compute(args.device)
        print(f"Loading model {args.model!r} (device={device!r}, compute_type={compute_type!r})…",
              flush=True)
        t0 = time.perf_counter()
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
        print(f"Model ready in {time.perf_counter() - t0:.1f}s", flush=True)

        use_vad = not args.no_vad
        segments, tinfo = model.transcribe(
            str(audio),
            language=None if args.language.lower() == "auto" else args.language,
            vad_filter=use_vad,
            vad_parameters={
                "min_silence_duration_ms": max(0, args.vad_min_silence_ms),
                "speech_pad_ms": max(0, args.vad_speech_pad_ms),
            } if use_vad else None,
            no_speech_threshold=args.no_speech_threshold,
            condition_on_previous_text=False,
        )
        duration = tinfo.duration
        print(f"Transcribing {timestamp(duration)} of audio…", flush=True)

        block_len = max(1.0, args.seconds)
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        sep = "=" * 80 + "\n"
        t_start = time.perf_counter()

        with out_path.open("w", encoding="utf-8") as f:
            f.write(
                f"{sep}"
                f"Source: {args.url}\n"
                f"Title: {title}\n"
                f"Audio: {audio_note}\n"
                f"Transcribed (local): {now}\n"
                f"Class: {course or '-'} | Session: {session}\n"
                f"Model: {args.model} | Language: {tinfo.language} | Length: {timestamp(duration)}\n"
                f"{sep}"
            )

            def write_block(index: int, parts: list[str]) -> None:
                start = index * block_len
                end = min(duration, start + block_len)
                text = " ".join(parts).strip() or "[no speech detected]"
                f.write(
                    f"\n{sep}"
                    f"Video time: {timestamp(start)} - {timestamp(end)}\n"
                    f"Class: {course or '-'} | Session: {session}\n"
                    f"Model: {args.model} | Chunk: {index + 1}\n"
                    f"{sep}"
                    f"{text}\n"
                )
                f.flush()

            block, parts = 0, []
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                if looks_like_silence_hallucination(text):
                    print(f"Dropped likely hallucination at {timestamp(seg.start)}: {text!r}",
                          file=sys.stderr)
                    continue
                seg_block = int(seg.start // block_len)
                while block < seg_block:
                    write_block(block, parts)
                    block, parts = block + 1, []
                    elapsed = time.perf_counter() - t_start
                    print(f"  {timestamp(block * block_len)} / {timestamp(duration)} "
                          f"({elapsed:.0f}s elapsed)", flush=True)
                parts.append(text)
            last_block = max(block, int(max(0.0, duration - 1e-6) // block_len))
            while block <= last_block:
                write_block(block, parts)
                block, parts = block + 1, []

    print(f"Done in {time.perf_counter() - t_start:.0f}s. Transcript: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
