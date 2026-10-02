# transcription-local-tool

Local lecture transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(Whisper `large-v3-turbo` by default, CPU + int8 on macOS). Everything runs on your machine.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install yt-dlp   # only needed for transcribe_video.py
```

FFmpeg must be installed (e.g. `brew install ffmpeg`).

Transcripts are written to `<School>/<class>/transcripts/<session>.txt`, where `<School>`
is the parent folder of this repo and each class is a folder inside it.

## Scripts

**`transcribe_mic.py`**: live microphone transcription during class.

```bash
.venv/bin/python transcribe_mic.py --list-classes
.venv/bin/python transcribe_mic.py --class ECS171 --session lecture3 --continuous --seconds 60
```

**`transcribe_video.py`**: transcribe an online lecture recording (UC Davis Kaltura
MediaSpace, or any site yt-dlp supports). For Kaltura it locates the original upload and
extracts its audio without re-encoding.

```bash
.venv/bin/python transcribe_video.py "https://video.ucdavis.edu/media/<name>/<entry-id>"
.venv/bin/python transcribe_video.py "<url>" --class ECS154A --session lecture2
.venv/bin/python transcribe_video.py "<url>" --cookies-from-browser brave   # login-only videos
```

*Canvas Media Gallery videos* don't show a direct link, but you can pass the Kaltura
entry ID instead. To find it:

1. Open the video in Canvas, then open DevTools (`Cmd+Option+I`) → **Network** tab.
2. Type `entryId` in the filter box and press play on the video.
3. A request appears containing `.../entryId/1_xxxxxxxx/...`. That `1_xxxxxxxx` is the ID.
   (If nothing shows up, reload the page with DevTools open and press play again.)

```bash
.venv/bin/python transcribe_video.py "kaltura:1770401:1_xxxxxxxx" --class ECS174 --session lecture3
```

`1770401` is UC Davis's Kaltura partner ID and never changes. Pass `--class` for these, since
titles like `ECS-174: 2026-10-01 13:37` don't match the class folder name.

**`transcribe_media.py`**: transcribe a local audio or video file.

```bash
.venv/bin/python transcribe_media.py recording.m4a
```
