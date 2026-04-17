"""
video_muxer.py
──────────────
Assemble the final dubbed MKV video with:

  • Original video stream (copied, not re-encoded)
  • Original English audio (stream 0)
  • Dubbed audio tracks — one per target language
  • Soft-subtitle tracks — toggleable in any player (VLC / mpv / PotPlayer)

Uses ffmpeg to multiplex everything into a Matroska (.mkv) container.
"""

from __future__ import annotations
import logging
import os
import subprocess
from config.settings import ALL_LANGUAGES

logger = logging.getLogger("nptel_pipeline")


def mux_to_mkv(
    original_video: str,
    audio_tracks: dict[str, str],
    subtitle_tracks: dict[str, str],
    output_path: str,
    original_audio_label: str = "English (Original)",
    video_source: str | None = None,
) -> str:
    """
    Create a single MKV file with all streams.

    Args:
        original_video:      Path to the source video (contains video + English audio).
        audio_tracks:        lang_code → path to aligned dubbed audio (mp3).
        subtitle_tracks:     lang_code → path to .srt file.
        output_path:         Destination .mkv path.
        original_audio_label: Display name for the original audio track.
        video_source:        If set, take the video track from this file instead of
                             original_video (used for lip-synced MKVs where the video
                             frames come from a Wav2Lip-processed MP4).

    Returns:
        Path to the generated MKV file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    use_separate_video = video_source is not None and video_source != original_video

    # ── Build the ffmpeg command ──────────────────────────────────────────
    cmd: list[str] = ["ffmpeg", "-y"]

    if use_separate_video:
        # Input 0: lip-synced video (video track only — audio ignored)
        cmd += ["-i", video_source]
        # Input 1: original video (English audio only — video ignored)
        cmd += ["-i", original_video]
        video_input_idx = 0
        orig_audio_input_idx = 1
        input_idx = 2
    else:
        # Input 0: original video (video + English audio)
        cmd += ["-i", original_video]
        video_input_idx = 0
        orig_audio_input_idx = 0
        input_idx = 1

    # Inputs N..M: dubbed audio files
    audio_input_map: list[tuple[int, str]] = []  # (input_index, lang_code)
    for lang_code in sorted(audio_tracks):
        cmd += ["-i", audio_tracks[lang_code]]
        audio_input_map.append((input_idx, lang_code))
        input_idx += 1

    # Inputs M+1..: subtitle files
    sub_input_map: list[tuple[int, str]] = []
    for lang_code in sorted(subtitle_tracks):
        cmd += ["-i", subtitle_tracks[lang_code]]
        sub_input_map.append((input_idx, lang_code))
        input_idx += 1

    # ── Map streams ───────────────────────────────────────────────────────
    # Video from the appropriate input
    cmd += ["-map", f"{video_input_idx}:v:0"]

    # Original English audio
    cmd += ["-map", f"{orig_audio_input_idx}:a:0"]

    # Dubbed audio tracks
    for inp_idx, _lang in audio_input_map:
        cmd += ["-map", f"{inp_idx}:a:0"]

    # Subtitle tracks
    for inp_idx, _lang in sub_input_map:
        cmd += ["-map", f"{inp_idx}:0"]

    # ── Codec settings ────────────────────────────────────────────────────
    # Copy video (no re-encode)
    cmd += ["-c:v", "copy"]
    # Copy audio (all tracks) – TTS audio is already mp3
    cmd += ["-c:a", "copy"]
    # Subtitles as SRT inside MKV
    cmd += ["-c:s", "srt"]

    # ── Metadata: track titles & language tags ────────────────────────────
    # Audio stream 0 = original English
    audio_stream_idx = 0
    cmd += [
        f"-metadata:s:a:{audio_stream_idx}",
        f"title={original_audio_label}",
        f"-metadata:s:a:{audio_stream_idx}",
        "language=eng",
    ]

    # Audio streams 1..N = dubbed languages
    for i, (_, lang_code) in enumerate(audio_input_map, start=1):
        lang_info = ALL_LANGUAGES.get(lang_code, {})
        lang_name = lang_info.get("name", lang_code)
        iso = _lang_to_iso639_2(lang_code)
        cmd += [
            f"-metadata:s:a:{i}",
            f"title={lang_name} (Dubbed)",
            f"-metadata:s:a:{i}",
            f"language={iso}",
        ]

    # Subtitle streams
    for i, (_, lang_code) in enumerate(sub_input_map):
        lang_info = ALL_LANGUAGES.get(lang_code, {})
        lang_name = lang_info.get("name", lang_code)
        iso = _lang_to_iso639_2(lang_code)
        cmd += [
            f"-metadata:s:s:{i}",
            f"title={lang_name}",
            f"-metadata:s:s:{i}",
            f"language={iso}",
        ]

    # Set default tracks
    # If dubbed tracks exist, set the first dubbed track as default;
    # otherwise fall back to the original English audio.
    if audio_input_map:
        cmd += ["-disposition:a:0", "0"]  # English = not default
        cmd += ["-disposition:a:1", "default"]  # first dubbed = default
        for i in range(2, len(audio_input_map) + 1):
            cmd += [f"-disposition:a:{i}", "0"]
    else:
        cmd += ["-disposition:a:0", "default"]

    if sub_input_map:
        cmd += ["-disposition:s:0", "default"]
        for i in range(1, len(sub_input_map)):
            cmd += [f"-disposition:s:{i}", "0"]

    cmd.append(output_path)

    # ── Run ───────────────────────────────────────────────────────────────
    logger.info("Muxing MKV → %s", output_path)
    logger.info("  Audio tracks: English + %s",
                [lc for _, lc in audio_input_map])
    logger.info("  Subtitle tracks: %s",
                [lc for _, lc in sub_input_map])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg stderr:\n%s", result.stderr[-2000:])
        result.check_returncode()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Done → %s (%.1f MB)", output_path, size_mb)
    return output_path


# ── Utility ──────────────────────────────────────────────────────────────────

_ISO_MAP = {
    "en": "eng",
    "hi": "hin",
    "te": "tel",
    "od": "ori",
}


def _lang_to_iso639_2(lang_code: str) -> str:
    """Convert our short lang code to ISO-639-2/B for MKV metadata."""
    return _ISO_MAP.get(lang_code, lang_code)


# ── Preview MP4 (web-playable) ───────────────────────────────────────────────

def create_preview_mp4(
    original_video: str,
    dubbed_audio: str,
    output_path: str,
    subtitle_path: str | None = None,
    subtitle_lang_code: str | None = None,
) -> str:
    """
    Create a lightweight MP4 preview with a single dubbed audio track.

    If *subtitle_path* is given (SRT or ASS), subtitles are **burned** into
    the video so they display in any player including Streamlit's ``st.video``.

    The video stream is re-encoded to H.264 **only** when subtitles are
    burned; otherwise it is copied losslessly.

    Audio is encoded to AAC for web compatibility.

    Args:
        original_video:   Path to the source video.
        dubbed_audio:     Path to the dubbed audio file (mp3).
        output_path:      Where to write the preview MP4.
        subtitle_path:    Optional SRT/ASS file to burn into the video.
        subtitle_lang_code: Language code for font selection (hi/te/od).

    Returns:
        Path to the generated MP4 file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", original_video, "-i", dubbed_audio]

    if subtitle_path and os.path.isfile(subtitle_path):
        # Burn subtitles using the subtitles filter.
        # Use a Unicode-capable font for code-mixed text.
        font_name = _get_subtitle_font(subtitle_lang_code)
        # Escape path for ffmpeg filter (Windows backslashes, colons)
        escaped_sub = subtitle_path.replace("\\", "/").replace(":", "\\:")
        vf = (
            f"subtitles='{escaped_sub}'"
            f":force_style='FontName={font_name},FontSize=22,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"Outline=2,Shadow=1,MarginV=30'"
        )
        cmd += [
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart",
            output_path,
        ]
    else:
        # No subtitles — copy video losslessly, encode audio to AAC
        cmd += [
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart",
            output_path,
        ]

    logger.info("Creating preview MP4 → %s (subs=%s)", output_path,
                subtitle_path or "none")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg preview error:\n%s", result.stderr[-2000:])
        result.check_returncode()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Preview MP4 → %s (%.1f MB)", output_path, size_mb)
    return output_path


# ── Font helper for subtitle burning ─────────────────────────────────────────

_SUBTITLE_FONTS: dict[str, str] = {
    "hi": "Noto Sans Devanagari",
    "te": "Noto Sans Telugu",
    "od": "Noto Sans Oriya",
    "en": "Arial",
}

# Fallback font that covers Latin + many Indic scripts
_FALLBACK_FONT = "Arial Unicode MS"


def _get_subtitle_font(lang_code: str | None) -> str:
    """Return the best font name for burning subtitles in the given language."""
    if lang_code and lang_code in _SUBTITLE_FONTS:
        return _SUBTITLE_FONTS[lang_code]
    return _FALLBACK_FONT
