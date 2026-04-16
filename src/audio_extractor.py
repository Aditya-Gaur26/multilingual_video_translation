"""
audio_extractor.py
──────────────────
Extract audio track from a video file using ffmpeg (via subprocess).
Outputs a mono WAV at 16 kHz – ideal for speech-to-text APIs.
"""

import os
import subprocess
from config.settings import AUDIO_FORMAT, AUDIO_SAMPLE_RATE, OUTPUT_DIR


def extract_audio(video_path: str, output_dir: str | None = None) -> str:
    """
    Extract audio from the given video file.

    Args:
        video_path:  Path to the input video file.
        output_dir:  Directory to write the audio file. Defaults to OUTPUT_DIR.

    Returns:
        Path to the extracted audio file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}.{AUDIO_FORMAT}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                          # drop video
        "-acodec", "pcm_s16le",         # 16-bit PCM
        "-ar", str(AUDIO_SAMPLE_RATE),  # sample rate
        "-ac", "1",                     # mono
        audio_path,
    ]

    print(f"[audio_extractor] Extracting audio → {audio_path}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[audio_extractor] Done. Size: {os.path.getsize(audio_path) / 1024:.1f} KB")

    return audio_path
