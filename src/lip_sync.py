"""
lip_sync.py
───────────
Wav2Lip-based lip synchronisation for dubbed video.

Wav2Lip paper: "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild"
Repo: https://github.com/Rudrabha/Wav2Lip

Setup (one-time):
  # 1. Clone Wav2Lip into the project root (or any path):
  git clone https://github.com/Rudrabha/Wav2Lip
  # 2. Install Wav2Lip dependencies:
  pip install -r Wav2Lip/requirements.txt
  # 3. Download the pretrained checkpoint into Wav2Lip/checkpoints/:
  #    wav2lip_gan.pth  (GAN version, better quality)
  #    URL: https://iiitaphyd-my.sharepoint.com/personal/...
  #    or use the official Google Drive link from the Wav2Lip README.
  # 4. Set WAV2LIP_PATH env var (or it defaults to ./Wav2Lip):
  export WAV2LIP_PATH=/path/to/Wav2Lip

When Wav2Lip is NOT installed, the module gracefully falls back to a
passthrough mode that simply replaces the audio track with ffmpeg —
so the pipeline always produces output.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger("nptel_pipeline")

# Default path – can be overridden by the WAV2LIP_PATH environment variable.
# Priority:
#   1. WAV2LIP_PATH env var (explicit override)
#   2. Wav2Lip/ inside this repo  (after git submodule update --init)
#   3. ../Wav2Lip  relative to the repo root (original manual-clone location)
#
#   __file__ → .../multilingual_video_translation/src/lip_sync.py
#   2× dirname → .../multilingual_video_translation/  (repo root)
#   3× dirname → .../working_project/
_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IN_REPO      = os.path.join(_REPO_ROOT, "Wav2Lip")         # submodule location
_SIBLING      = os.path.join(os.path.dirname(_REPO_ROOT), "Wav2Lip")  # ../Wav2Lip

if os.path.isdir(_IN_REPO):
    _DEFAULT_WAV2LIP_PATH = _IN_REPO
else:
    _DEFAULT_WAV2LIP_PATH = _SIBLING

WAV2LIP_REPO = os.environ.get("WAV2LIP_PATH", _DEFAULT_WAV2LIP_PATH)


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class LipSyncResult:
    """Result of a single lip-sync run."""
    output_path: str
    success: bool
    language: str
    processing_time_sec: float
    quality_score: Optional[float] = None   # 0–1, higher is better
    fallback_used: bool = False             # True when Wav2Lip was unavailable
    error_message: Optional[str] = None


# ── Public helper ────────────────────────────────────────────────────────────

def is_wav2lip_available(
    wav2lip_path: str = WAV2LIP_REPO,
    checkpoint: str = "checkpoints/wav2lip_gan.pth",
) -> bool:
    """
    Return True only when Wav2Lip is properly installed with its checkpoint.
    The pipeline can run without it (passthrough mode), so this is checked
    only to surface status in the UI.
    """
    inference_py = os.path.join(wav2lip_path, "inference.py")
    ckpt_path = os.path.join(wav2lip_path, checkpoint)
    return os.path.isfile(inference_py) and os.path.isfile(ckpt_path)


# ── Main processor class ──────────────────────────────────────────────────────

class LipSyncProcessor:
    """
    Wav2Lip-based lip synchronisation for a translated/dubbed video.

    The processor accepts the original video (which has the speaker's face)
    and a new audio track (dubbed in a target language) and produces an
    output video where the speaker's mouth movements match the new audio.

    When Wav2Lip is not installed it transparently falls back to a simple
    ffmpeg audio-swap so the rest of the pipeline is never blocked.

    Typical usage::

        processor = LipSyncProcessor()
        result = processor.sync(
            video_path="lecture.mp4",
            audio_path="lecture_hi_aligned.mp3",
            output_path="lecture_hi_lipsync.mp4",
            language="hi",
        )
        if result.success and not result.fallback_used:
            print(f"Lip sync quality: {result.quality_score:.2f}")
    """

    def __init__(
        self,
        wav2lip_path: str = WAV2LIP_REPO,
        checkpoint: str = "checkpoints/wav2lip_gan.pth",
        face_detection_batch_size: int = 4,
        wav2lip_batch_size: int = 64,
        resize_factor: int = 1,
        pads: tuple = (0, 10, 0, 0),
    ):
        self.wav2lip_path = wav2lip_path
        self.checkpoint = os.path.join(wav2lip_path, checkpoint)
        self.face_batch = face_detection_batch_size
        self.wav2lip_batch = wav2lip_batch_size
        self.resize_factor = resize_factor
        self.pads = pads
        self._available = is_wav2lip_available(wav2lip_path, checkpoint)

        if not self._available:
            logger.warning(
                "Wav2Lip not found at '%s'. "
                "Lip sync will use PASSTHROUGH mode (audio-swap only — no facial animation). "
                "To enable full lip sync:\n"
                "  1. git clone https://github.com/Rudrabha/Wav2Lip\n"
                "  2. pip install -r Wav2Lip/requirements.txt\n"
                "  3. Download wav2lip_gan.pth → Wav2Lip/checkpoints/\n"
                "  4. Set WAV2LIP_PATH env var to the Wav2Lip directory.",
                wav2lip_path,
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        language: str = "unknown",
    ) -> LipSyncResult:
        """
        Apply lip synchronisation: make the speaker's mouth match *audio_path*.

        Args:
            video_path:   Original video containing the speaker's face.
            audio_path:   Dubbed / translated audio track.
            output_path:  Destination MP4 path for the synced output.
            language:     Target language code (for logging/metadata).

        Returns:
            :class:`LipSyncResult` — always succeeds (falls back to passthrough).
        """
        t_start = time.time()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if not self._available:
            logger.warning(
                "Wav2Lip unavailable — using audio-swap passthrough for '%s'", language
            )
            result = self._passthrough(video_path, audio_path, output_path, language)
            result.fallback_used = True
            result.processing_time_sec = time.time() - t_start
            return result

        try:
            logger.info("Running Wav2Lip lip sync for language: %s", language)
            self._run_wav2lip(video_path, audio_path, output_path)
            quality = self._compute_quality_score(output_path)

            return LipSyncResult(
                output_path=output_path,
                success=True,
                language=language,
                processing_time_sec=time.time() - t_start,
                quality_score=quality,
                fallback_used=False,
            )

        except Exception as exc:
            logger.error(
                "Wav2Lip failed for '%s': %s — falling back to audio-swap.\n"
                "Full error: %s",
                language, type(exc).__name__, str(exc),
            )
            result = self._passthrough(video_path, audio_path, output_path, language)
            result.fallback_used = True
            result.error_message = str(exc)
            result.processing_time_sec = time.time() - t_start
            return result

    def batch_sync(
        self,
        video_path: str,
        audio_paths: Dict[str, str],
        output_dir: str,
    ) -> Dict[str, LipSyncResult]:
        """
        Apply lip sync for every language audio track.

        Args:
            video_path:   Original video.
            audio_paths:  ``{lang_code: audio_path}``
            output_dir:   Directory for the output lip-synced MP4s.

        Returns:
            ``{lang_code: LipSyncResult}``
        """
        os.makedirs(output_dir, exist_ok=True)
        results: Dict[str, LipSyncResult] = {}

        for lang, audio_path in audio_paths.items():
            output_path = os.path.join(output_dir, f"lipsync_{lang}.mp4")
            results[lang] = self.sync(video_path, audio_path, output_path, lang)

        successful = sum(1 for r in results.values() if r.success)
        real_sync = sum(1 for r in results.values() if r.success and not r.fallback_used)
        logger.info(
            "Batch lip sync: %d/%d succeeded (%d with Wav2Lip, %d passthrough)",
            successful, len(results), real_sync, successful - real_sync,
        )
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_wav2lip(self, video_path: str, audio_path: str, output_path: str) -> None:
        """Segment-aware Wav2Lip entry point.

        For NPTEL-style lecture videos the speaker's face is only visible in
        some segments — the rest are slide-only frames.  Running Wav2Lip on the
        whole video fails as soon as it meets a frame with no face.

        Strategy
        --------
        1. Sample every ~0.5 s and run face detection to build a list of
           (start, end, has_face) time segments.
        2. For each *face* segment: extract the clip, extract the matching audio
           slice, run Wav2Lip on that clip alone.
        3. For each *no-face* segment: replace the audio track with ffmpeg
           (keeps the slide video intact, just swaps the sound).
        4. Concatenate all processed clips with ffmpeg.

        Wav2Lip's inference.py uses unquoted ffmpeg format strings, so all
        intermediate files live in a temp directory with no spaces in its path.
        """
        tmpdir = tempfile.mkdtemp(prefix="wav2lip_")
        try:
            fps, segments = self._detect_face_segments(video_path)

            # Fast path: all frames have faces → single Wav2Lip pass
            if all(s["has_face"] for s in segments):
                logger.info("All segments have faces – single-pass Wav2Lip")
                self._wav2lip_clip(video_path, audio_path, output_path, tmpdir, idx=0)
                return

            logger.info(
                "Mixed face/no-face video (%d segments) – segment-based processing",
                len(segments),
            )

            clip_paths = []
            for i, seg in enumerate(segments):
                t_start = seg["start"]
                t_dur   = seg["end"] - seg["start"]
                has_face = seg["has_face"]

                seg_video = os.path.join(tmpdir, f"s{i:03d}_v.mp4")
                seg_audio = os.path.join(tmpdir, f"s{i:03d}_a.wav")
                seg_out   = os.path.join(tmpdir, f"s{i:03d}_out.mp4")

                # Cut video segment (re-encode to ensure clean keyframes at cuts)
                subprocess.run(
                    ["ffmpeg", "-y",
                     "-ss", f"{t_start:.3f}", "-t", f"{t_dur:.3f}",
                     "-i", video_path,
                     "-c:v", "libx264", "-preset", "fast", "-an",
                     seg_video],
                    check=True, capture_output=True, timeout=300,
                )

                # Cut matching audio chunk
                subprocess.run(
                    ["ffmpeg", "-y",
                     "-ss", f"{t_start:.3f}", "-t", f"{t_dur:.3f}",
                     "-i", audio_path,
                     "-ar", "16000", "-ac", "1",
                     seg_audio],
                    check=True, capture_output=True, timeout=300,
                )

                if has_face:
                    try:
                        self._wav2lip_clip(seg_video, seg_audio, seg_out, tmpdir, idx=i)
                        logger.info("  segment %d [%.1f–%.1fs]: Wav2Lip OK", i, t_start, seg["end"])
                    except Exception as exc:
                        logger.warning(
                            "  segment %d [%.1f–%.1fs]: Wav2Lip failed (%s), using audio-swap",
                            i, t_start, seg["end"], exc,
                        )
                        self._ffmpeg_swap_clip(seg_video, seg_audio, seg_out)
                else:
                    logger.info(
                        "  segment %d [%.1f–%.1fs]: no face – audio-swap only",
                        i, t_start, seg["end"],
                    )
                    self._ffmpeg_swap_clip(seg_video, seg_audio, seg_out)

                clip_paths.append(seg_out)

            # Concatenate all processed clips
            concat_file = os.path.join(tmpdir, "concat.txt")
            with open(concat_file, "w") as f:
                for p in clip_paths:
                    f.write(f"file '{p}'\n")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            r = subprocess.run(
                ["ffmpeg", "-y",
                 "-f", "concat", "-safe", "0", "-i", concat_file,
                 "-c:v", "libx264", "-preset", "fast",
                 "-c:a", "aac", "-b:a", "128k",
                 output_path],
                capture_output=True, timeout=3600,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg concat failed:\n{r.stderr.decode()[-2000:]}"
                )
            logger.info("Segment-based Wav2Lip wrote: %s", output_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── segment detection ─────────────────────────────────────────────────────

    def _detect_face_segments(self, video_path: str):
        """Sample video at ~0.5 s intervals, run face detection, return segments.

        Returns
        -------
        fps : float
        segments : list of {'start': float, 'end': float, 'has_face': bool}
        """
        import cv2
        import numpy as np
        import torch

        if self.wav2lip_path not in sys.path:
            sys.path.insert(0, self.wav2lip_path)
        import face_detection as fd

        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur   = total / fps

        # Sample every ~0.5 s (but at least every 15 frames)
        step = max(1, int(fps * 0.5))
        indices, frames = [], []
        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                indices.append(idx)
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()

        logger.info("Face detection: %d sample frames at step=%d", len(frames), step)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = fd.FaceAlignment(fd.LandmarksType._2D, flip_input=False, device=device)
        preds = []
        for i in range(0, len(frames), self.face_batch):
            preds.extend(
                detector.get_detections_for_batch(np.array(frames[i:i + self.face_batch]))
            )
        del detector

        has_face = [p is not None for p in preds]

        # Build raw segments from per-frame states
        raw = []
        cur_state  = has_face[0]
        cur_start  = 0.0
        for frame_idx, state in zip(indices[1:], has_face[1:]):
            if state != cur_state:
                raw.append({"start": cur_start, "end": frame_idx / fps, "has_face": cur_state})
                cur_state = state
                cur_start = frame_idx / fps
        raw.append({"start": cur_start, "end": dur, "has_face": cur_state})

        # Merge short (<1 s) opposite-state gaps into their neighbours to avoid
        # many tiny clips (e.g. a single bad detection in a face run)
        merged = [dict(raw[0])]
        for seg in raw[1:]:
            seg_dur = seg["end"] - seg["start"]
            if merged[-1]["has_face"] == seg["has_face"]:
                merged[-1]["end"] = seg["end"]          # same state → extend
            elif seg_dur < 1.0:
                merged[-1]["end"] = seg["end"]          # tiny blip → absorb
            else:
                merged.append(dict(seg))

        for s in merged:
            logger.info(
                "  [%.1f – %.1f s]  has_face=%s", s["start"], s["end"], s["has_face"]
            )
        return fps, merged

    # ── per-clip helpers ──────────────────────────────────────────────────────

    def _wav2lip_clip(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        tmpdir: str,
        idx: int = 0,
    ) -> None:
        """Run Wav2Lip inference on a single clip.

        All I/O goes through safe (no-space) paths inside *tmpdir*.
        """
        v_ext = os.path.splitext(video_path)[1] or ".mp4"
        a_ext = os.path.splitext(audio_path)[1] or ".wav"

        safe_video  = os.path.join(tmpdir, f"face_{idx}{v_ext}")
        safe_audio  = os.path.join(tmpdir, f"audio_{idx}{a_ext}")
        safe_output = os.path.join(tmpdir, f"out_{idx}.mp4")

        if os.path.abspath(video_path) != os.path.abspath(safe_video):
            shutil.copy2(video_path, safe_video)
        if os.path.abspath(audio_path) != os.path.abspath(safe_audio):
            shutil.copy2(audio_path, safe_audio)

        pad_str = [str(p) for p in self.pads]
        cmd = [
            sys.executable,
            os.path.join(self.wav2lip_path, "inference.py"),
            "--checkpoint_path", self.checkpoint,
            "--face",            safe_video,
            "--audio",           safe_audio,
            "--outfile",         safe_output,
            "--face_det_batch_size", str(self.face_batch),
            "--wav2lip_batch_size",  str(self.wav2lip_batch),
            "--resize_factor",       str(self.resize_factor),
            "--pads",                *pad_str,
        ]
        logger.info("Wav2Lip clip[%d] cmd: %s", idx, " ".join(cmd))

        proc = subprocess.run(
            cmd, cwd=self.wav2lip_path,
            capture_output=True, text=True, timeout=3600,
        )
        if proc.stdout:
            logger.info("Wav2Lip stdout:\n%s", proc.stdout[-3000:])
        if proc.stderr:
            logger.info("Wav2Lip stderr:\n%s", proc.stderr[-3000:])
        if proc.returncode != 0:
            raise RuntimeError(
                f"Wav2Lip exited with code {proc.returncode}:\n{proc.stderr[-3000:]}"
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.move(safe_output, output_path)
        logger.info("Wav2Lip clip[%d] wrote: %s", idx, output_path)

    def _ffmpeg_swap_clip(
        self, video_path: str, audio_path: str, output_path: str
    ) -> None:
        """Replace audio on a clip (no face animation). Used for slide-only segments."""
        r = subprocess.run(
            ["ffmpeg", "-y",
             "-i", video_path, "-i", audio_path,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-shortest",
             output_path],
            capture_output=True, timeout=300,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio-swap failed:\n{r.stderr.decode()[-1000:]}"
            )

    def _passthrough(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        language: str,
    ) -> LipSyncResult:
        """
        Fallback: replace the audio track only using ffmpeg (no face animation).
        This always produces a valid output file so the pipeline is never blocked.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i",  video_path,
            "-i",  audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            return LipSyncResult(
                output_path=output_path,
                success=True,
                language=language,
                processing_time_sec=0.0,
                quality_score=None,
            )
        except subprocess.CalledProcessError as exc:
            return LipSyncResult(
                output_path=output_path,
                success=False,
                language=language,
                processing_time_sec=0.0,
                error_message=str(exc),
            )

    def _compute_quality_score(self, video_path: str) -> Optional[float]:
        """
        Heuristic quality score [0, 1] based on mouth-region motion variance.

        Uses OpenCV to sample every 5th frame, measures pixel-variance in the
        estimated mouth region (lower-centre of frame), then measures how much
        that variance fluctuates — more fluctuation → more animation → better
        lip sync signal.  Returns None if OpenCV is not available.
        """
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            mouth_vals: list[float] = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 5 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    region = gray[
                        int(h * 0.65): int(h * 0.85),
                        int(w * 0.30): int(w * 0.70),
                    ]
                    mouth_vals.append(float(np.std(region)))
                frame_idx += 1

            cap.release()

            if len(mouth_vals) < 2:
                return None

            arr = np.array(mouth_vals)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            score = float(np.clip(np.std(arr) * 4.0, 0.0, 1.0))
            logger.info("Lip sync quality score: %.3f", score)
            return score

        except Exception as exc:
            logger.warning("Quality score computation skipped: %s", exc)
            return None
