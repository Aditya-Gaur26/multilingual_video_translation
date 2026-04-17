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

import numpy as np

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

# Google Drive file IDs for the pretrained checkpoints
_WAV2LIP_GAN_GDRIVE_ID  = "1_OvqStxNxLc7bXzlaVG5sz695p-FVfYY"   # wav2lip_gan.pth (~416 MB)
_S3FD_URL               = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"


def _ensure_wav2lip_checkpoints(wav2lip_path: str = WAV2LIP_REPO) -> None:
    """
    Auto-download required Wav2Lip checkpoints if not already present.

    Downloads (one-time, ~430 MB total):
      • checkpoints/wav2lip_gan.pth  — from Google Drive via gdown
      • face_detection/detection/sfd/s3fd.pth — from adrianbulat.com
    """
    # ── wav2lip_gan.pth ──────────────────────────────────────────────────────
    ckpt_dir  = os.path.join(wav2lip_path, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "wav2lip_gan.pth")
    if not os.path.isfile(ckpt_path):
        logger.info(
            "wav2lip_gan.pth not found — downloading from Google Drive (~436 MB)…"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            import gdown
            gdown.download(
                id=_WAV2LIP_GAN_GDRIVE_ID,
                output=ckpt_path,
                quiet=False,
            )
            logger.info("Downloaded wav2lip_gan.pth → %s", ckpt_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download wav2lip_gan.pth: {exc}\n"
                "Manual download from the official Wav2Lip folder:\n"
                "  https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k\n"
                f"  → place wav2lip_gan.pth at {ckpt_path}"
            ) from exc

    # ── s3fd.pth (face detection) ────────────────────────────────────────────
    sfd_dir  = os.path.join(wav2lip_path, "face_detection", "detection", "sfd")
    sfd_path = os.path.join(sfd_dir, "s3fd.pth")
    if not os.path.isfile(sfd_path):
        logger.info("s3fd.pth not found — downloading face detection model (~85 MB)…")
        os.makedirs(sfd_dir, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(_S3FD_URL, sfd_path)
            logger.info("Downloaded s3fd.pth → %s", sfd_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download s3fd.pth: {exc}\n"
                f"Manual download: {_S3FD_URL}\n"
                f"  → place at {sfd_path}"
            ) from exc


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
    Return True only when Wav2Lip is properly installed with all checkpoints.
    The pipeline can run without it (passthrough mode), so this is checked
    only to surface status in the UI.
    """
    inference_py = os.path.join(wav2lip_path, "inference.py")
    ckpt_path    = os.path.join(wav2lip_path, checkpoint)
    sfd_path     = os.path.join(wav2lip_path, "face_detection", "detection", "sfd", "s3fd.pth")
    return (
        os.path.isfile(inference_py)
        and os.path.isfile(ckpt_path)
        and os.path.isfile(sfd_path)
    )


def is_wav2lip_repo_present(wav2lip_path: str = WAV2LIP_REPO) -> bool:
    """Return True if the Wav2Lip repo is cloned (inference.py exists), regardless of checkpoints."""
    return os.path.isfile(os.path.join(wav2lip_path, "inference.py"))


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
        face_detection_batch_size: int = 1,
        wav2lip_batch_size: int = 8,
        resize_factor: int = 1,
        pads: tuple = (0, 10, 0, 0),
        debug_dir: Optional[str] = None,
    ):
        self.wav2lip_path = wav2lip_path
        self.checkpoint = os.path.join(wav2lip_path, checkpoint)
        self.face_batch = face_detection_batch_size
        self.wav2lip_batch = wav2lip_batch_size
        self.resize_factor = resize_factor
        self.pads = pads
        self.debug_dir = debug_dir

        # Auto-download checkpoints if Wav2Lip repo is present but weights are missing
        if os.path.isfile(os.path.join(wav2lip_path, "inference.py")):
            try:
                _ensure_wav2lip_checkpoints(wav2lip_path)
            except Exception as exc:
                logger.warning("Wav2Lip checkpoint download failed: %s", exc)

        self._available = is_wav2lip_available(wav2lip_path, checkpoint)

        if not self._available:
            logger.warning(
                "Wav2Lip not ready at '%s' — lip sync will use PASSTHROUGH mode "
                "(audio-swap only, no facial animation).",
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
        # Free any GPU memory left over from earlier pipeline stages
        # (OpenVoice, Resemblyzer, Whisper) before loading Wav2Lip models.
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                free_mb = torch.cuda.mem_get_info()[0] / 1024 ** 2
                logger.info("Wav2Lip pre-run GPU free: %.0f MB", free_mb)
        except Exception:
            pass

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = fd.FaceAlignment(fd.LandmarksType._2D, flip_input=False, device=device)

        # Process one frame at a time to avoid loading all frames into RAM at once.
        # For a 5-min 1080p video sampled every 0.5s this would be ~3.5 GB otherwise.
        debug_out = None
        if self.debug_dir:
            debug_out = os.path.join(self.debug_dir, "face_detection")
            os.makedirs(debug_out, exist_ok=True)
            logger.info("Face-detection debug frames will be saved to: %s", debug_out)

        indices   = []
        has_face  = []
        n_sampled = 0
        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            # detector expects (N, H, W, C); pass single frame, discard immediately
            preds = detector.get_detections_for_batch(np.array([frame_rgb]))
            detected = preds[0] is not None
            indices.append(idx)
            has_face.append(detected)
            n_sampled += 1

            # ── debug: save annotated frame every 3 s ──────────────────
            if debug_out is not None:
                ts_sec = idx / fps
                # Only write one frame per 3-second window to keep output manageable
                if int(ts_sec) % 3 == 0 and (n_sampled == 1 or int((idx - step) / fps) % 3 != 0):
                    annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    if detected:
                        # preds[0] shape: (N, 5) or (N, 4) — [x1,y1,x2,y2(,conf)]
                        for det in np.atleast_2d(preds[0]):
                            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                annotated, "FACE",
                                (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                            )
                    else:
                        cv2.putText(
                            annotated, "NO FACE",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                        )
                    fname = os.path.join(debug_out, f"frame_{idx:06d}_t{ts_sec:.0f}s.jpg")
                    cv2.imwrite(fname, annotated)
            # ────────────────────────────────────────────────────────────

            del frame_rgb

        cap.release()
        del detector
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Face detection: %d sample frames at step=%d", n_sampled, step)

        if not indices:
            # No frames readable — treat entire video as no-face
            return fps, [{"start": 0.0, "end": dur, "has_face": False}]

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

        Long clips are split into ≤MAX_CHUNK_SECS sub-chunks before being
        passed to inference.py, so that inference.py never has to hold more
        than ~2 GB of frames in RAM at once (720p × 30 s × 25 fps ≈ 2 GB).
        All sub-chunk results are concatenated back into one output.

        All I/O goes through safe (no-space) paths inside *tmpdir*.
        """
        # Maximum seconds per sub-chunk fed to inference.py.
        # 720 p × 30 s × 25 fps × 3 ch ≈ 2.07 GB — safe for most machines.
        MAX_CHUNK_SECS = 30

        v_ext = os.path.splitext(video_path)[1] or ".mp4"
        a_ext = os.path.splitext(audio_path)[1] or ".wav"

        safe_video = os.path.join(tmpdir, f"face_{idx}{v_ext}")
        safe_audio = os.path.join(tmpdir, f"audio_{idx}{a_ext}")

        if os.path.abspath(video_path) != os.path.abspath(safe_video):
            shutil.copy2(video_path, safe_video)
        if os.path.abspath(audio_path) != os.path.abspath(safe_audio):
            shutil.copy2(audio_path, safe_audio)

        # ── 1. Pre-scale to ≤720p ────────────────────────────────────────────
        # Wav2Lip resizes every face crop to 96×96 internally, so >720p input
        # gives no quality gain but uses 2-4× more RAM per frame.
        _MAX_HEIGHT = 720
        try:
            _probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=height",
                 "-of", "default=noprint_wrappers=1:nokey=1", safe_video],
                capture_output=True, timeout=10,
            )
            _vid_h = int(_probe.stdout.strip())
        except Exception:
            _vid_h = 9999

        if _vid_h > _MAX_HEIGHT:
            _scaled = os.path.join(tmpdir, f"face_{idx}_scaled.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", safe_video,
                 "-vf", f"scale=-2:{_MAX_HEIGHT}",
                 "-c:v", "libx264", "-preset", "fast", "-an", _scaled],
                check=True, capture_output=True, timeout=300,
            )
            safe_video = _scaled
            logger.info(
                "Wav2Lip clip[%d]: pre-scaled to %dp (was %dpx tall)",
                idx, _MAX_HEIGHT, _vid_h,
            )

        # ── 2. Get clip duration ─────────────────────────────────────────────
        try:
            _dp = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", safe_video],
                capture_output=True, timeout=10,
            )
            clip_dur = float(_dp.stdout.strip())
        except Exception:
            clip_dur = 0.0

        # ── 3. Single-pass fast path ─────────────────────────────────────────
        if clip_dur <= MAX_CHUNK_SECS:
            safe_output = os.path.join(tmpdir, f"out_{idx}.mp4")
            self._run_inference(safe_video, safe_audio, safe_output, tmpdir, idx, sub=None)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            shutil.move(safe_output, output_path)
            logger.info("Wav2Lip clip[%d] wrote: %s", idx, output_path)
            return

        # ── 4. Chunked path: split → infer → concatenate ─────────────────────
        n_chunks = int(np.ceil(clip_dur / MAX_CHUNK_SECS))
        logger.info(
            "Wav2Lip clip[%d]: %.1f s clip → splitting into %d × %ds sub-chunks",
            idx, clip_dur, n_chunks, MAX_CHUNK_SECS,
        )

        chunk_outputs = []
        for ci in range(n_chunks):
            t_start = ci * MAX_CHUNK_SECS
            t_dur   = min(MAX_CHUNK_SECS, clip_dur - t_start)

            cv = os.path.join(tmpdir, f"chunk_{idx}_{ci}_v.mp4")
            ca = os.path.join(tmpdir, f"chunk_{idx}_{ci}_a.wav")
            co = os.path.join(tmpdir, f"chunk_{idx}_{ci}_out.mp4")

            # Cut video chunk (re-encode so Wav2Lip gets clean keyframes)
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", f"{t_start:.3f}", "-t", f"{t_dur:.3f}",
                 "-i", safe_video,
                 "-c:v", "libx264", "-preset", "fast", "-an", cv],
                check=True, capture_output=True, timeout=300,
            )
            # Cut matching audio chunk
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", f"{t_start:.3f}", "-t", f"{t_dur:.3f}",
                 "-i", safe_audio,
                 "-ar", "16000", "-ac", "1", ca],
                check=True, capture_output=True, timeout=300,
            )

            self._run_inference(cv, ca, co, tmpdir, idx, sub=ci)
            chunk_outputs.append(co)
            logger.info(
                "  chunk %d/%d [%.1f–%.1fs] done",
                ci + 1, n_chunks, t_start, t_start + t_dur,
            )

        # Concatenate all chunks into final output
        concat_file = os.path.join(tmpdir, f"concat_{idx}.txt")
        with open(concat_file, "w") as f:
            for p in chunk_outputs:
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
                f"ffmpeg chunk concat failed:\n{r.stderr.decode()[-2000:]}"
            )
        logger.info("Wav2Lip clip[%d] wrote (chunked): %s", idx, output_path)

    def _run_inference(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        tmpdir: str,
        idx: int,
        sub,          # chunk index or None
    ) -> None:
        """Call inference.py for a single (already-sized) video + audio pair."""
        pad_str = [str(p) for p in self.pads]
        tag = f"{idx}" if sub is None else f"{idx}c{sub}"
        cmd = [
            sys.executable,
            os.path.join(self.wav2lip_path, "inference.py"),
            "--checkpoint_path", self.checkpoint,
            "--face",            video_path,
            "--audio",           audio_path,
            "--outfile",         output_path,
            "--face_det_batch_size", str(self.face_batch),
            "--wav2lip_batch_size",  str(self.wav2lip_batch),
            "--resize_factor",       str(self.resize_factor),
            "--pads",                *pad_str,
        ]
        logger.info("Wav2Lip infer[%s] cmd: %s", tag, " ".join(cmd))
        proc = subprocess.run(
            cmd, cwd=self.wav2lip_path,
            capture_output=True, timeout=3600,
            encoding="utf-8", errors="replace",
        )
        if proc.stdout:
            logger.info("Wav2Lip stdout[%s]:\n%s", tag, proc.stdout[-2000:])
        if proc.stderr:
            logger.info("Wav2Lip stderr[%s]:\n%s", tag, proc.stderr[-2000:])
        if proc.returncode != 0:
            raise RuntimeError(
                f"Wav2Lip exited with code {proc.returncode}:\n{proc.stderr[-2000:]}"
            )

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
