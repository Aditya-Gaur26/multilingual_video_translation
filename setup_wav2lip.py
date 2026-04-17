"""
setup_wav2lip.py
================
One-time setup for the Wav2Lip submodule.
Run this after cloning the repo (or after `git submodule update --init`):

    python setup_wav2lip.py

What it does
------------
1. Applies the librosa-compatibility patch to Wav2Lip/audio.py
   (librosa ≥0.10 requires keyword args for filters.mel)
2. Downloads the two required model weights (~500 MB total):
     • Wav2Lip/checkpoints/wav2lip_gan.pth   (416 MB)
     • Wav2Lip/face_detection/detection/sfd/s3fd.pth  (86 MB)
"""

from __future__ import annotations
import os, sys, urllib.request, hashlib, shutil

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WAV2LIP   = os.path.join(REPO_ROOT, "Wav2Lip")

# ── 1. Apply audio.py patch ────────────────────────────────────────────────

AUDIO_PY = os.path.join(WAV2LIP, "audio.py")
if not os.path.isfile(AUDIO_PY):
    sys.exit(
        "ERROR: Wav2Lip/audio.py not found.\n"
        "Run: git submodule update --init --recursive"
    )

with open(AUDIO_PY, "r", encoding="utf-8") as f:
    src = f.read()

OLD = "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,"
NEW = "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"

if OLD in src:
    patched = src.replace(OLD, NEW)
    with open(AUDIO_PY, "w", encoding="utf-8") as f:
        f.write(patched)
    print("✓ Patched Wav2Lip/audio.py (librosa keyword args)")
elif NEW in src:
    print("✓ Wav2Lip/audio.py already patched")
else:
    print("⚠ Could not find expected line in audio.py — check manually")

# ── 2. Download model weights ───────────────────────────────────────────────

MODELS = [
    {
        "name":  "wav2lip_gan.pth",
        "url":   "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8dvkNr3DTNPkCRg?e=CdW6FH",
        "dest":  os.path.join(WAV2LIP, "checkpoints", "wav2lip_gan.pth"),
        "size_mb": 416,
    },
    {
        "name":  "s3fd.pth",
        "url":   "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "dest":  os.path.join(WAV2LIP, "face_detection", "detection", "sfd", "s3fd.pth"),
        "size_mb": 86,
    },
]

def _download(url: str, dest: str, name: str, expected_mb: int) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest):
        actual_mb = os.path.getsize(dest) / (1024 * 1024)
        if actual_mb > expected_mb * 0.9:
            print(f"✓ {name} already present ({actual_mb:.0f} MB)")
            return
        else:
            print(f"⚠ {name} exists but looks truncated ({actual_mb:.0f} MB) — re-downloading")
            os.remove(dest)

    print(f"  Downloading {name} ({expected_mb} MB) … ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        actual_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"done ({actual_mb:.0f} MB)")
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        print(f"  Please download manually:\n    URL: {url}\n    Dest: {dest}")

print("\nDownloading model weights (this only runs once):")
for m in MODELS:
    _download(m["url"], m["dest"], m["name"], m["size_mb"])

print("\nSetup complete. Run the app with: streamlit run app.py")
