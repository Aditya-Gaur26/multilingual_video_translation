import librosa
import numpy as np
import os

paths = [
    os.popen("ls -t /home/aditya-gaur/BTP/output/0_150__Lecture*/*_trim_xtts.wav | head -1").read().strip(),
    "/home/aditya-gaur/BTP/output/0_150__Lecture 1 _ Introduction/tts_segments_hi/segment_000.wav",
    "/home/aditya-gaur/BTP/output/0_150__Lecture 1 _ Introduction/0_150__Lecture 1 _ Introduction_hi_aligned.mp3",
    "/home/aditya-gaur/BTP/output/0_150__Lecture 1 _ Introduction/0_150__Lecture 1 _ Introduction_hi_mixed.mp3"
]

print("--- AUDIO ANALYSIS ---")
for p in paths:
    if not os.path.exists(p) or not p:
        print(f"[{os.path.basename(p) if p else 'None'}] ❌ NOT FOUND")
        continue

    try:
        y, sr = librosa.load(p, sr=None)
        max_amp = np.max(np.abs(y))
        mean_amp = np.mean(np.abs(y))
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Check clipping
        clipping_pct = np.sum(np.abs(y) >= 0.99) / len(y) * 100
        
        print(f"[{os.path.basename(p)}] SR: {sr}Hz | Dur: {duration:.2f}s | Max Amp: {max_amp:.4f} | Mean Amp: {mean_amp:.4f} | Clipping: {clipping_pct:.2f}%")
        
    except Exception as e:
        print(f"[{os.path.basename(p)}] ❌ ERROR: {e}")

