import torch
import os
import io
import soundfile as sf
import subprocess

from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

text = "सभी को नमस्कार, आपका मेरे चैनल में स्वागत है"

# Generate 
out_wav = "test_raw_xtts.wav"
ref_wav = "/home/aditya-gaur/BTP/output/0_150__Lecture 1 _ Introduction/demucs_sep/htdemucs/0_150__Lecture 1 _ Introduction/vocals.wav_trim_xtts.wav"

tts.tts_to_file(text=text, file_path=out_wav, speaker_wav=ref_wav, language="hi")

# Let's also do rubberband
subprocess.run(["ffmpeg", "-y", "-i", out_wav, "-af", "rubberband=tempo=1.3:formant=preserved:transients=smooth", "test_rb.wav"], capture_output=True)

# Let's do atempo
subprocess.run(["ffmpeg", "-y", "-i", out_wav, "-af", "atempo=1.3", "test_atempo.wav"], capture_output=True)

print("Check the sizes or listen to test_raw_xtts.wav, test_rb.wav, test_atempo.wav")
