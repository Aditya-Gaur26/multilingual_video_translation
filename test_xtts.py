import torch
from TTS.api import TTS
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

text = "सभी को नमस्कार, संरचना विश्लेषण में आपका स्वागत है।"
tts.tts_to_file(
    text=text,
    file_path="xtts_test_pure.wav",
    speaker_wav="/home/aditya-gaur/BTP/output/0_150__Lecture 1 _ Introduction/demucs_sep/htdemucs/0_150__Lecture 1 _ Introduction/vocals.wav",
    language="hi"
)
print("Done")
