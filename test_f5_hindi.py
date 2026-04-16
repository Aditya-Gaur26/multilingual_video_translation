from f5_tts.api import F5TTS
import torchaudio
import torch

tts = F5TTS(device="cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", tts.device)

ref_file = "/home/aditya-gaur/BTP/output/output/output.wav"
ref_text = "Good morning. So we will start this course with a brief introduction to programming languages."
gen_text = "नमस्ते दुनिया, मुझे उम्मीद है कि यह आवाज स्पष्ट है।"

wav, sr, _ = tts.infer(
    ref_file=ref_file,
    ref_text=ref_text,
    gen_text=gen_text,
)

torchaudio.save("/home/aditya-gaur/BTP/output/output/testhindi_f5.wav", torch.tensor(wav).unsqueeze(0), sr)
print("SUCCESS: Generated Hindi")
