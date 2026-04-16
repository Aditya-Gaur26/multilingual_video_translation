from f5_tts.infer.utils_infer import load_model, infer_process
from f5_tts.model import DiT

# Fast quick test
model = load_model(model_cls=DiT, model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), ckpt_path="")
# Actually F5 TTS needs huggingface weights downloaded.
