# CODE MAP of encoders/videomae.py
#   Settings       model, input size, ImageNet normalisation
#   Preprocess     centre square crop (hand and object are always in the middle band), resize to 224x224
#   Encode         frozen VideoMAE on all 16 frames at once: one clip vector and one vector per time step
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# VIDEO encoder: it sees the 16 frames together, in order. Its tokens are 2-frame x 16x16-pixel blocks,
# so they describe motion as well as appearance. Tree node: "does the motion of the squeeze matter?"
# Output per clip: vec (768,)    mean of all tokens + the model's final norm (what its Kinetics classifier sees)
#                  time (8, 768) mean of the tokens of each 2-frame time step, for nodes that use the order

# %% Settings
import cv2
import numpy as np
import torch

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"   # ViT-B, 16 frames at 224x224, ~86M parameters,
FILE = "videomae_base"                                   # self-supervised then fine-tuned on 400 human actions
SIZE = 224                                               # fixed by the model's position encoding
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# %% Preprocess
def to_tensor(frames):
    """List of 16 HxWx3 uint8 RGB frames -> normalised float tensor (1, 16, 3, SIZE, SIZE)."""
    h, w = frames[0].shape[:2]
    s = min(h, w)                                        # 480x854 portrait -> middle 480x480
    top, left = (h - s) // 2, (w - s) // 2
    x = np.stack([cv2.resize(f[top:top + s, left:left + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                  for f in frames])
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)[None]


# %% Encode
def load():
    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID).eval()   # frozen, forward pass only

    @torch.inference_mode()
    def encode(frames):
        h = model.videomae(pixel_values=to_tensor(frames)).last_hidden_state   # (1, 8*14*14, 768), time-major
        t = h.shape[1] // (SIZE // 16) ** 2                                       # 8 time steps of 2 frames
        return {"vec": model.fc_norm(h.mean(1))[0].numpy(),
                "time": h[0].reshape(t, -1, h.shape[-1]).mean(1).numpy()}

    return encode
