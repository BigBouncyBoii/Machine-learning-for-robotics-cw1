# CODE MAP of encoders/dinov2.py
#   Settings       model, input size, ImageNet normalisation
#   Preprocess     resize whole frame to 224x392 (no crop)
#   Encode         frozen DINOv2 on each frame separately: CLS vector and mean of patch vectors
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# IMAGE encoder: every frame is encoded on its own, so the order of frames is lost unless
# the pooling in common.FEATURES uses it (e.g. squeeze frame minus rest frame).
# Output per clip: cls (16, 384), patch (16, 384)

# %% Settings
import cv2
import numpy as np
import torch

MODEL_ID = "facebook/dinov2-small"   # ViT-S/14, 384-dim features, ~22M parameters
FILE = "dinov2_small"
IN_W, IN_H = 224, 392                # clips are portrait 480x854; both sides are multiples of the 14-px patch
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet statistics used to train DINOv2
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# %% Preprocess
def to_tensor(frames):
    """List of HxWx3 uint8 RGB frames -> normalised float tensor (n, 3, IN_H, IN_W)."""
    x = np.stack([cv2.resize(f, (IN_W, IN_H), interpolation=cv2.INTER_AREA) for f in frames])
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)


# %% Encode
def load():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID).eval()   # frozen: never trained, only run forward

    @torch.inference_mode()
    def encode(frames):
        h = model(pixel_values=to_tensor(frames)).last_hidden_state   # (16, 1 + n_patches, 384), after final LayerNorm
        return {"cls": h[:, 0].numpy(), "patch": h[:, 1:].mean(1).numpy()}

    return encode
