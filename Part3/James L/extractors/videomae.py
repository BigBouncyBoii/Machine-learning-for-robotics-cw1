# CODE MAP of extractors/videomae.py   (tree node: frozen VideoMAE video encoder)
#   Settings        model, input size, ImageNet normalisation, cache file
#   Encoder         centre square crop, resize to 224x224; frozen VideoMAE on all 16 frames at once
#   Cache           encode all 160 clips once (no labels) -> cache/videomae_base.npz
#   Feature sets    "videomae": one 768-number vector per clip; load() -> {name: X (160, 768)}
#   AI use          Claude Opus 5.5 (claude-opus-5-5), default setting
#
# VIDEO encoder: it sees the 16 frames together, in order. Its tokens are 2-frame x 16x16-pixel blocks,
# so they describe motion as well as appearance. Tree node: "does the motion of the squeeze matter?"
# (was encoders/videomae.py and extract_features.py)

# %% Settings
import time

import cv2
import numpy as np

import common

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"   # ViT-B, 16 frames at 224x224, ~86M parameters,
SIZE = 224                                               # self-supervised then fine-tuned on 400 human actions
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CACHE = common.CACHE_DIR / "videomae_base.npz"


# %% Encoder
def load_encoder():
    """Frozen VideoMAE -> encode(16 frames) -> {'vec': (768,), 'time': (8, 768)} for one clip."""
    import torch
    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID).eval()   # frozen, forward pass only

    @torch.inference_mode()
    def encode(frames):
        h, w = frames[0].shape[:2]
        s = min(h, w)                                    # 480x854 portrait -> middle 480x480
        top, left = (h - s) // 2, (w - s) // 2
        x = np.stack([cv2.resize(f[top:top + s, left:left + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                      for f in frames])
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        h = model.videomae(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2)[None]).last_hidden_state
        t = h.shape[1] // (SIZE // 16) ** 2                                   # 8 time steps of 2 frames
        return {"vec": model.fc_norm(h.mean(1))[0].numpy(),                  # what its Kinetics classifier sees
                "time": h[0].reshape(t, -1, h.shape[-1]).mean(1).numpy()}

    return encode


# %% Cache
def load_cache(clips):
    """{'vec': (160, 768), 'time': (160, 8, 768)} in clips order; built on the first call (~10 min CPU)."""
    if not CACHE.exists():
        encode, arrays, t0 = load_encoder(), {}, time.time()
        for i, video in enumerate(clips["video"]):
            for k, v in encode(common.sample_frames(video, 16)[0]).items():
                arrays.setdefault(k, []).append(v)
            if (i + 1) % 20 == 0 or i + 1 == len(clips):
                print(f"  VideoMAE: {i + 1}/{len(clips)} clips ({time.time() - t0:.0f} s)")
        CACHE.parent.mkdir(exist_ok=True)
        np.savez_compressed(CACHE, **{k: np.stack(v) for k, v in arrays.items()},
                            clip=np.asarray(clips["file"], dtype=str))
    z = np.load(CACHE)
    row = {c: i for i, c in enumerate(z["clip"])}
    order = [row[f] for f in clips["file"]]                  # match by file name, not by row order
    return {k: z[k][order] for k in ("vec", "time") if k in z.files}


# %% Feature sets
# name -> (frame count, pooling: arrays with a clip axis -> (n_clips, 768)), same shape as dinov2.POOL
POOL = {"videomae": (16, lambda f: f["vec"])}                # one vector for the whole clip
FEATURE_SETS = list(POOL)


def load(clips, names=FEATURE_SETS):
    """{feature set name: X (n_clips, 768)}, rows in clips order."""
    f = load_cache(clips)
    return {s: POOL[s][1](f) for s in names}
