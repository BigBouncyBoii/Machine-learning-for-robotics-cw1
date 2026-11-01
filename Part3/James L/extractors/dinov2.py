# CODE MAP of extractors/dinov2.py   (tree nodes: frozen DINOv2 image encoder, pooled over frames)
#   Settings        model, input size, ImageNet normalisation, cache files
#   Encoder         frozen DINOv2 on each frame separately: CLS vector and mean of patch vectors (384 each)
#   Cache           encode all 160 clips once (no labels) -> cache/dinov2_small.npz (16 frames),
#                   cache/dinov2_small_8f.npz (8 frames re-sampled from the video)
#   Feature sets    POOL: how the per-frame vectors become one vector per clip (one tree node each);
#                   load() -> {name: X (160, 384)}
#   AI use          Claude Opus 5.5 (claude-opus-5-5), default setting
#
# IMAGE encoder: every frame is encoded on its own, so the order of frames is lost unless the pooling uses it.
# The encoder is frozen (never trained); only the classifier on top is trained. 160 clips are far too few
# to train a vision model, so a pretrained one turns each frame into a feature vector.
# (was encoders/dinov2.py, extract_features.py, common.FEATURES and frames_node.py)

# %% Settings
import time

import cv2
import numpy as np

import common

MODEL_ID = "facebook/dinov2-small"   # ViT-S/14, 384-dim features, ~22M parameters
IN_W, IN_H = 224, 392                # clips are portrait 480x854; both sides are multiples of the 14-px patch
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet statistics used to train DINOv2
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CACHE = {16: common.CACHE_DIR / "dinov2_small.npz", 8: common.CACHE_DIR / "dinov2_small_8f.npz"}


# %% Encoder
def load_encoder():
    """Frozen DINOv2 -> encode(frames) -> {'cls': (n, 384), 'patch': (n, 384)} for one clip's frames."""
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID).eval()   # frozen: never trained, only run forward

    @torch.inference_mode()
    def encode(frames):
        x = np.stack([cv2.resize(f, (IN_W, IN_H), interpolation=cv2.INTER_AREA) for f in frames])   # whole frame
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        h = model(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2)).last_hidden_state   # (n, 1 + patches, 384)
        return {"cls": h[:, 0].numpy(), "patch": h[:, 1:].mean(1).numpy()}

    return encode


# %% Cache
def load_cache(clips, n_frames):
    """{'cls': (160, n_frames, 384), ...} in clips order; built from the videos on the first call (~2 min CPU)."""
    path = CACHE[n_frames]
    if not path.exists():
        encode, arrays, t0 = load_encoder(), {}, time.time()
        for i, video in enumerate(clips["video"]):
            for k, v in encode(common.sample_frames(video, n_frames)[0]).items():
                arrays.setdefault(k, []).append(v)
            if (i + 1) % 20 == 0 or i + 1 == len(clips):
                print(f"  DINOv2, {n_frames} frames: {i + 1}/{len(clips)} clips ({time.time() - t0:.0f} s)")
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **{k: np.stack(v) for k, v in arrays.items()},
                            clip=np.asarray(clips["file"], dtype=str))
    z = np.load(path)
    row = {c: i for i, c in enumerate(z["clip"])}
    order = [row[f] for f in clips["file"]]                  # match by file name, not by row order
    return {k: z[k][order] for k in ("cls", "patch") if k in z.files}


# %% Feature sets
# name -> (frame count of the cache, pooling: arrays with a clip axis -> (n_clips, 384))
POOL = {
    "dinov2_cls_mean":     (16, lambda f: f["cls"].mean(axis=1)),             # mean CLS of the 16 frames (baseline)
    "dinov2_patch_mean":   (16, lambda f: f["patch"].mean(axis=1)),           # mean patch vector: local detail
    # fewer frames (subsets of the cached 16, evenly spaced): does averaging more frames help?
    "dinov2_1frame":       (16, lambda f: f["cls"][:, 8]),                    # middle frame only
    "dinov2_4frames":      (16, lambda f: f["cls"][:, ::4].mean(axis=1)),     # frames 0, 4, 8, 12
    "dinov2_8frames":      (16, lambda f: f["cls"][:, ::2].mean(axis=1)),     # every 2nd frame
    "dinov2_resampled_8":  (8, lambda f: f["cls"].mean(axis=1)),              # 8 frames re-sampled from the video
    # one phase of the grip: where is the signal?
    "dinov2_start":        (16, lambda f: f["cls"][:, :3].mean(axis=1)),      # first 3 frames (not gripping yet)
    "dinov2_middle":       (16, lambda f: f["cls"][:, 5:11].mean(axis=1)),    # middle 6 frames (squeeze / hold)
}
FEATURE_SETS = list(POOL)


def load(clips, names=FEATURE_SETS):
    """{feature set name: X (n_clips, 384)}, rows in clips order; each cache is read once."""
    caches = {n: load_cache(clips, n) for n in sorted({POOL[s][0] for s in names})}
    return {s: POOL[s][1](caches[POOL[s][0]]) for s in names}
