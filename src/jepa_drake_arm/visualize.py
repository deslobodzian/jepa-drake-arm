import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .embedder import VJepa2Embedder


@torch.no_grad()
def patch_feature_video(embedder: VJepa2Embedder, frames: np.ndarray,
                        layer: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute a per-patch feature visualization for a video.

    Returns (clip, vis): the model-length resampled clip (T, H, W, 3) and a
    matching feature video of the same shape.

    Uses an intermediate encoder layer (default: a third of the way in): by
    the final layer, global attention has mixed the arm's motion into every
    token, so nothing localizes; early-to-middle layers cleanly silhouette
    the moving arm. Color = PCA of each patch's deviation from its temporal
    mean (feature identity), brightness = deviation magnitude (motion energy).
    """
    clip = embedder._resample(frames)
    video = torch.from_numpy(clip).permute(0, 3, 1, 2)
    inputs = embedder.processor(video, return_tensors="pt").to(embedder.device)
    out = embedder.model(**inputs, output_hidden_states=True)
    if layer is None:
        layer = max(1, len(out.hidden_states) // 3)
    features = out.hidden_states[layer]  # (1, N, D)

    cfg = embedder.model.config
    tubelet = getattr(cfg, "tubelet_size", 2)
    patch = getattr(cfg, "patch_size", 16)
    crop = getattr(cfg, "crop_size", 256)
    t_tokens = embedder.frames_per_clip // tubelet
    s_tokens = crop // patch

    grid = features[0].float().cpu().numpy().reshape(t_tokens, s_tokens, s_tokens, -1)

    # Deviation from each patch location's temporal mean = what changes.
    dev = grid - grid.mean(axis=0, keepdims=True)
    flat = dev.reshape(-1, dev.shape[-1])

    # Brightness: motion energy per patch.
    energy = np.linalg.norm(flat, axis=-1)
    alpha = np.clip(energy / (np.percentile(energy, 99) + 1e-8), 0.0, 1.0)
    # Cut background haze, then gamma-lift so slow motions stay visible.
    alpha = np.clip((alpha - 0.12) / 0.88, 0.0, 1.0) ** 0.6

    # Hue: PCA of the deviation -> 3 channels in [0, 1].
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    pca = flat @ vt[:3].T
    scale = np.percentile(np.abs(pca), 98, axis=0) + 1e-8
    color = np.clip(pca / scale, -1.0, 1.0) * 0.5 + 0.5

    rgb = (color * alpha[:, None] * 255).astype(np.uint8)
    rgb = rgb.reshape(t_tokens, s_tokens, s_tokens, 3)

    # Upsample each token frame to the clip resolution; each temporal token
    # covers `tubelet` consecutive frames.
    h, w = clip.shape[1:3]
    vis = np.empty_like(clip)
    for i in range(t_tokens):
        img = Image.fromarray(rgb[i]).resize((w, h), Image.BILINEAR)
        vis[i * tubelet:(i + 1) * tubelet] = np.asarray(img)
    return clip, vis


def _label(frame_width: int, text: str, height: int = 28) -> np.ndarray:
    """Render a one-line dark label bar of the given width."""
    bar = Image.new("RGB", (frame_width, height), (26, 26, 25))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((8, height // 2), text, fill=(235, 235, 230),
              font=font, anchor="lm")
    return np.asarray(bar)


def side_by_side(left: np.ndarray, right: np.ndarray,
                 left_label: str = "Drake sim",
                 right_label: str = "V-JEPA2 motion features (PCA)",
                 gap: int = 4) -> np.ndarray:
    """Compose two equal-shape videos horizontally with labels and a divider."""
    assert left.shape == right.shape, (left.shape, right.shape)
    t, h, w, _ = left.shape
    bar = np.concatenate([
        _label(w, left_label),
        np.zeros((28, gap, 3), dtype=np.uint8),
        _label(w, right_label),
    ], axis=1)
    divider = np.zeros((t, h, gap, 3), dtype=np.uint8)
    body = np.concatenate([left, divider, right], axis=2)
    return np.concatenate([np.broadcast_to(bar, (t, *bar.shape)), body], axis=1)


def save_video(path, frames: np.ndarray, fps: float, quality: int = 9) -> None:
    """Write an MP4 at high quality (imageio-ffmpeg quality scale 0-10)."""
    import imageio.v2 as iio2

    iio2.mimwrite(path, list(frames), fps=fps, quality=quality,
                  macro_block_size=1)
