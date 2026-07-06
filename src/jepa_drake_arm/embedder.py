import numpy as np
import torch

DEFAULT_MODEL = "facebook/vjepa2-vitg-fpc64-384"  # ViT-giant, 384px
VITL_MODEL = "facebook/vjepa2-vitl-fpc64-256"  # smaller/faster alternative


class VJepa2Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        from transformers import AutoModel, AutoVideoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device).eval()
        self.frames_per_clip = getattr(self.model.config, "frames_per_clip", 64)

    def _resample(self, frames: np.ndarray) -> np.ndarray:
        """Uniformly resample a (T, H, W, 3) clip to the model's clip length."""
        idx = np.round(
            np.linspace(0, len(frames) - 1, self.frames_per_clip)
        ).astype(int)
        return frames[idx]

    @torch.no_grad()
    def embed(self, frames: np.ndarray) -> np.ndarray:
        """Embed a video (T, H, W, 3) uint8 -> L2-normalized 1D feature vector.

        Pools the V-JEPA2 encoder's patch tokens by mean.
        """
        clip = self._resample(frames)
        video = torch.from_numpy(clip).permute(0, 3, 1, 2)  # T, C, H, W
        inputs = self.processor(video, return_tensors="pt").to(self.device)
        if hasattr(self.model, "get_vision_features"):
            features = self.model.get_vision_features(**inputs)
        else:
            features = self.model(**inputs).last_hidden_state
        emb = features.mean(dim=1).float().cpu().numpy()[0]
        return emb / np.linalg.norm(emb)
