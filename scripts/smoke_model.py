"""Load V-JEPA2 and embed a synthetic clip; sanity-check the model pipeline."""
import numpy as np

from jepa_drake_arm.embedder import VJepa2Embedder


def main():
    embedder = VJepa2Embedder()
    print(f"model loaded on {embedder.device}, "
          f"frames_per_clip={embedder.frames_per_clip}")

    rng = np.random.default_rng(0)
    video_a = rng.integers(0, 255, (64, 256, 256, 3), dtype=np.uint8)
    video_b = rng.integers(0, 255, (64, 256, 256, 3), dtype=np.uint8)

    emb_a = embedder.embed(video_a)
    emb_a2 = embedder.embed(video_a)
    emb_b = embedder.embed(video_b)
    print(f"embedding shape: {emb_a.shape}")
    print(f"self-similarity (must be 1.0): {np.dot(emb_a, emb_a2):.4f}")
    print(f"cross-similarity (noise vs noise): {np.dot(emb_a, emb_b):.4f}")


if __name__ == "__main__":
    main()
