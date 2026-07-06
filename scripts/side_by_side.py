"""Render sim episodes and compose side-by-side videos: sim | V-JEPA2 features."""
import argparse
from pathlib import Path

import numpy as np

from jepa_drake_arm.embedder import VJepa2Embedder
from jepa_drake_arm.motions import MOTION_CLASSES, sample_params
from jepa_drake_arm.sim import run_episode
from jepa_drake_arm.visualize import patch_feature_video, save_video, side_by_side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="*", default=list(MOTION_CLASSES),
                        choices=list(MOTION_CLASSES))
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None, help="HF model id (default: embedder default)")
    parser.add_argument("--out", default="output/side_by_side")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    embedder = VJepa2Embedder(**({"model_name": args.model} if args.model else {}))
    print(f"V-JEPA2 loaded on {embedder.device}")

    for name in args.classes:
        frames = run_episode(name, sample_params(rng), duration=args.duration,
                             fps=args.fps, image_size=args.image_size)
        clip, vis = patch_feature_video(embedder, frames)
        video = side_by_side(clip, vis)
        path = out / f"{name}.mp4"
        save_video(path, video, fps=args.fps)
        print(f"wrote {path} {video.shape}")


if __name__ == "__main__":
    main()
