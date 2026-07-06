"""Render one short episode per motion class; sanity-check the Drake pipeline."""
import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from jepa_drake_arm.motions import MOTION_CLASSES, sample_params
from jepa_drake_arm.sim import run_episode
from jepa_drake_arm.visualize import save_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--out", default="output/smoke")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for name in MOTION_CLASSES:
        frames = run_episode(name, sample_params(rng),
                             duration=args.duration, fps=args.fps)
        save_video(out / f"{name}.mp4", frames, fps=args.fps)
        iio.imwrite(out / f"{name}_frame0.png", frames[0])
        print(f"{name}: {frames.shape} dtype={frames.dtype} "
              f"mean={frames.mean():.1f} std={frames.std():.1f}")
    print(f"wrote videos to {out}/")


if __name__ == "__main__":
    main()
