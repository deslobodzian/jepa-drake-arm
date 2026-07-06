"""Reward / progress signal demo: V-JEPA2 goal similarity as a dense reward.

One reference episode ("demonstration") of a reach-and-hold task defines the
goal embedding. Four test rollouts are then scored by sliding-window goal
similarity:
  nominal     - same task, different randomization -> reward rises to ~goal
  undershoot  - weak reach (40% amplitude)         -> plateaus in between
  stalled     - freezes mid-task (fault)           -> flatlines below goal
  wrong task  - wrist_twist instead of reach       -> stays low
A usable RL reward must rank these correctly and rise monotonically for the
nominal rollout.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from jepa_drake_arm import reward as rw
from jepa_drake_arm.embedder import VJepa2Embedder
from jepa_drake_arm.motions import reach_hold, sample_params, stall_after, wrist_twist
from jepa_drake_arm.sim import run_episode
from jepa_drake_arm.visualize import save_video

STALL_TIME = 1.6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--model", default=None, help="HF model id (default: embedder default)")
    parser.add_argument("--out", default="output/reward")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(100)

    ref_params = sample_params(rng)
    nominal_params = sample_params(rng)
    undershoot_params = dataclasses.replace(sample_params(rng), amp_scale=0.4)
    stalled_params = sample_params(rng)
    wrong_params = sample_params(rng)

    rollouts = {
        "nominal": (reach_hold, nominal_params),
        "undershoot": (reach_hold, undershoot_params),
        f"stalled at {STALL_TIME}s": (stall_after(reach_hold, STALL_TIME),
                                      stalled_params),
        "wrong task": (wrist_twist, wrong_params),
    }

    print("simulating reference (demonstration) episode...")
    ref_frames = run_episode(reach_hold, ref_params,
                             duration=args.duration, fps=args.fps)
    save_video(out / "reference.mp4", ref_frames, fps=args.fps)

    embedder = VJepa2Embedder(**({"model_name": args.model} if args.model else {}))
    print(f"V-JEPA2 loaded on {embedder.device}")
    goal_emb = rw.goal_embedding(embedder, ref_frames, window=args.window)

    curves, results = {}, {}
    for name, (motion, params) in rollouts.items():
        frames = run_episode(motion, params,
                             duration=args.duration, fps=args.fps)
        save_video(out / f"{name.split()[0]}.mp4", frames, fps=args.fps)
        times, rewards = rw.progress_signal(
            embedder, frames, goal_emb, args.fps, window=args.window)
        curves[name] = (times, rewards)
        results[name] = {"final_reward": round(float(rewards[-1]), 4),
                         "max_reward": round(float(rewards.max()), 4)}
        print(f"{name}: final reward {rewards[-1]:.4f}")

    results["nominal_monotonicity_spearman"] = round(
        rw.monotonicity(*curves["nominal"]), 3)
    rw.plot_reward_curves(curves, out / "reward_curves.png",
                          stall_time=STALL_TIME)
    np.savez(out / "curves.npz",
             **{name: np.stack(c) for name, c in curves.items()})
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== goal-similarity reward ===")
    for name, r in results.items():
        print(f"{name}: {r}")
    print(f"artifacts in {out}/")


if __name__ == "__main__":
    main()
