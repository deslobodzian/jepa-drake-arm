"""Closed-loop MPC in Drake with the V-JEPA2-AC world model.

Loop: render -> encode -> CEM-plan translation actions against a goal image's
latent -> execute the first action via differential IK -> repeat. This is
Meta's zero-shot planning recipe, pointed at a Drake sim.

Outputs per step: EE distance to the goal position (ground truth, for
evaluation only — the planner sees pixels and proprio, never the goal pose).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from jepa_drake_arm.ac import VJepa2AC, cem_plan
from jepa_drake_arm.evaluate import CATEGORICAL, INK, _style_axes
from jepa_drake_arm.sim import ARMS, InteractiveArmSim
from jepa_drake_arm.visualize import save_video, side_by_side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="panda", choices=["panda", "iiwa"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--goal-delta", type=float, nargs=3,
                        default=[0.25, 0.0, -0.12],
                        help="EE goal displacement from home (m)")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--cem-steps", type=int, default=5)
    parser.add_argument("--rollout", type=int, default=2)
    parser.add_argument("--maxnorm", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="output/ac_mpc")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Render the goal image by driving a throwaway sim to the goal.
    goal_sim = InteractiveArmSim(arm=args.arm)
    for _ in range(4):
        goal_sim.move_ee_delta(np.asarray(args.goal_delta) / 4, duration=0.5)
    goal_sim.advance(0.5)
    goal_frame = goal_sim.render()
    goal_xyz = goal_sim.ee_pose()[:3]

    ac = VJepa2AC()
    z_goal = ac.encode_frames(goal_frame[None])

    sim = InteractiveArmSim(arm=args.arm)
    frames = [sim.render()]
    dists = [float(np.linalg.norm(sim.ee_pose()[:3] - goal_xyz))]
    print(f"start EE-goal distance: {dists[0]:.3f} m")

    for step in range(args.steps):
        z_ctx = ac.encode_frames(frames[-1][None])
        plan = cem_plan(ac, z_ctx, sim.ee_pose(), z_goal,
                        rollout=args.rollout, samples=args.samples,
                        cem_steps=args.cem_steps, maxnorm=args.maxnorm,
                        seed=args.seed + step)
        action = plan[0, :3]
        sim.move_ee_delta(action, duration=0.5)
        frames.append(sim.render())
        dists.append(float(np.linalg.norm(sim.ee_pose()[:3] - goal_xyz)))
        print(f"step {step + 1}: action {np.round(action, 3)}, "
              f"EE-goal distance {dists[-1]:.3f} m")

    # Rollout video with the goal frame alongside.
    rollout_frames = np.stack(frames)
    goal_track = np.broadcast_to(goal_frame, rollout_frames.shape)
    video = side_by_side(rollout_frames, np.ascontiguousarray(goal_track),
                         left_label="MPC rollout (1 frame per control step)",
                         right_label="goal image")
    save_video(out / f"mpc_{args.arm}.mp4", video, fps=2.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.set_axisbelow(True)
    ax.grid(color=INK["grid"], lw=0.7)
    ax.plot(range(len(dists)), dists, color=CATEGORICAL[0], lw=2, marker="o",
            markersize=6)
    ax.set_xlabel("control step", color=INK["secondary"], fontsize=9)
    ax.set_ylabel("EE distance to goal (m)", color=INK["secondary"], fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title(f"V-JEPA2-AC MPC toward a goal image ({args.arm})",
                 color=INK["primary"], fontsize=11, pad=12)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / f"mpc_distance_{args.arm}.png", facecolor="white")

    results = {"arm": args.arm, "goal_delta": args.goal_delta,
               "distances_m": [round(d, 4) for d in dists],
               "start_m": round(dists[0], 4), "final_m": round(dists[-1], 4),
               "improved": dists[-1] < dists[0]}
    (out / f"results_{args.arm}.json").write_text(json.dumps(results, indent=2))
    print(f"final EE-goal distance: {dists[-1]:.3f} m "
          f"({'improved' if results['improved'] else 'did not improve'} "
          f"from {dists[0]:.3f} m)")
    print(f"artifacts in {out}/")


if __name__ == "__main__":
    main()
