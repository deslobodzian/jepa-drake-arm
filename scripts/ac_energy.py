"""Validate the V-JEPA2-AC action-conditioned predictor on Drake, zero-shot.

A scripted end-effector move produces an episode with known EE displacement.
We then ask the world model: which candidate action best explains the change
from frame A to frame B? (energy = L1 between predicted and actual latent).

Diagnostics reported (see README for interpretation):
- action sensitivity: how much predictions differ across actions
- domain floor: |pred(no-op) - context| — how far predictions drift from
  reality regardless of action (high = visual domain gap)
- energy landscape over a 3-D action grid + true-action percentile

V-JEPA2-AC was trained on Franka arms (DROID), so --arm panda is the default;
--arm iiwa shows the additional embodiment gap.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from jepa_drake_arm.ac import VJepa2AC, pose_diff
from jepa_drake_arm.evaluate import INK, SEQUENTIAL_BLUE, _style_axes
from jepa_drake_arm.sim import InteractiveArmSim
from jepa_drake_arm.visualize import save_video


def probe_episode(arm: str, steps: int = 6, delta=(0.05, 0.0, -0.02)):
    sim = InteractiveArmSim(arm=arm)
    frames, poses = [sim.render()], [sim.ee_pose()]
    for _ in range(steps):
        sim.move_ee_delta(list(delta), duration=0.5)
        frames.append(sim.render())
        poses.append(sim.ee_pose())
    return np.stack(frames), np.stack(poses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="panda", choices=["panda", "iiwa"])
    parser.add_argument("--grid-n", type=int, default=9)
    parser.add_argument("--grid-size", type=float, default=0.12)
    parser.add_argument("--out", default="output/ac_energy")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, poses = probe_episode(args.arm)
    save_video(out / f"probe_{args.arm}.mp4", frames, fps=2.0)
    i0, i1 = 0, 2
    true_a = pose_diff(poses[i0], poses[i1])

    ac = VJepa2AC()
    z = ac.encode_frames(frames[[i0, i1]])
    z_ctx, z_tgt = z[:, :ac.tokens_per_frame], z[:, ac.tokens_per_frame:]
    state = torch.from_numpy(poses[i0]).float().to(ac.device)[None, None]

    def pred(dxyz):
        a = torch.zeros(1, 1, 7, device=ac.device)
        a[0, 0, :3] = torch.tensor(np.asarray(dxyz, dtype=np.float32))
        return ac.predict_next(z_ctx, a, state)

    def E(a, b):
        return float(ac.energy(a, b))

    p_no = pred([0, 0, 0])
    diagnostics = {
        "action_sensitivity |pred(+x)-pred(-x)|":
            round(E(pred([0.1, 0, 0]), pred([-0.1, 0, 0])), 4),
        "domain_floor |pred(no-op)-context|": round(E(p_no, z_ctx), 4),
        "|context-target|": round(E(z_ctx, z_tgt), 4),
        "|pred(true)-target|": round(E(pred(true_a[:3]), z_tgt), 4),
        "|pred(no-op)-target|": round(E(p_no, z_tgt), 4),
        "|pred(-true)-target|": round(E(pred(-true_a[:3]), z_tgt), 4),
    }

    # Energy over a 3-D grid of candidate translations.
    axis = np.linspace(-args.grid_size, args.grid_size, args.grid_n)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    cand = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float32)
    energies = []
    with torch.no_grad():
        for chunk in np.array_split(np.arange(len(cand)), 8):
            B = len(chunk)
            a = torch.zeros(B, 1, 7, device=ac.device)
            a[:, 0, :3] = torch.from_numpy(cand[chunk]).to(ac.device)
            zp = ac.predict_next(z_ctx.repeat(B, 1, 1), a, state.repeat(B, 1, 1))
            energies.append(ac.energy(zp, z_tgt.repeat(B, 1, 1)).cpu().numpy())
    e = np.concatenate(energies)
    energy = e.reshape(args.grid_n, args.grid_n, args.grid_n)

    amin = np.unravel_index(np.argmin(energy), energy.shape)
    best = np.array([axis[a] for a in amin])
    t = true_a[:3]
    nearest = np.argmin(np.linalg.norm(cand - t, axis=1))
    results = {
        "arm": args.arm,
        "true_dxyz": [round(v, 4) for v in t],
        "argmin_dxyz": [round(v, 4) for v in best],
        "true_action_energy_percentile": round(100 * float((e < e[nearest]).mean()), 1),
        "true_beats_reversed": diagnostics["|pred(true)-target|"]
            < diagnostics["|pred(-true)-target|"],
        **diagnostics,
    }
    (out / f"results_{args.arm}.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    for ax, (i, j, k), (li, lj) in [(axes[0], (0, 1, 2), ("dx", "dy")),
                                    (axes[1], (0, 2, 1), ("dx", "dz"))]:
        im = ax.imshow(energy.min(axis=k).T, origin="lower",
                       cmap=SEQUENTIAL_BLUE,
                       extent=[axis[0], axis[-1], axis[0], axis[-1]])
        ax.scatter([t[i]], [t[j]], marker="*", s=220, color="white",
                   edgecolors=INK["primary"], linewidths=1.2, zorder=4,
                   label="true action")
        ax.scatter([best[i]], [best[j]], marker="o", s=90, facecolors="none",
                   edgecolors="#e34948", linewidths=2, zorder=4,
                   label="energy minimum")
        ax.set_xlabel(f"{li} (m)", color=INK["secondary"], fontsize=9)
        ax.set_ylabel(f"{lj} (m)", color=INK["secondary"], fontsize=9)
        _style_axes(ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle(f"V-JEPA2-AC energy landscape, {args.arm} in Drake "
                 "(min-projected slices)", color=INK["primary"], fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"energy_landscape_{args.arm}.png", facecolor="white")

    print(json.dumps(results, indent=2))
    print(f"artifacts in {out}/")


if __name__ == "__main__":
    main()
