import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pose_diff(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Action that takes EE pose `start` to `end` (Meta's poses_to_diff)."""
    dxyz = end[:3] - start[:3]
    r_start = Rotation.from_euler("xyz", start[3:6])
    r_end = Rotation.from_euler("xyz", end[3:6])
    dtheta = (r_end * r_start.inv()).as_euler("xyz")
    return np.concatenate([dxyz, dtheta, end[6:] - start[6:]])


def apply_action(pose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """New pose after `action` (Meta's compute_new_pose). [B, 1, 7] each."""
    p = pose[:, 0].cpu().numpy()
    a = action[:, 0].cpu().numpy()
    new_xyz = p[:, :3] + a[:, :3]
    new_angle = np.stack([
        (Rotation.from_euler("xyz", da) * Rotation.from_euler("xyz", th)
         ).as_euler("xyz")
        for th, da in zip(p[:, 3:6], a[:, 3:6])
    ])
    new_grip = np.clip(p[:, -1:] + a[:, -1:], 0.0, 1.0)
    new_pose = np.concatenate([new_xyz, new_angle, new_grip], axis=-1)
    return torch.from_numpy(new_pose).to(pose.device, pose.dtype)[:, None]


class VJepa2AC:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        encoder, predictor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant", trust_repo=True)
        self.encoder = encoder.to(self.device).eval()
        self.predictor = predictor.to(self.device).eval()
        self.crop = 256
        self.tokens_per_frame = (self.crop // encoder.patch_size) ** 2

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 -> normalized [3, crop, crop] float."""
        t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(self.crop, self.crop),
                          mode="bilinear", align_corners=False)[0]
        return (t - IMAGENET_MEAN) / IMAGENET_STD

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) uint8 -> layer-normed tokens [1, T*tokens_per_frame, D].

        Each frame is duplicated to fill one tubelet, matching Meta's
        per-frame encoding for the AC predictor.
        """
        t = torch.stack([self._preprocess(f) for f in frames]).to(self.device)
        clips = t.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [T, 3, 2, H, W]
        h = self.encoder(clips)  # [T, tokens_per_frame, D]
        h = h.reshape(1, -1, h.size(-1))
        return F.layer_norm(h, (h.size(-1),))

    @torch.no_grad()
    def predict_next(self, z_ctx: torch.Tensor, actions: torch.Tensor,
                     states: torch.Tensor) -> torch.Tensor:
        """Predict the next frame's tokens.

        z_ctx: [B, T*tokens_per_frame, D]; actions/states: [B, T, 7].
        Returns layer-normed [B, tokens_per_frame, D].
        """
        z = self.predictor(z_ctx, actions, states)[:, -self.tokens_per_frame:]
        return F.layer_norm(z, (z.size(-1),))

    def energy(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """Mean L1 distance per batch element. [B, N, D] x [., N, D] -> [B]."""
        return (z_pred - z_target).abs().mean(dim=(-2, -1))


@torch.no_grad()
def cem_plan(ac: VJepa2AC, z_ctx: torch.Tensor, pose: np.ndarray,
             z_goal: torch.Tensor, rollout: int = 2, samples: int = 200,
             topk: int = 20, cem_steps: int = 5, maxnorm: float = 0.075,
             seed: int = 0) -> np.ndarray:
    """Cross-Entropy Method over translation-only action sequences.

    Returns the planned action sequence [rollout, 7] (first action is the one
    to execute). Follows Meta's cem() but optimizes xyz only.
    """
    device = ac.device
    gen = torch.Generator(device=device).manual_seed(seed)
    mean = torch.zeros(rollout, 3, device=device)
    std = torch.full((rollout, 3), maxnorm, device=device)

    z0 = z_ctx.repeat(samples, 1, 1)
    s0 = torch.from_numpy(pose).float().to(device)[None, None].repeat(samples, 1, 1)
    z_goal_b = z_goal.repeat(samples, 1, 1)

    for _ in range(cem_steps):
        dxyz = torch.randn(samples, rollout, 3, generator=gen, device=device)
        dxyz = torch.clip(dxyz * std + mean, -maxnorm, maxnorm)
        actions = torch.cat([dxyz, torch.zeros(samples, rollout, 4, device=device)],
                            dim=-1)
        z_traj, s_traj = z0, s0
        for h in range(rollout):
            z_next = ac.predict_next(z_traj, actions[:, :h + 1], s_traj)
            z_traj = torch.cat([z_traj, z_next], dim=1)
            s_next = apply_action(s_traj[:, -1:], actions[:, h:h + 1])
            s_traj = torch.cat([s_traj, s_next], dim=1)
        e = ac.energy(z_traj[:, -ac.tokens_per_frame:], z_goal_b)
        best = e.topk(topk, largest=False).indices
        mean = 0.85 * dxyz[best].mean(dim=0) + 0.15 * mean
        std = 0.25 * dxyz[best].std(dim=0) + 0.75 * std

    planned = torch.cat([mean, torch.zeros(rollout, 4, device=device)], dim=-1)
    return planned.cpu().numpy()
