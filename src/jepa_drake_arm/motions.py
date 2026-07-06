"""
Each motion class is a function q(t) -> 7-vector. Episodes within a class are
randomized (amplitude / frequency / phase) so the embedding evaluation sees
intra-class variety, not identical videos.
"""
from __future__ import annotations

import dataclasses

import numpy as np

NUM_JOINTS = 7

Q_HOME = np.array([0.0, 0.5, 0.0, -1.7, 0.0, 1.0, 0.0])


@dataclasses.dataclass
class MotionParams:
    amp_scale: float = 1.0
    freq_scale: float = 1.0
    phase: float = 0.0
    home_offset: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(NUM_JOINTS)
    )


def sample_params(rng: np.random.Generator) -> MotionParams:
    return MotionParams(
        amp_scale=rng.uniform(0.8, 1.2),
        freq_scale=rng.uniform(0.85, 1.15),
        phase=rng.uniform(0.0, 0.4),
        home_offset=rng.uniform(-0.05, 0.05, NUM_JOINTS),
    )


def _out_and_back(t: float, duration: float) -> float:
    """Smooth 0 -> 1 -> 0 profile over the episode."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * t / duration)


def base_wave(t: float, duration: float, p: MotionParams) -> np.ndarray:
    q = Q_HOME + p.home_offset
    q = q.copy()
    q[0] += 0.9 * p.amp_scale * np.sin(2.0 * np.pi * 0.5 * p.freq_scale * t + p.phase)
    return q


def reach_forward(t: float, duration: float, p: MotionParams) -> np.ndarray:
    q_target = np.array([0.0, 1.15, 0.0, -0.55, 0.0, 0.75, 0.0])
    s = _out_and_back(t, duration) * p.amp_scale
    s = np.clip(s, 0.0, 1.0)
    return (1.0 - s) * (Q_HOME + p.home_offset) + s * q_target


def lift_overhead(t: float, duration: float, p: MotionParams) -> np.ndarray:
    q_target = np.array([0.0, -0.75, 0.0, -1.0, 0.0, 1.2, 0.0])
    s = _out_and_back(t, duration) * p.amp_scale
    s = np.clip(s, 0.0, 1.0)
    return (1.0 - s) * (Q_HOME + p.home_offset) + s * q_target


def wrist_twist(t: float, duration: float, p: MotionParams) -> np.ndarray:
    q = (Q_HOME + p.home_offset).copy()
    w = 2.0 * np.pi * 0.6 * p.freq_scale
    q[4] += 0.9 * p.amp_scale * np.sin(w * t + p.phase)
    q[5] += 0.5 * p.amp_scale * np.sin(w * t + p.phase + 0.7)
    q[6] += 1.4 * p.amp_scale * np.sin(w * t + p.phase + 1.3)
    return q


def reach_hold(t: float, duration: float, p: MotionParams) -> np.ndarray:
    """One-way reach: smoothly extend forward, then hold the goal pose.

    Unlike `reach_forward` (out-and-back), this has a distinct goal state,
    which makes it suitable for goal-conditioned reward experiments.
    """
    q_target = np.array([0.0, 1.15, 0.0, -0.55, 0.0, 0.75, 0.0])
    u = np.clip(t / (0.6 * duration), 0.0, 1.0)
    s = (0.5 - 0.5 * np.cos(np.pi * u)) * p.amp_scale
    s = np.clip(s, 0.0, 1.0)
    return (1.0 - s) * (Q_HOME + p.home_offset) + s * q_target


def stall_after(fn, t_freeze: float):
    """Fault model: the motion freezes at `t_freeze` and holds that pose."""
    return lambda t, duration, p: fn(min(t, t_freeze), duration, p)


MOTION_CLASSES = {
    "base_wave": base_wave,
    "reach_forward": reach_forward,
    "lift_overhead": lift_overhead,
    "wrist_twist": wrist_twist,
}


def make_joint_trajectory(motion, duration: float, p: MotionParams,
                          sample_hz: float = 100.0):
    """Sample the motion into a cubic trajectory usable by the ID controller.

    `motion` is a MOTION_CLASSES key or a callable q(t, duration, params).
    """
    from pydrake.trajectories import PiecewisePolynomial

    fn = motion if callable(motion) else MOTION_CLASSES[motion]
    times = np.arange(0.0, duration + 1.0 / sample_hz, 1.0 / sample_hz)
    samples = np.column_stack([fn(t, duration, p) for t in times])
    return PiecewisePolynomial.CubicShapePreserving(
        times, samples, zero_end_point_derivatives=True
    )
