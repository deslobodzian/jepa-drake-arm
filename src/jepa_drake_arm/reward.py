import numpy as np
from matplotlib import pyplot as plt

from .embedder import VJepa2Embedder
from .evaluate import CATEGORICAL, INK, _style_axes


def goal_embedding(embedder: VJepa2Embedder, reference_frames: np.ndarray,
                   window: int = 16) -> np.ndarray:
    """Embedding of the reference rollout's final `window` frames."""
    return embedder.embed(reference_frames[-window:])


def progress_signal(embedder: VJepa2Embedder, frames: np.ndarray,
                    goal_emb: np.ndarray, fps: float,
                    window: int = 16, stride: int = 2
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Reward r(t) = cosine(embed(last `window` frames at t), goal).

    Returns (times, rewards); times are the right edge of each window.
    """
    times, rewards = [], []
    for end in range(window, len(frames) + 1, stride):
        emb = embedder.embed(frames[end - window:end])
        rewards.append(float(emb @ goal_emb))
        times.append(end / fps)
    return np.array(times), np.array(rewards)


def monotonicity(times: np.ndarray, rewards: np.ndarray) -> float:
    """Spearman rank correlation of reward with time (1.0 = perfectly rising)."""
    from scipy.stats import spearmanr

    return float(spearmanr(times, rewards).statistic)


def plot_reward_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path,
                       stall_time: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    ax.set_axisbelow(True)
    ax.grid(color=INK["grid"], lw=0.7)
    for i, (name, (times, rewards)) in enumerate(curves.items()):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(times, rewards, color=color, lw=2, label=name)
    # Direct labels at line ends, staggered so near-equal finals don't collide.
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.05
    prev_y = None
    for name, (times, rewards) in sorted(
            curves.items(), key=lambda kv: -kv[1][1][-1]):
        y = rewards[-1] if prev_y is None else min(rewards[-1], prev_y - min_gap)
        prev_y = y
        ax.annotate(name, (times[-1], y),
                    textcoords="offset points", xytext=(6, 0),
                    color=INK["primary"], fontsize=9, va="center")
    if stall_time is not None:
        ax.axvline(stall_time, color=INK["secondary"], lw=1, ls=":")
        ax.annotate("stall", (stall_time, ymin),
                    textcoords="offset points", xytext=(4, 6),
                    color=INK["secondary"], fontsize=8)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.22)  # room for direct labels
    ax.set_xlabel("rollout time (s)", color=INK["secondary"], fontsize=9)
    ax.set_ylabel("cosine similarity to goal embedding",
                  color=INK["secondary"], fontsize=9)
    ax.set_title("V-JEPA2 goal-similarity reward over rollouts",
                 color=INK["primary"], fontsize=11, pad=12)
    legend = ax.legend(loc="upper left", frameon=True, fontsize=9)
    legend.get_frame().set_edgecolor(INK["grid"])
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
