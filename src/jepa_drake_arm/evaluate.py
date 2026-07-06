import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical palette (fixed slot order) and sequential blue ramp.
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
INK = {"primary": "#1a1a19", "secondary": "#5c5b54", "grid": "#e4e3dd"}


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    x = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return x @ x.T


def loo_nearest_neighbor_accuracy(embeddings: np.ndarray, labels: list[str]) -> float:
    """Leave-one-out 1-NN classification accuracy under cosine similarity."""
    sim = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argmax(sim, axis=1)
    labels = np.asarray(labels)
    return float(np.mean(labels[nearest] == labels))


def class_similarity_stats(sim: np.ndarray, labels: list[str]) -> dict:
    labels = np.asarray(labels)
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)
    return {
        "mean_within_class": float(sim[same & off_diag].mean()),
        "mean_between_class": float(sim[~same].mean()),
    }


def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_color(INK["grid"])
    ax.tick_params(colors=INK["secondary"], labelsize=9)


def plot_similarity_matrix(sim: np.ndarray, labels: list[str], path: str) -> None:
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    im = ax.imshow(sim, cmap=SEQUENTIAL_BLUE, vmin=sim.min(), vmax=1.0)

    # Class-boundary separators and centered class tick labels.
    boundaries = np.flatnonzero(labels[:-1] != labels[1:]) + 0.5
    for b in boundaries:
        ax.axhline(b, color="white", lw=2)
        ax.axvline(b, color="white", lw=2)
    edges = np.concatenate([[-0.5], boundaries, [len(labels) - 0.5]])
    centers = (edges[:-1] + edges[1:]) / 2
    class_names = [labels[int(np.ceil(e + 0.5))] for e in edges[:-1]]
    ax.set_xticks(centers, class_names, rotation=30, ha="right")
    ax.set_yticks(centers, class_names)
    _style_axes(ax)

    ax.set_title("V-JEPA2 embedding cosine similarity (episodes grouped by motion class)",
                 color=INK["primary"], fontsize=11, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(colors=INK["secondary"], labelsize=8)
    cbar.outline.set_edgecolor(INK["grid"])
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_pca_scatter(embeddings: np.ndarray, labels: list[str], path: str) -> None:
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2).fit_transform(embeddings)
    labels = np.asarray(labels)
    classes = list(dict.fromkeys(labels))  # first-appearance order

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    ax.set_axisbelow(True)
    ax.grid(color=INK["grid"], lw=0.7)
    for i, cls in enumerate(classes):
        pts = coords[labels == cls]
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.scatter(pts[:, 0], pts[:, 1], s=70, color=color, edgecolors="white",
                   linewidths=1.2, label=cls, zorder=3)
        centroid = pts.mean(axis=0)
        ax.annotate(cls, centroid, textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=9, color=INK["primary"], zorder=4)
    _style_axes(ax)
    ax.set_xlabel("PC 1", color=INK["secondary"], fontsize=9)
    ax.set_ylabel("PC 2", color=INK["secondary"], fontsize=9)
    ax.set_title("V-JEPA2 episode embeddings, PCA projection",
                 color=INK["primary"], fontsize=11, pad=12)
    legend = ax.legend(loc="best", frameon=True, fontsize=9)
    legend.get_frame().set_edgecolor(INK["grid"])
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
