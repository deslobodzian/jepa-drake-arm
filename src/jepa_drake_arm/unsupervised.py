from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from .evaluate import CATEGORICAL, INK, SEQUENTIAL_BLUE, _style_axes


def silhouette_by_k(embeddings: np.ndarray, k_range=range(2, 9)) -> dict[int, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in k_range:
        ids = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)
        scores[k] = float(silhouette_score(embeddings, ids, metric="cosine"))
    return scores


def cluster(embeddings: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)


def cluster_medoids(embeddings: np.ndarray, ids: np.ndarray) -> list[int]:
    """Index of the most central episode of each cluster (by cosine)."""
    medoids = []
    for c in np.unique(ids):
        members = np.flatnonzero(ids == c)
        x = embeddings[members]
        sim = x @ x.T
        medoids.append(int(members[np.argmax(sim.sum(axis=1))]))
    return medoids


def verify_against_labels(ids: np.ndarray, labels: list[str]) -> dict:
    """Post-hoc check only — the labels play no role in the pipeline."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    labels = np.asarray(labels)
    purity = np.mean([
        np.max(np.unique(labels[ids == c], return_counts=True)[1]) / np.sum(ids == c)
        for c in np.unique(ids)
    ])
    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels, ids)),
        "nmi": float(normalized_mutual_info_score(labels, ids)),
        "cluster_purity": float(purity),
    }


def plot_silhouette_curve(scores: dict[int, float], chosen_k: int, path) -> None:
    ks = sorted(scores)
    vals = [scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.set_axisbelow(True)
    ax.grid(color=INK["grid"], lw=0.7)
    ax.plot(ks, vals, color=CATEGORICAL[0], lw=2, marker="o", markersize=7)
    ax.scatter([chosen_k], [scores[chosen_k]], s=160, facecolors="none",
               edgecolors=INK["primary"], linewidths=1.6, zorder=4)
    ax.annotate(f"best k = {chosen_k}", (chosen_k, scores[chosen_k]),
                textcoords="offset points", xytext=(12, 8),
                color=INK["primary"], fontsize=10)
    ax.set_xlabel("number of clusters k", color=INK["secondary"], fontsize=9)
    ax.set_ylabel("silhouette score (cosine)", color=INK["secondary"], fontsize=9)
    ax.set_title("Model selection without labels: silhouette vs k",
                 color=INK["primary"], fontsize=11, pad=12)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_contingency(ids: np.ndarray, labels: list[str], path) -> None:
    labels = np.asarray(labels)
    classes = list(dict.fromkeys(labels))
    clusters = sorted(np.unique(ids))
    counts = np.array([[np.sum((ids == c) & (labels == cls)) for cls in classes]
                       for c in clusters])

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=150)
    im = ax.imshow(counts, cmap=SEQUENTIAL_BLUE, vmin=0)
    for i in range(len(clusters)):
        for j in range(len(classes)):
            v = counts[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                    color="white" if v > counts.max() * 0.55 else INK["primary"])
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(clusters)), [f"cluster {c}" for c in clusters])
    ax.set_xlabel("true motion class (revealed post-hoc)",
                  color=INK["secondary"], fontsize=9)
    ax.set_title("Discovered clusters vs ground truth",
                 color=INK["primary"], fontsize=11, pad=12)
    _style_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(colors=INK["secondary"], labelsize=8)
    cbar.outline.set_edgecolor(INK["grid"])
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def retrieval_panel(embeddings: np.ndarray, files: list[str],
                    medoids: list[int], path, n_neighbors: int = 3) -> None:
    """Image grid: each row = a cluster-medoid query + its nearest neighbors.

    Cells show each video's middle frame, captioned with its filename — the
    filename is the post-hoc reveal that retrieved episodes match the query's
    motion.
    """
    import imageio.v3 as iio
    from .visualize import _label

    sim = embeddings @ embeddings.T
    rows = []
    for q in medoids:
        order = np.argsort(-sim[q])
        picks = [q] + [i for i in order if i != q][:n_neighbors]
        cells = []
        for rank, i in enumerate(picks):
            frames = iio.imread(files[i], plugin="pyav")
            frame = frames[len(frames) // 2]
            caption = ("query: " if rank == 0 else f"#{rank}: ") + Path(files[i]).stem
            cells.append(np.concatenate([_label(frame.shape[1], caption), frame]))
        gap = np.full((cells[0].shape[0], 6, 3), 255, dtype=np.uint8)
        row = cells[0]
        for cell in cells[1:]:
            row = np.concatenate([row, gap, cell], axis=1)
        rows.append(row)
    gap = np.full((6, rows[0].shape[1], 3), 255, dtype=np.uint8)
    panel = rows[0]
    for row in rows[1:]:
        panel = np.concatenate([panel, gap, row])
    iio.imwrite(path, panel)
