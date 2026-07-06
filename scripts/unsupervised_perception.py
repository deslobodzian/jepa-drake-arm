"""Perception WITHOUT labels: discover motion structure from embeddings alone.

Pipeline (no ground truth anywhere):
  1. k-means over V-JEPA2 episode embeddings, k chosen by silhouette score
  2. medoid of each discovered cluster -> nearest-neighbor retrieval panel
Ground-truth class names are revealed only at the end, to verify what the
clustering found (ARI / NMI / purity / contingency table).

Requires output/embeddings.npz from run_experiment.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from jepa_drake_arm import unsupervised as u
from jepa_drake_arm.evaluate import plot_pca_scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="output/embeddings.npz")
    parser.add_argument("--out", default="output/unsupervised")
    args = parser.parse_args()

    data = np.load(args.embeddings)
    embeddings, labels, files = (data["embeddings"], list(data["labels"]),
                                 list(data["files"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- label-free pipeline -------------------------------------------------
    scores = u.silhouette_by_k(embeddings)
    k = max(scores, key=scores.get)
    ids = u.cluster(embeddings, k)
    medoids = u.cluster_medoids(embeddings, ids)

    u.plot_silhouette_curve(scores, k, out / "silhouette_vs_k.png")
    plot_pca_scatter(embeddings, [f"cluster {c}" for c in ids],
                     out / "pca_by_cluster.png")
    u.retrieval_panel(embeddings, files, medoids, out / "retrieval.png")

    # --- post-hoc verification (labels revealed only here) -------------------
    verdict = u.verify_against_labels(ids, labels)
    u.plot_contingency(ids, labels, out / "contingency.png")

    results = {"chosen_k": int(k),
               "silhouette_by_k": {str(kk): round(v, 4) for kk, v in scores.items()},
               **verdict}
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print("=== perception without labels ===")
    print(f"discovered k = {k} clusters "
          f"(true number of motion classes: {len(set(labels))})")
    print(f"adjusted Rand index vs ground truth: {verdict['adjusted_rand_index']:.3f}"
          " (1.0 = perfect, 0.0 = random)")
    print(f"NMI: {verdict['nmi']:.3f}   cluster purity: "
          f"{verdict['cluster_purity']:.1%}")
    print(f"artifacts in {out}/")


if __name__ == "__main__":
    main()
