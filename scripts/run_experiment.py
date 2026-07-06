"""End-to-end experiment: Drake sim -> V-JEPA2 embeddings -> evaluation.

Simulates N randomized episodes of each motion class, embeds each rendered
video with V-JEPA2, then checks that the embeddings separate the classes
(leave-one-out 1-NN accuracy, within- vs between-class similarity, plots).
"""
import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from jepa_drake_arm import evaluate
from jepa_drake_arm.embedder import VJepa2Embedder
from jepa_drake_arm.motions import MOTION_CLASSES, sample_params
from jepa_drake_arm.sim import run_episode
from jepa_drake_arm.visualize import save_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=6, help="episodes per class")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=None, help="HF model id (default: embedder default)")
    parser.add_argument("--out", default="output")
    parser.add_argument("--skip-sim", action="store_true",
                        help="reuse videos already in <out>/videos")
    args = parser.parse_args()

    out = Path(args.out)
    video_dir = out / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. Simulate and render.
    episodes = []  # (class_name, video_path)
    for name in MOTION_CLASSES:
        for ep in range(args.episodes):
            path = video_dir / f"{name}_{ep:02d}.mp4"
            params = sample_params(rng)  # keep RNG stream identical either way
            if not (args.skip_sim and path.exists()):
                t0 = time.time()
                frames = run_episode(name, params,
                                     duration=args.duration, fps=args.fps)
                save_video(path, frames, fps=args.fps)
                print(f"simulated {path.name} ({time.time() - t0:.1f}s)")
            episodes.append((name, path))

    # 2. Embed with V-JEPA2.
    embedder = VJepa2Embedder(**({"model_name": args.model} if args.model else {}))
    print(f"V-JEPA2 loaded on {embedder.device}")
    labels, embeddings = [], []
    for name, path in episodes:
        frames = iio.imread(path, plugin="pyav")
        embeddings.append(embedder.embed(np.asarray(frames)))
        labels.append(name)
        print(f"embedded {path.name}")
    embeddings = np.stack(embeddings)
    np.savez(out / "embeddings.npz", embeddings=embeddings, labels=labels,
             files=[str(path) for _, path in episodes])

    # 3. Evaluate.
    sim_matrix = evaluate.cosine_similarity_matrix(embeddings)
    accuracy = evaluate.loo_nearest_neighbor_accuracy(embeddings, labels)
    stats = evaluate.class_similarity_stats(sim_matrix, labels)
    from sklearn.metrics import silhouette_score
    silhouette = float(silhouette_score(embeddings, labels, metric="cosine"))

    results = {
        "episodes_per_class": args.episodes,
        "classes": list(MOTION_CLASSES),
        "loo_1nn_accuracy": accuracy,
        "chance_accuracy": 1.0 / len(MOTION_CLASSES),
        "silhouette_cosine": silhouette,
        **stats,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    evaluate.plot_similarity_matrix(sim_matrix, labels, out / "similarity_matrix.png")
    evaluate.plot_pca_scatter(embeddings, labels, out / "pca_scatter.png")

    print("\n=== V-JEPA2 on Drake iiwa sim ===")
    print(f"episodes: {len(labels)} ({args.episodes} x {len(MOTION_CLASSES)} classes)")
    print(f"leave-one-out 1-NN accuracy: {accuracy:.1%} "
          f"(chance {results['chance_accuracy']:.0%})")
    print(f"mean within-class similarity:  {stats['mean_within_class']:.3f}")
    print(f"mean between-class similarity: {stats['mean_between_class']:.3f}")
    print(f"silhouette score (cosine): {silhouette:.3f}")
    print(f"artifacts in {out}/")


if __name__ == "__main__":
    main()
