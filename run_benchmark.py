"""
Multi-seed KNN benchmark for saliencytools metrics.

Replicates the experiment in tutorial.ipynb with:
  - configurable number of seeds (default 10)
  - macro-F1 (consistent with paper; notebook used weighted)
  - k=20 prototypes per class (as stated in paper)
  - checkpoint/resume: safe to interrupt and re-run

Usage:
    python run_benchmark.py                        # 10 seeds -> results_seeds.json
    python run_benchmark.py --seeds 5              # 5 seeds
    python run_benchmark.py --k 5                  # k=5 prototypes -> results_k5.json
    python run_benchmark.py --out my_results.json  # custom output file
    python run_benchmark.py --resume               # skip already-completed (seed, metric, config) triples

Estimated runtime (single machine, CPU):
    Fast metrics (L2, cosine, MAE, MSE, Jaccard, Czek, SAR, KL): ~3s/config
    AUC-Judd: ~8s/config,  PSNR: ~12s/config,  Correlation: ~17s/config
    SSIM: ~63s/config,  EMD: ~73s/config
    Total for 10 seeds x 13 metrics x 8 configs ~ 4-5 hours
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import sobel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

from saliencytools.maskcompare import (
    clip_mask,
    normalize_mask_0_1,
    euclidean_distance,
    cosine_distance,
    mean_absolute_error,
    mean_squared_error,
    emd,
    correlation_distance,
    psnr,
    jaccard_distance,
    czenakowski_distance,
    sign_agreement_ratio,
    ssim,
    kl_divergence,
    auc_judd,
)

METRICS = [
    sign_agreement_ratio,
    ssim,
    mean_absolute_error,
    czenakowski_distance,
    jaccard_distance,
    cosine_distance,
    correlation_distance,
    euclidean_distance,
    mean_squared_error,
    psnr,
    emd,
    kl_divergence,
    auc_judd,
]

K_PROTOTYPES = 20       # prototypes per class (overridden by --k)
N_CLASSES    = 10
TEST_SAMPLES = 5000     # first N test images (after stratified split)
TRAIN_SEED   = 42       # fixed seed for train/test split (not varied)


# ── Data loading ────────────────────────────────────────────────────────────

def load_data():
    dataset = MNIST(root="data", train=True, download=True)
    X, y = np.array(dataset.data, dtype=np.float32) / 255.0, np.array(dataset.targets)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=TRAIN_SEED
    )

    # Centre and apply Sobel — mirrors tutorial.ipynb preprocessing
    mean = X_train.mean()
    X_train = np.array([sobel(img - mean) for img in X_train])
    X_test  = np.array([sobel(img - mean) for img in X_test])

    y_test  = y_test[:TEST_SAMPLES]
    X_test  = X_test[:TEST_SAMPLES]
    return X_train, y_train, X_test, y_test


def sample_prototypes(X_train, y_train, seed: int, k: int = K_PROTOTYPES) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prototypes = np.zeros((N_CLASSES, k, 28, 28), dtype=np.float32)
    for cls in range(N_CLASSES):
        idx = np.where(y_train == cls)[0]
        chosen = rng.choice(idx, size=k, replace=False)
        prototypes[cls] = X_train[chosen]
    return prototypes


# ── KNN classifier ──────────────────────────────────────────────────────────

def preprocess(images: np.ndarray, clip: bool, normalize: bool, sobel_f: bool) -> np.ndarray:
    out = images.copy()
    if sobel_f:
        out = np.array([sobel(img) for img in out])
    if clip:
        out = np.array([clip_mask(img) for img in out])
    if normalize:
        out = np.array([normalize_mask_0_1(img) for img in out])
    return out


def knn_predict(X_test, prototypes_prep, metric_fn) -> np.ndarray:
    """Predict class for each test image using 1-NN over prototypes."""
    preds = np.empty(len(X_test), dtype=int)
    for i, img in enumerate(X_test):
        # distances shape: (N_CLASSES, K_PROTOTYPES)
        dists = np.array([
            [metric_fn(img, proto) for proto in prototypes_prep[cls]]
            for cls in range(N_CLASSES)
        ])
        preds[i] = dists.min(axis=1).argmin()
    return preds


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def load_done(out_path: Path) -> set:
    """Return set of (seed, metric_name, clip, normalize, sobel) already saved."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["seed"], row["saliency_metric"],
                      row["clip"], row["normalize"], row["sobel"]))
    return done


def append_row(out_path: Path, row: dict):
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds (default: 10)")
    parser.add_argument("--k", type=int, default=K_PROTOTYPES,
                        help="Prototypes per class (default: 20)")
    parser.add_argument("--out", default=None,
                        help="Output file (JSONL). Defaults to results_seeds.json or results_k{k}.json")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (seed, metric, config) triples already in output file")
    args = parser.parse_args()

    k = args.k
    default_out = "results_seeds.json" if k == K_PROTOTYPES else f"results_k{k}.json"
    out_path = Path(args.out if args.out else default_out)
    seeds = list(range(args.seeds))
    configs = list(product([True, False], repeat=3))  # (clip, normalize, sobel)

    total = len(seeds) * len(METRICS) * len(configs)
    done_set = load_done(out_path) if args.resume else set()
    skipped = len(done_set)

    print(f"Loading MNIST...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"Seeds: {seeds}  Metrics: {len(METRICS)}  Configs: {len(configs)}  k={k}")
    print(f"Total runs: {total}  Already done: {skipped}  Remaining: {total - skipped}")
    print(f"Output: {out_path}\n")

    completed = skipped
    for seed in seeds:
        prototypes = sample_prototypes(X_train, y_train, seed, k=k)

        for clip, normalize, sobel_f in configs:
            # Preprocess prototypes once per config (shared across metrics)
            protos_prep = preprocess(
                prototypes.reshape(-1, 28, 28), clip, normalize, sobel_f
            ).reshape(N_CLASSES, k, 28, 28)

            # Preprocess test set once per config
            X_test_prep = preprocess(X_test, clip, normalize, sobel_f)

            for metric in METRICS:
                key = (seed, metric.__name__, clip, normalize, sobel_f)
                if key in done_set:
                    continue

                start = time.time()
                preds = knn_predict(X_test_prep, protos_prep, metric)
                elapsed = time.time() - start
                f1 = f1_score(y_test, preds, average="macro")

                row = {
                    "seed":            seed,
                    "saliency_metric": metric.__name__,
                    "f1_score":        round(f1, 9),
                    "elapsed_time":    round(elapsed, 6),
                    "clip":            clip,
                    "normalize":       normalize,
                    "sobel":           sobel_f,
                }
                append_row(out_path, row)
                done_set.add(key)
                completed += 1

                pct = 100 * completed / total
                print(
                    f"[{completed}/{total} {pct:.0f}%] "
                    f"seed={seed} {metric.__name__:30s} "
                    f"C={int(clip)} N={int(normalize)} S={int(sobel_f)} "
                    f"F1={f1:.3f}  t={elapsed:.1f}s"
                )

    print(f"\nDone. Results written to {out_path}")


if __name__ == "__main__":
    main()
