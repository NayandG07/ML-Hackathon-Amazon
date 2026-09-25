"""
run_pipeline.py
===============
End-to-end orchestration script for the Business Entity Resolution pipeline.

Stages:
  0. Setup and environment checks
  1. Preprocessing (preprocessing.py)
  2. Blocking / Candidate Generation (blocking.py)
  3. Feature Engineering (feature_engineering.py)
  4. Training (train.py) — skipped if --inference-only
  5. Inference (inference.py)
  6. Validation (validate_local.py) — if --validate
  7. Submission validation (utils/validate_submission.py)

Usage — full training + test inference:
    python run_pipeline.py

Usage — inference only (model already trained):
    python run_pipeline.py --inference-only

Usage — validate on held-out train portion:
    python run_pipeline.py --validate
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("run_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent.parent.parent   # student_resource/


def run(cmd: list[str], cwd: Path = ROOT_DIR) -> None:
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), check=True)
    if result.returncode != 0:
        log.error(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full ER pipeline.")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training; load existing model.")
    parser.add_argument("--validate", action="store_true",
                        help="Run local validation after inference.")
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-dir", default="output/models")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=50,
                        help="TF-IDF top-K candidates per S1 entity.")
    parser.add_argument("--no-minhash", action="store_true",
                        help="Disable MinHash LSH blocking (faster but lower recall).")
    args = parser.parse_args()

    py = sys.executable
    src = str(SRC_DIR)

    # ----------------------------------------------------------------
    # Stage 1: Preprocessing
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 1: Preprocessing")
    log.info("=" * 60)
    run([py, f"{src}/preprocessing.py",
         "--train-dir", "dataset/train",
         "--test-dir", "dataset/test",
         "--out-dir", args.processed_dir])

    # ----------------------------------------------------------------
    # Stage 2: Training blocking + features + model
    # ----------------------------------------------------------------
    if not args.inference_only:
        # --- Blocking on training data (for features) ---
        log.info("=" * 60)
        log.info("STAGE 2a: Blocking (training set)")
        log.info("=" * 60)
        cmd = [py, f"{src}/blocking.py",
               "--processed-dir", args.processed_dir,
               "--output-dir", f"{args.output_dir}/train_blocking",
               "--split", "train",
               "--top-k", str(args.top_k),
               "--artifacts-dir", f"{args.output_dir}/artifacts"]
        if args.no_minhash:
            cmd.append("--no-minhash")
        run(cmd)

        # --- Feature engineering (training set) ---
        log.info("=" * 60)
        log.info("STAGE 2b: Feature Engineering (training set)")
        log.info("=" * 60)
        run([py, f"{src}/feature_engineering.py",
             "--processed-dir", args.processed_dir,
             "--candidate-file", f"{args.output_dir}/train_blocking/candidate_pairs.tsv",
             "--ground-truth", "dataset/train/train_ground_truth.tsv",
             "--split", "train",
             "--output-file", f"{args.output_dir}/features_train.parquet"])

        # --- Training ---
        log.info("=" * 60)
        log.info("STAGE 2c: Training LightGBM")
        log.info("=" * 60)
        run([py, f"{src}/train.py",
             "--features-file", f"{args.output_dir}/features_train.parquet",
             "--model-dir", args.model_dir,
             "--n-estimators", str(args.n_estimators)])

    # ----------------------------------------------------------------
    # Stage 3: Test set blocking + inference
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 3: Blocking (test set)")
    log.info("=" * 60)
    cmd = [py, f"{src}/blocking.py",
           "--processed-dir", args.processed_dir,
           "--output-dir", args.output_dir,
           "--split", "test",
           "--top-k", str(args.top_k),
           "--artifacts-dir", f"{args.output_dir}/artifacts"]
    if args.no_minhash:
        cmd.append("--no-minhash")
    run(cmd)

    log.info("=" * 60)
    log.info("STAGE 4: Inference")
    log.info("=" * 60)
    run([py, f"{src}/inference.py",
         "--processed-dir", args.processed_dir,
         "--candidate-file", f"{args.output_dir}/candidate_pairs.tsv",
         "--model-dir", args.model_dir,
         "--output-dir", args.output_dir])

    # ----------------------------------------------------------------
    # Stage 4: Optional local validation
    # ----------------------------------------------------------------
    if args.validate:
        log.info("=" * 60)
        log.info("STAGE 5: Local Validation")
        log.info("=" * 60)
        run([py, f"{src}/validate_local.py",
             "--matching-file", f"{args.output_dir}/matching_results.tsv",
             "--candidate-file", f"{args.output_dir}/candidate_pairs.tsv",
             "--ground-truth", "dataset/train/train_ground_truth.tsv"])

    # ----------------------------------------------------------------
    # Stage 5: Submission format validation
    # ----------------------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 6: Format Validation")
    log.info("=" * 60)
    run([py, "utils/validate_submission.py",
         "--matching", f"{args.output_dir}/matching_results.tsv",
         "--candidate", f"{args.output_dir}/candidate_pairs.tsv",
         "--test-dir", "dataset/test"])

    log.info("=" * 60)
    log.info("✅ Pipeline complete! Ready to submit.")
    log.info(f"   Upload: {args.output_dir}/matching_results.tsv")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
