"""
run_v4_pipeline.py
==================
Full pipeline to generate the v4 submission after:
  1. New sharded blocking (output/candidate_pairs.tsv is rebuilt)
  2. GPU embeddings for test (output/embeddings/test_*.npy exist)

Steps this script runs:
  A. Feature engineering on new candidate pairs
  B. Score embedding similarity for new candidates
  C. Merge embedding feature into features parquet
  D. Run inference with lower threshold (use model from training)
  E. Validate submission
  F. Generate zip

Usage:
    python code/business_entity_resolution/src/run_v4_pipeline.py
"""

import sys
import os
import subprocess
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_v4_pipeline")

SRC = Path("code/business_entity_resolution/src")
OUT = Path("output")
PROCESSED = Path("dataset/processed")

PYTHON = sys.executable


def run(cmd, desc, log_file=None):
    log.info(f"► {desc}")
    log.info(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    if log_file:
        with open(log_file, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log.error(f"  FAILED (exit {result.returncode}) after {elapsed:.0f}s — check {log_file}")
        raise RuntimeError(f"Step failed: {desc}")
    log.info(f"  OK in {elapsed:.0f}s")
    return result


def main():
    os.chdir(Path(__file__).parent.parent.parent.parent)  # go to D:\OpenSource\student_resource
    log.info(f"Working dir: {os.getcwd()}")

    # ──────────────────────────────────────────────────────────────────────
    # Step A: Feature engineering on new candidate pairs
    # ──────────────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP A: Feature engineering on new blocking candidates")
    log.info("="*60)
    run(
        [PYTHON, str(SRC / "fast_features.py"),
         "--candidate-pairs", "output/candidate_pairs.tsv",
         "--processed-dir", "dataset/processed",
         "--output", "output/features_test_v4.parquet",
         "--split", "test",
         "--n-workers", "16",
         ],
        "Feature engineering (v4 candidates)",
        log_file="output/features_v4.log",
    )

    # ──────────────────────────────────────────────────────────────────────
    # Step B: Score embedding similarity for new candidates
    # ──────────────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP B: Embedding similarity scoring")
    log.info("="*60)
    run(
        [PYTHON, str(SRC / "embed_features.py"), "score",
         "--embeddings-dir", "output/embeddings",
         "--candidate-file", "output/candidate_pairs.tsv",
         "--output-file", "output/embed_scores_v4.parquet",
         "--split", "test",
         ],
        "Embedding scoring",
        log_file="output/embed_score_v4.log",
    )

    # ──────────────────────────────────────────────────────────────────────
    # Step C: Merge embedding feature
    # ──────────────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP C: Merging embedding cosine sim into features")
    log.info("="*60)
    run(
        [PYTHON, str(SRC / "embed_features.py"), "merge",
         "--features-file", "output/features_test_v4.parquet",
         "--embed-scores-file", "output/embed_scores_v4.parquet",
         "--output-file", "output/features_test_v4_with_embeds.parquet",
         ],
        "Merging embeddings",
        log_file="output/embed_merge_v4.log",
    )

    # ──────────────────────────────────────────────────────────────────────
    # Step D: Inference with lower threshold
    # ──────────────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP D: Inference with threshold=0.40, max_k=4")
    log.info("="*60)

    # Load calibrated threshold from model meta
    meta_path = OUT / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        old_thresh = meta.get("threshold", 0.8029)
        log.info(f"  Old calibrated threshold: {old_thresh}")
    
    # Use 0.40 as the new threshold (from our sweep)
    NEW_THRESH = 0.40
    MAX_K = 4

    run(
        [PYTHON, str(SRC / "inference.py"),
         "--features-file", "output/features_test_v4_with_embeds.parquet",
         "--model-path", "output/models/lgbm_model.pkl",
         "--output", "output/matching_results_v4.tsv",
         "--threshold", str(NEW_THRESH),
         "--max-matches-per-entity", str(MAX_K),
         "--entity-batch-size", "10000",
         ],
        f"Inference (threshold={NEW_THRESH}, max_k={MAX_K})",
        log_file="output/inference_v4.log",
    )

    # ──────────────────────────────────────────────────────────────────────
    # Step E: Validate
    # ──────────────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP E: Validation")
    log.info("="*60)
    import polars as pl
    df = pl.read_csv("output/matching_results_v4.tsv", separator="\t", infer_schema_length=0)
    n = len(df)
    n_empty = df.filter(pl.col("matched_entity_ids").is_null()).height
    log.info(f"  Rows: {n:,}")
    log.info(f"  Singletons: {n_empty:,} ({n_empty/n*100:.2f}%)")
    log.info(f"  Non-singletons: {n-n_empty:,}")

    # ──────────────────────────────────────────────────────────────────────
    # Step F: Create zip
    # ──────────────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP F: Creating submission zip")
    log.info("="*60)
    import zipfile
    zip_path = OUT / "submission_v4.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("output/matching_results_v4.tsv", arcname="matching_results_v4.tsv")
    log.info(f"  ✓ Created {zip_path}")
    log.info("  ► READY TO SUBMIT: output/submission_v4.zip")


if __name__ == "__main__":
    main()
