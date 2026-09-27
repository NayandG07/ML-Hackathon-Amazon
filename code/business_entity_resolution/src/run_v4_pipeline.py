"""
run_v4_pipeline.py
==================
Full pipeline to generate the v4 submission:
  1. (Prerequisite) Test embeddings exist:
     - output/embeddings/test_s1_embeds.npy
     - output/embeddings/test_s23_embeds.npy
  2. (Prerequisite) Retrained model with embedding feature exists:
     - output/models/lgbm_model_v2.pkl
     - output/models/model_meta_v2.json
  3. (Optional) Run candidate augmentation:
     - Uses word-level TF-IDF (top_k=200) + house number blocking to boost recall
  4. Score embedding cosine similarity for candidate pairs:
     - python embed_features.py score
  5. Run streaming inference with Dual-Agreement Gate + embed feature + optimal threshold:
     - python inference.py --model-name lgbm_model_v2.pkl --embed-scores-file ...
  6. Validate submission format
  7. Generate submission_v4.zip
"""

import sys
import os
import subprocess
import json
import logging
import time
import zipfile
from pathlib import Path
import polars as pl

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
        with open(log_file, "w", encoding="utf-8") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log.error(f"  FAILED (exit {result.returncode}) after {elapsed:.0f}s — check {log_file}")
        raise RuntimeError(f"Step failed: {desc}")
    log.info(f"  ✓ Finished in {elapsed:.1f}s")
    return result


def main():
    os.chdir(Path(__file__).parent.parent.parent.parent)
    log.info(f"Working dir: {os.getcwd()}")

    candidate_file = OUT / "candidate_pairs.tsv"
    embed_scores_file = OUT / "embed_scores_test.parquet"
    model_name = "lgbm_model_v2.pkl"
    meta_name = "model_meta_v2.json"

    # Verify embeddings exist
    s1_emb = OUT / "embeddings" / "test_s1_embeds.npy"
    s23_emb = OUT / "embeddings" / "test_s23_embeds.npy"
    if not (s1_emb.exists() and s23_emb.exists()):
        log.error("Test embeddings not ready yet! Wait for task-2442 to finish.")
        sys.exit(1)

    # Verify model exists
    if not (OUT / "models" / model_name).exists():
        log.warning(f"Model {model_name} not found yet. Falling back to lgbm_model.pkl if available.")
        model_name = "lgbm_model.pkl"

    # Step 1: Score embeddings on candidate pairs
    log.info("=" * 60)
    log.info("STEP 1: Computing candidate embedding cosine similarities")
    log.info("=" * 60)
    run(
        [
            PYTHON, str(SRC / "embed_features.py"), "score",
            "--embeddings-dir", str(OUT / "embeddings"),
            "--candidate-file", str(candidate_file),
            "--output-file", str(embed_scores_file),
            "--split", "test",
        ],
        "Score candidate embeddings",
        log_file=str(OUT / "embed_score_test.log"),
    )

    # Step 2: Optimal inference threshold and max_k
    log.info("=" * 60)
    log.info("STEP 2: Streaming Inference with LightGBM + Dual-Gate + Embeddings")
    log.info("=" * 60)

    meta_path = OUT / "models" / meta_name
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        thresh = float(m.get("threshold", 0.88))
        log.info(f"Loaded calibrated threshold from {meta_name}: {thresh:.4f} (val F0.5: {m.get('val_macro_f05', 0):.4f})")
    else:
        thresh = 0.88
        log.info(f"Using default high-precision threshold: {thresh:.4f}")

    run(
        [
            PYTHON, str(SRC / "inference.py"),
            "--processed-dir", str(PROCESSED),
            "--candidate-file", str(candidate_file),
            "--model-dir", str(OUT / "models"),
            "--model-name", model_name,
            "--output-dir", str(OUT),
            "--threshold-override", str(thresh),
            "--max-matches-per-entity", "4",
            "--entity-batch-size", "10000",
            "--min-name-sim", "0.40",
            "--min-addr-sim", "0.25",
            "--embed-scores-file", str(embed_scores_file),
        ],
        "Streaming inference",
        log_file=str(OUT / "inference_v4.log"),
    )

    # Step 3: Validate output matching_results.tsv
    log.info("=" * 60)
    log.info("STEP 3: Validation")
    log.info("=" * 60)
    res_path = OUT / "matching_results.tsv"
    if not res_path.exists():
        log.error("matching_results.tsv not found!")
        sys.exit(1)

    df = pl.read_csv(str(res_path), separator="\t", infer_schema_length=0)
    n = len(df)
    n_singletons = df.filter(pl.col("matched_entity_ids") == "").height
    log.info(f"Total entities: {n:,}")
    log.info(f"Singletons: {n_singletons:,} ({n_singletons / n * 100:.2f}%) [Target: ~5.58%]")
    log.info(f"Entities with matches: {n - n_singletons:,} ({(n - n_singletons) / n * 100:.2f}%)")

    # Step 4: Create zip
    log.info("=" * 60)
    log.info("STEP 4: Creating submission_v4.zip")
    log.info("=" * 60)
    zip_path = OUT / "submission_v4.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(res_path), arcname="matching_results.tsv")

    log.info(f"✓ Created {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB)")
    log.info("Pipeline v4 complete! Ready to submit.")


if __name__ == "__main__":
    main()
