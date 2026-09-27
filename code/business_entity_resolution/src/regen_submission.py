"""
Fast submission regeneration from scores_raw.parquet.
Uses the best threshold + max_k found from threshold_sweep.py.
No need to re-run inference — uses cached scores.

Usage:
    python code/business_entity_resolution/src/regen_submission.py \
        --threshold 0.40 --max-k 5 \
        --output output/matching_results_v2.tsv

Then zip: Compress-Archive -Path output/matching_results_v2.tsv -DestinationPath output/submission_v2.zip
"""

import sys
import os
import argparse
import polars as pl
from pathlib import Path
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCORES_PATH = "output/scores_raw.parquet"
SUBMISSION_PATH = "output/matching_results.tsv"   # to get full S1 entity list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.40, help="Score threshold for a match")
    parser.add_argument("--max-k", type=int, default=5, help="Max matches per S1 entity")
    parser.add_argument("--output", type=str, default="output/matching_results_v2.tsv", help="Output TSV path")
    parser.add_argument("--zip-output", type=str, default="output/submission_v2.zip", help="Zip output path")
    args = parser.parse_args()

    log.info(f"Config: threshold={args.threshold}, max_k={args.max_k}")
    
    # Load raw scores
    log.info("Loading scores_raw.parquet ...")
    scores = pl.read_parquet(SCORES_PATH)
    log.info(f"  {len(scores):,} pairs with score >= 0.40")

    # Load entity list (to ensure all S1 entities are in output, even singletons)
    log.info("Loading entity list ...")
    submission = pl.read_csv(SUBMISSION_PATH, separator="\t", infer_schema_length=0)
    all_s1_ids = submission["source1_entity_id"].to_list()
    n_entities = len(all_s1_ids)
    log.info(f"  {n_entities:,} S1 entities")

    # Filter and apply max-k per entity
    log.info(f"Filtering pairs with score >= {args.threshold} ...")
    filtered = scores.filter(pl.col("score") >= args.threshold)
    log.info(f"  {len(filtered):,} pairs after threshold filter")

    # Rank and top-k
    log.info(f"Applying top-{args.max_k} per S1 entity ...")
    ranked = (
        filtered
        .sort("score", descending=True)
        .group_by("s1_id")
        .head(args.max_k)
    )
    log.info(f"  {len(ranked):,} pairs after top-k cap")

    # Group by s1_id -> comma-separated cand_ids
    log.info("Building match groups ...")
    grouped = (
        ranked
        .group_by("s1_id")
        .agg(pl.col("cand_id").sort_by("score", descending=True).str.join(",").alias("matched_entity_ids"))
    )
    # Note: score column isn't in grouped, sort within group happens via sort before group_by
    
    # Re-sort cand_ids by score within each group (ranked already sorted before agg, polars preserves order)
    match_map = dict(zip(grouped["s1_id"].to_list(), grouped["matched_entity_ids"].to_list()))

    # Write output
    log.info(f"Writing submission to {args.output} ...")
    out_rows = []
    for s1_id in all_s1_ids:
        matched = match_map.get(s1_id, "")
        out_rows.append(f"{s1_id}\t{matched}")
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("\n".join(out_rows))
        f.write("\n")
    
    log.info(f"Written {len(out_rows):,} rows.")

    # Stats
    n_singletons = sum(1 for _, m in zip(all_s1_ids, out_rows) if "\t" not in m or m.split("\t")[1] == "")
    n_singletons = sum(1 for s1_id in all_s1_ids if match_map.get(s1_id, "") == "")
    log.info(f"  Singletons: {n_singletons:,} ({n_singletons/n_entities*100:.2f}%)")
    log.info(f"  Non-singletons: {n_entities-n_singletons:,} ({(n_entities-n_singletons)/n_entities*100:.2f}%)")
    
    # Zip it
    log.info(f"Creating zip: {args.zip_output} ...")
    with zipfile.ZipFile(args.zip_output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output, arcname=Path(args.output).name)
    log.info(f"Zip written: {args.zip_output}")
    log.info("Done! Ready to submit.")


if __name__ == "__main__":
    main()
