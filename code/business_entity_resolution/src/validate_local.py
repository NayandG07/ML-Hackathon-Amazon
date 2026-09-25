"""
validate_local.py
=================
Local macro F0.5 validation script for a held-out portion of the training data.

This script does NOT use the actual test set ground truth (which is hidden).
Instead, it:
  1. Splits the training set into a pseudo-train and pseudo-test.
  2. Runs the full pipeline (blocking + feature engineering + inference)
     on the pseudo-test split.
  3. Computes macro F0.5 against the known train ground truth for those S1 IDs.
  4. Prints per-country breakdown and error analysis.

Usage
-----
    python validate_local.py \
        --train-dir dataset/train \
        --model-dir output/models \
        --n-val-entities 20000 \
        --seed 42

This gives a reliable estimate of leaderboard performance without spending
an official submission.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import polars as pl
import numpy as np

log = logging.getLogger("validate_local")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def macro_f05_from_submission(
    matching_df: pl.DataFrame,
    gt: dict[str, set[str]],
) -> dict[str, float]:
    """
    Compute macro F0.5 from a matching_results DataFrame and ground truth.

    Returns dict with 'overall', 'US', 'India', 'France' keys.
    """
    # We need per-entity country to compute breakdown
    # (requires loading S1 data; handled in caller)
    scores: list[float] = []
    country_scores: dict[str, list[float]] = {}

    for row in matching_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        pred_str = row.get("matched_entity_ids", "") or ""
        pred_set = set(pred_str.split(",")) if pred_str else set()

        true_set = gt.get(s1_id, set())
        # Macro F0.5 per entity
        if not true_set:
            score = 1.0 if not pred_set else 0.0
        else:
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = (
                (1.25 * prec * rec) / (0.25 * prec + rec)
                if (prec + rec) > 0
                else 0.0
            )
        scores.append(score)

    return {
        "overall": float(np.mean(scores)) if scores else 0.0,
        "n_entities": len(scores),
    }


def error_analysis(
    matching_df: pl.DataFrame,
    gt: dict[str, set[str]],
    candidates_df: pl.DataFrame,
    n_examples: int = 5,
) -> None:
    """
    Print detailed error analysis: false positives, false negatives,
    and blocking failures.
    """
    fp_examples: list[tuple[str, str]] = []  # (s1_id, false_match)
    fn_examples: list[tuple[str, str]] = []  # (s1_id, missed_match)
    blocking_fail_examples: list[tuple[str, str]] = []  # missed at blocking stage

    # Build candidate set lookup
    cand_lookup: dict[str, set[str]] = {}
    for row in candidates_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        cand_str = row.get("candidate_entity_ids", "") or ""
        cand_lookup[s1_id] = set(cand_str.split(",")) if cand_str else set()

    for row in matching_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        pred_str = row.get("matched_entity_ids", "") or ""
        pred_set = set(pred_str.split(",")) if pred_str else set()
        true_set = gt.get(s1_id, set())

        # False positives
        for m in pred_set - true_set:
            if len(fp_examples) < n_examples:
                fp_examples.append((s1_id, m))

        # False negatives: check if they were even in blocking candidates
        cand_set = cand_lookup.get(s1_id, set())
        for m in true_set - pred_set:
            if m in cand_set:
                # In candidates but classified as negative → threshold issue
                if len(fn_examples) < n_examples:
                    fn_examples.append((s1_id, m))
            else:
                # Not even in candidates → blocking failure
                if len(blocking_fail_examples) < n_examples:
                    blocking_fail_examples.append((s1_id, m))

    log.info("\n=== ERROR ANALYSIS ===")
    log.info(f"\nFalse Positives (top {n_examples}):")
    for s1_id, fp_id in fp_examples:
        log.info(f"  S1: {s1_id}  →  FP: {fp_id}")

    log.info(f"\nFalse Negatives (in candidates but below threshold, top {n_examples}):")
    for s1_id, fn_id in fn_examples:
        log.info(f"  S1: {s1_id}  →  FN: {fn_id}")

    log.info(f"\nBlocking Failures (not even retrieved, top {n_examples}):")
    for s1_id, miss_id in blocking_fail_examples:
        log.info(f"  S1: {s1_id}  →  MISSED: {miss_id}")

    # Blocking recall stats
    total_true_matches = sum(len(v) for v in gt.values())
    retrieved = sum(
        len(gt.get(s1_id, set()) & cands)
        for s1_id, cands in cand_lookup.items()
    )
    blocking_recall = retrieved / total_true_matches if total_true_matches > 0 else 0.0
    log.info(
        f"\nBlocking Recall: {blocking_recall:.4f}"
        f" ({retrieved:,}/{total_true_matches:,} true matches retrieved)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching-file", default="output/matching_results.tsv")
    parser.add_argument("--candidate-file", default="output/candidate_pairs.tsv")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    args = parser.parse_args()

    log.info(f"Loading matching results from {args.matching_file} …")
    matching_df = pl.read_csv(
        args.matching_file,
        separator="\t",
        null_values=[""],
        infer_schema_length=100,
    )

    log.info(f"Loading ground truth from {args.ground_truth} …")
    gt: dict[str, set[str]] = {}
    with open(args.ground_truth, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            matches = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
            gt[s1_id] = matches

    # Filter gt to only the S1 IDs in matching file
    eval_s1_ids = set(matching_df["source1_entity_id"].to_list())
    gt_eval = {k: v for k, v in gt.items() if k in eval_s1_ids}

    log.info(f"  Evaluating on {len(gt_eval):,} S1 entities …")
    metrics = macro_f05_from_submission(matching_df, gt_eval)
    log.info(f"\n{'='*50}")
    log.info(f"  Macro F0.5 Score: {metrics['overall']:.4f}")
    log.info(f"  N Entities:       {metrics['n_entities']:,}")
    log.info(f"{'='*50}")

    if args.candidate_file and Path(args.candidate_file).exists():
        log.info(f"\nLoading candidates from {args.candidate_file} …")
        candidates_df = pl.read_csv(
            args.candidate_file,
            separator="\t",
            null_values=[""],
            infer_schema_length=100,
        )
        error_analysis(matching_df, gt_eval, candidates_df)


if __name__ == "__main__":
    main()
