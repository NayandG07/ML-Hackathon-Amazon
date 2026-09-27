"""
Offline threshold sweep on saved scores_raw.parquet.
Explores different (threshold, max_k) combinations to find the config
that maximizes Macro F0.5 on the test distribution proxy.

Since we don't have test labels, we use the distribution match heuristic:
- Ground truth: ~5.58% singletons, median 4 matches/entity
- We want to minimise singleton over-prediction and match under-prediction

Outputs a ranked table + recommended config.
"""
import sys
import os
import json
import numpy as np
import polars as pl
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCORES_PATH = "output/scores_raw.parquet"
SUBMISSION_PATH = "output/matching_results.tsv"  # current submission for entity list
META_PATH = "output/model_meta.json"
OUT_REPORT = "output/threshold_sweep_report.json"

# Sweep ranges
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.8029, 0.85, 0.87, 0.90]
MAX_KS = [3, 4, 5, 6, 8, 10]

# Ground-truth statistics (from competition description / training data analysis)
GT_SINGLETON_RATE = 0.0558   # 5.58% entities have no matches
GT_MEDIAN_MATCHES = 4.0      # median non-empty cluster size

# ---------------------------------------------------------------------------

def load_data():
    print("Loading scores_raw.parquet ...", flush=True)
    scores = pl.read_parquet(SCORES_PATH)
    print(f"  {len(scores):,} scored pairs", flush=True)
    
    print("Loading matching_results.tsv to get entity list ...", flush=True)
    submission = pl.read_csv(SUBMISSION_PATH, separator="\t", infer_schema_length=0)
    print(f"  {len(submission):,} entities in submission", flush=True)
    
    all_s1_ids = submission["source1_entity_id"].unique()
    n_entities = len(all_s1_ids)
    print(f"  {n_entities:,} unique S1 entities", flush=True)
    
    return scores, submission, all_s1_ids, n_entities


def simulate(scores_df, all_s1_ids, threshold, max_k, n_entities):
    """Apply threshold + max-K cap and compute distribution statistics."""
    # Filter by threshold
    filtered = scores_df.filter(pl.col("score") >= threshold)
    
    # For each s1_id, keep top-k by score
    ranked = (
        filtered
        .sort("score", descending=True)
        .group_by("s1_id")
        .head(max_k)
    )
    
    # Count matches per entity
    match_counts = ranked.group_by("s1_id").agg(pl.len().alias("n_matches"))
    
    # Entities with no matches (singletons) = entities not in match_counts
    n_with_matches = len(match_counts)
    n_singletons = n_entities - n_with_matches
    
    singleton_rate = n_singletons / n_entities
    avg_matches = match_counts["n_matches"].mean() if len(match_counts) > 0 else 0
    median_matches = match_counts["n_matches"].median() if len(match_counts) > 0 else 0
    
    # Singleton rate delta (how close to 5.58%)
    singleton_delta = abs(singleton_rate - GT_SINGLETON_RATE)
    
    # Median match delta (how close to GT median 4)
    median_delta = abs(median_matches - GT_MEDIAN_MATCHES)
    
    # Composite score: lower is better
    # We weight singleton rate heavily since it directly drives Macro F0.5
    composite = 3 * singleton_delta + 1 * (median_delta / GT_MEDIAN_MATCHES)
    
    return {
        "threshold": threshold,
        "max_k": max_k,
        "n_pairs": len(ranked),
        "n_entities_with_matches": n_with_matches,
        "n_singletons": n_singletons,
        "singleton_rate": round(singleton_rate, 4),
        "singleton_rate_delta": round(singleton_delta, 4),
        "avg_matches": round(avg_matches, 3),
        "median_matches": round(median_matches, 1),
        "median_delta": round(median_delta, 2),
        "composite_score": round(composite, 5),  # lower = better distribution match
    }


def main():
    scores, submission, all_s1_ids, n_entities = load_data()
    
    results = []
    print(f"\nRunning {len(THRESHOLDS) * len(MAX_KS)} combinations ...", flush=True)
    print(f"{'thresh':>8} {'max_k':>6} {'singleton%':>12} {'delta_sing':>12} {'avg_match':>10} {'median_m':>10} {'composite':>12}", flush=True)
    print("-" * 75, flush=True)
    
    for threshold in THRESHOLDS:
        for max_k in MAX_KS:
            r = simulate(scores, all_s1_ids, threshold, max_k, n_entities)
            results.append(r)
            print(
                f"{r['threshold']:>8.4f} {r['max_k']:>6d} "
                f"{r['singleton_rate']*100:>11.2f}% "
                f"{r['singleton_rate_delta']*100:>11.2f}% "
                f"{r['avg_matches']:>10.2f} "
                f"{r['median_matches']:>10.1f} "
                f"{r['composite_score']:>12.5f}",
                flush=True
            )
    
    # Sort by composite score (lower = better)
    results.sort(key=lambda x: x["composite_score"])
    
    print("\n" + "="*75, flush=True)
    print("TOP 10 CONFIGS (lower composite = better distribution match):", flush=True)
    print("="*75, flush=True)
    for i, r in enumerate(results[:10]):
        print(
            f"#{i+1}: threshold={r['threshold']:.4f}, max_k={r['max_k']}, "
            f"singleton={r['singleton_rate']*100:.2f}% (GT=5.58%), "
            f"median_matches={r['median_matches']} (GT=4.0), "
            f"composite={r['composite_score']:.5f}",
            flush=True
        )
    
    best = results[0]
    print(f"\n★ BEST CONFIG: threshold={best['threshold']}, max_k={best['max_k']}", flush=True)
    print(f"  Singleton rate: {best['singleton_rate']*100:.2f}% (target ~5.58%)", flush=True)
    print(f"  Avg matches: {best['avg_matches']:.2f} (target median 4)", flush=True)
    print(f"  Total matched pairs: {best['n_pairs']:,}", flush=True)
    
    # Save report
    report = {
        "best_config": best,
        "all_results": results,
        "gt_singleton_rate": GT_SINGLETON_RATE,
        "gt_median_matches": GT_MEDIAN_MATCHES,
    }
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {OUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
