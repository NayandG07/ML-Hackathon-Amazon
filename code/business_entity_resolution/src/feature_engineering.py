"""
feature_engineering.py
=======================
Pairwise feature extraction for Business Entity Resolution.

For each (S1_entity, S2/S3_candidate) pair, this module produces a dense
feature vector suitable for training LightGBM / XGBoost classifiers.

Feature Groups
--------------
A. Name Similarity Features  (15 features)
B. Address Similarity Features (12 features)
C. Structural / Metadata Features (8 features)
D. Cross / Interaction Features (5 features)
                            ----------------
   Total:                   40 features

All features are computed via RapidFuzz (C++ backed) for maximum speed.
No ML embeddings at this stage — those are optional Stage-2 re-rankers.

Design Decisions
----------------
* Use Token Sort Ratio instead of ratio() as the default: handles word-reorder
  variants ("Dahlia Power Reliable" vs "Reliable Power Dahlia") which are
  very common in Indian business names.
* Jaccard on sets of tokens: robust to repetition and more interpretable than
  edit distance for multi-word names.
* Address token intersection is computed on the deduplicated set of 3+ char
  tokens. Numeric tokens (house numbers, PINs) are extracted separately.
* Empty address is flagged explicitly — the model must learn that two records
  with empty addresses can still match on name alone.
* Country consistency: a mismatched country between S1 and candidate is a
  very strong negative signal (set to 0.0 hard).

Usage
-----
    python feature_engineering.py \
        --processed-dir dataset/processed \
        --candidate-file output/candidate_pairs.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --split train \
        --output-file output/features_train.parquet

For test inference (no ground truth):
    python feature_engineering.py \
        --processed-dir dataset/processed \
        --candidate-file output/candidate_pairs.tsv \
        --split test \
        --output-file output/features_test.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from rapidfuzz import fuzz, distance as rfuzz_dist

from pipeline_utils import StageTimer, console, get_logger, make_progress, print_metrics
from parallel_config import CHUNK_FEATURES, N_THREAD_WORKERS, safe_n_workers

log = get_logger("feature_engineering")

# Re-export FEATURE_COLS so callers that do
#   from feature_engineering import FEATURE_COLS
# continue to work after the canonical definition moved to fast_features.py
from fast_features import FEATURE_COLS  # noqa: F401  (intentional re-export)



# ---------------------------------------------------------------------------
# Atomic similarity functions
# ---------------------------------------------------------------------------

def _safe_fuzz(func, a: str, b: str) -> float:
    """Run a rapidfuzz function safely, returning 0.0 on empty input."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return func(a, b) / 100.0


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text))


def jaccard_tokens(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def token_overlap_count(a: str, b: str) -> int:
    ta, tb = _token_set(a), _token_set(b)
    return len(ta & tb)


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+\b", text))


def numeric_jaccard(a: str, b: str) -> float:
    na, nb = _numeric_tokens(a), _numeric_tokens(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


# ---------------------------------------------------------------------------
# Feature computation for a single pair
# ---------------------------------------------------------------------------

def compute_pair_features(
    s1: dict,
    cand: dict,
) -> dict[str, float]:
    """
    Compute all 40 features for one (S1, candidate) pair.

    Parameters
    ----------
    s1   : dict with keys: norm_name, norm_address, name_tokens,
           address_tokens, postal_code, house_number, country
    cand : same keys as above

    Returns
    -------
    dict mapping feature_name → float value
    """
    feat: dict[str, float] = {}

    n1 = s1.get("norm_name", "") or ""
    n2 = cand.get("norm_name", "") or ""
    a1 = s1.get("norm_address", "") or ""
    a2 = cand.get("norm_address", "") or ""
    nt1 = s1.get("name_tokens", "") or ""
    nt2 = cand.get("name_tokens", "") or ""
    at1 = s1.get("address_tokens", "") or ""
    at2 = cand.get("address_tokens", "") or ""
    pc1 = s1.get("postal_code") or ""
    pc2 = cand.get("postal_code") or ""
    hn1 = s1.get("house_number") or ""
    hn2 = cand.get("house_number") or ""
    c1 = (s1.get("country") or "").upper()
    c2 = (cand.get("country") or "").upper()

    # ----------------------------------------------------------------
    # A. Name similarity features
    # ----------------------------------------------------------------
    # A1: Full string ratio (character-level Levenshtein similarity)
    feat["name_ratio"] = _safe_fuzz(fuzz.ratio, n1, n2)

    # A2: Partial ratio (best substring alignment)
    feat["name_partial_ratio"] = _safe_fuzz(fuzz.partial_ratio, n1, n2)

    # A3: Token sort ratio (handles word reordering)
    feat["name_token_sort_ratio"] = _safe_fuzz(fuzz.token_sort_ratio, n1, n2)

    # A4: Token set ratio (subset containment, ignores extra words)
    feat["name_token_set_ratio"] = _safe_fuzz(fuzz.token_set_ratio, n1, n2)

    # A5: Jaccard on normalised name tokens
    feat["name_jaccard"] = jaccard_tokens(nt1, nt2)

    # A6: Token overlap count (absolute)
    feat["name_token_overlap"] = float(token_overlap_count(nt1, nt2))

    # A7: Normalised Jaro-Winkler distance
    if n1 and n2:
        feat["name_jaro_winkler"] = rfuzz_dist.JaroWinkler.normalized_similarity(n1, n2)
    else:
        feat["name_jaro_winkler"] = 0.0

    # A8: Length ratio (min/max chars)
    len1, len2 = len(n1), len(n2)
    feat["name_len_ratio"] = min(len1, len2) / max(len1, len2, 1)

    # A9: Absolute length difference
    feat["name_len_diff"] = float(abs(len1 - len2))

    # A10: Token count ratio
    tc1 = len(nt1.split())
    tc2 = len(nt2.split())
    feat["name_token_count_ratio"] = min(tc1, tc2) / max(tc1, tc2, 1)

    # A11: Shortest unique prefix match length (first N chars)
    prefix_len = min(10, len(n1), len(n2))
    feat["name_prefix_match"] = float(n1[:prefix_len] == n2[:prefix_len]) if prefix_len > 0 else 0.0

    # A12: Character n-gram (bigram) Jaccard on name
    bg1 = set(n1[i:i+2] for i in range(len(n1)-1))
    bg2 = set(n2[i:i+2] for i in range(len(n2)-1))
    if bg1 | bg2:
        feat["name_bigram_jaccard"] = len(bg1 & bg2) / len(bg1 | bg2)
    else:
        feat["name_bigram_jaccard"] = 0.0

    # A13: Trigram Jaccard on name
    tg1 = set(n1[i:i+3] for i in range(len(n1)-2))
    tg2 = set(n2[i:i+3] for i in range(len(n2)-2))
    if tg1 | tg2:
        feat["name_trigram_jaccard"] = len(tg1 & tg2) / len(tg1 | tg2)
    else:
        feat["name_trigram_jaccard"] = 0.0

    # A14: Shared first token match (most discriminative for Indian business names)
    tok_list1 = nt1.split()
    tok_list2 = nt2.split()
    feat["name_first_token_match"] = float(
        bool(tok_list1 and tok_list2 and tok_list1[0] == tok_list2[0])
    )

    # A15: Token count diff (absolute)
    feat["name_token_count_diff"] = float(abs(tc1 - tc2))

    # ----------------------------------------------------------------
    # B. Address similarity features
    # ----------------------------------------------------------------
    # B1: Full string ratio on normalised address
    feat["addr_ratio"] = _safe_fuzz(fuzz.ratio, a1, a2)

    # B2: Token sort ratio on address
    feat["addr_token_sort_ratio"] = _safe_fuzz(fuzz.token_sort_ratio, a1, a2)

    # B3: Token set ratio on address
    feat["addr_token_set_ratio"] = _safe_fuzz(fuzz.token_set_ratio, a1, a2)

    # B4: Jaccard on address tokens
    feat["addr_jaccard"] = jaccard_tokens(at1, at2)

    # B5: Address token overlap count
    feat["addr_token_overlap"] = float(token_overlap_count(at1, at2))

    # B6: Numeric Jaccard on addresses (house numbers, flat numbers)
    feat["addr_numeric_jaccard"] = numeric_jaccard(a1, a2)

    # B7: Postal code match (exact)
    feat["postal_code_match"] = float(bool(pc1 and pc2 and pc1 == pc2))

    # B8: Postal code both missing (empty × empty)
    feat["postal_both_empty"] = float(not pc1 and not pc2)

    # B9: House number exact match
    feat["house_num_match"] = float(bool(hn1 and hn2 and hn1 == hn2))

    # B10: Address empty flags
    feat["s1_addr_empty"] = float(not a1.strip())
    feat["cand_addr_empty"] = float(not a2.strip())
    feat["both_addr_empty"] = float(not a1.strip() and not a2.strip())

    # ----------------------------------------------------------------
    # C. Structural / Metadata features
    # ----------------------------------------------------------------
    # C1: Country exact match
    feat["country_match"] = float(c1 == c2 and bool(c1))

    # C2: Source indicator (0=S2, 1=S3)
    cand_id = cand.get("entity_id", "")
    feat["cand_is_s3"] = float(cand_id.startswith("S3-"))

    # C3: Address / name length ratios across pair
    feat["combined_len_ratio"] = (
        (len(n1) + len(a1) + 1) / (len(n2) + len(a2) + 1)
    )

    # C4: Address completely covers name tokens (S3 often has domain names)
    if tok_list1:
        name_in_addr = sum(1 for t in tok_list1 if t in _token_set(at2))
        feat["name_tokens_in_cand_addr"] = name_in_addr / len(tok_list1)
    else:
        feat["name_tokens_in_cand_addr"] = 0.0

    # C5: Whether S1 name is substantially longer (abbreviation risk)
    feat["name_s1_longer"] = float(len(n1) > len(n2) * 1.5)
    feat["name_cand_longer"] = float(len(n2) > len(n1) * 1.5)

    # ----------------------------------------------------------------
    # D. Cross / interaction features
    # ----------------------------------------------------------------
    # D1: Name × Address joint score (harmonic mean)
    ns = feat["name_token_sort_ratio"]
    as_ = feat["addr_token_sort_ratio"]
    feat["name_addr_harmonic"] = (
        2 * ns * as_ / (ns + as_) if (ns + as_) > 0 else 0.0
    )

    # D2: Name × Jaccard product
    feat["name_jaccard_x_addr_jaccard"] = feat["name_jaccard"] * feat["addr_jaccard"]

    # D3: Max of name token sort and name jaccard
    feat["name_max_sim"] = max(feat["name_token_sort_ratio"], feat["name_jaccard"])

    # D4: Country mismatch hard penalty (informative as a feature, not as a rule)
    feat["country_mismatch_penalty"] = float(c1 != c2 and bool(c1) and bool(c2))

    # D5: Both short name (≤ 4 chars) — very ambiguous pairs
    feat["both_short_name"] = float(len(n1) <= 4 and len(n2) <= 4)

    # E: Embedding feature placeholder (filled in by embed_features.py)
    # Default = 0.0; overwritten when GPU embeddings are merged in.
    feat["embed_cosine_sim"] = 0.0

    return feat


# ---------------------------------------------------------------------------
# Batch feature computation
# ---------------------------------------------------------------------------

def load_entity_lookup(df: pl.DataFrame) -> dict[str, dict]:
    """Convert a Polars DataFrame to an entity_id → row dict for O(1) lookup."""
    rows = df.to_dicts()
    return {r["entity_id"]: r for r in rows}


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    """
    Load train_ground_truth.tsv into {s1_id: set(matched_ids)}.
    An S1 entity with no matches maps to an empty set.
    """
    labels: dict[str, set[str]] = {}
    with open(str(path), "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if len(parts) > 1 and parts[1]:
                labels[s1_id] = set(parts[1].split(","))
            else:
                labels[s1_id] = set()
    return labels


# ---------------------------------------------------------------------------
# Parallel worker — module-level for joblib pickling on Windows
# ---------------------------------------------------------------------------

def _featurise_chunk(
    chunk: list[tuple[str, dict, str, dict, Optional[float]]],
) -> list[dict]:
    """
    Process a chunk of (s1_id, s1_rec, cand_id, cand_rec, label) tuples.
    Module-level for Windows loky pickling. Called from featurise_candidates().
    rapidfuzz C++ extensions release the GIL → threading backend preferred.
    """
    results: list[dict] = []
    for s1_id, s1_rec, cand_id, cand_rec, label in chunk:
        feat = compute_pair_features(s1_rec, cand_rec)
        feat["s1_id"] = s1_id
        feat["cand_id"] = cand_id
        if label is not None:
            feat["label"] = label
        results.append(feat)
    return results


def featurise_candidates(
    candidates_df: pl.DataFrame,
    s1_lookup: dict[str, dict],
    s23_lookup: dict[str, dict],
    ground_truth: Optional[dict[str, set[str]]] = None,
) -> pl.DataFrame:
    """
    Expand candidate_pairs.tsv into a flat (pair, features, label) dataframe.

    Parallel execution:
      - Builds a flat list of (s1_id, s1_rec, cand_id, cand_rec, label) tuples
      - Splits into chunks of CHUNK_FEATURES pairs
      - Processes chunks using joblib threading backend
      - rapidfuzz is C++ backed and releases the GIL → ~8-12x speedup on 14 threads
    """
    # ── Step 1: Build flat pair list ──────────────────────────────────────
    log.info("Building flat pair list …")
    flat_pairs: list[tuple[str, dict, str, dict, Optional[float]]] = []
    missing_s1 = 0
    missing_cand = 0

    for row in candidates_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        cand_str = row.get("candidate_entity_ids", "") or ""
        if not cand_str:
            continue
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            missing_s1 += 1
            continue
        true_matches = (ground_truth.get(s1_id, set())
                        if ground_truth is not None else None)
        for cid in (c.strip() for c in cand_str.split(",") if c.strip()):
            cand_rec = s23_lookup.get(cid)
            if cand_rec is None:
                missing_cand += 1
                continue
            label = (float(cid in true_matches)
                     if true_matches is not None else None)
            flat_pairs.append((s1_id, s1_rec, cid, cand_rec, label))

    n_pairs = len(flat_pairs)
    log.info(f"  [highlight]{n_pairs:,}[/highlight] candidate pairs to featurise")
    if missing_s1:
        log.warning(f"  {missing_s1:,} S1 IDs not found in lookup")
    if missing_cand:
        log.warning(f"  {missing_cand:,} candidate IDs not found in lookup")

    # ── Step 2: Parallel feature computation ──────────────────────────────
    n_workers = safe_n_workers(N_THREAD_WORKERS)
    chunk_size = max(1, CHUNK_FEATURES)
    chunks = [flat_pairs[i:i + chunk_size]
              for i in range(0, n_pairs, chunk_size)]

    log.info(f"  Featurising: [highlight]{n_workers}[/highlight] threads  "
             f"× [highlight]{len(chunks)}[/highlight] chunks "
             f"(~{chunk_size:,} pairs each)")

    with make_progress() as progress:
        task = progress.add_task(
            f"  Computing features [{n_workers} threads]",
            total=len(chunks)
        )
        all_results: list[list[dict]] = []
        for chunk_result in Parallel(
            n_jobs=n_workers,
            backend="threading",   # threading: shared dicts in memory, rapidfuzz releases GIL
            return_as="generator",
        )(delayed(_featurise_chunk)(c) for c in chunks):
            all_results.append(chunk_result)
            progress.advance(task)

    # ── Step 3: Flatten and build DataFrame ───────────────────────────────
    records = [row for chunk in all_results for row in chunk]
    log.info(f"  [success]Done[/success] — {len(records):,} pairs featurised")
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--candidate-file", default="output/candidate_pairs.tsv")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output-file", default="output/features_train.parquet")
    parser.add_argument("--use-fast", action="store_true", default=True,
                        help="Use vectorized fast_features engine (default: on)")
    args = parser.parse_args()

    proc_dir = Path(args.processed_dir)

    with StageTimer(f"Feature Engineering — {args.split.upper()} set"):
        log.info("Loading preprocessed data …")
        s1_df  = pl.read_parquet(str(proc_dir / f"{args.split}_s1.parquet"))
        s2_df  = pl.read_parquet(str(proc_dir / f"{args.split}_s2.parquet"))
        s3_df  = pl.read_parquet(str(proc_dir / f"{args.split}_s3.parquet"))
        s23_df = pl.concat([s2_df, s3_df])

        log.info(f"Loading candidates from [highlight]{args.candidate_file}[/highlight] …")
        candidates_df = pl.read_csv(
            args.candidate_file, separator="\t",
            null_values=[""], infer_schema_length=100,
        )

        gt: Optional[dict[str, set[str]]] = None
        if args.split == "train":
            log.info(f"Loading ground truth from [highlight]{args.ground_truth}[/highlight] …")
            gt = load_ground_truth(args.ground_truth)

        if args.use_fast:
            # ── Fast path: Polars join + vectorized numpy/rapidfuzz ──────
            from fast_features import build_flat_pairs, vectorized_featurise
            flat_df = build_flat_pairs(candidates_df, s1_df, s23_df)
            features_df = vectorized_featurise(flat_df, ground_truth=gt, chunk_size=500_000)
        else:
            # ── Fallback: row-by-row Python loop (slower but simpler) ────
            log.info("Building entity lookup dicts …")
            s1_lookup  = load_entity_lookup(s1_df)
            s23_lookup = load_entity_lookup(s23_df)
            features_df = featurise_candidates(candidates_df, s1_lookup, s23_lookup, gt)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.write_parquet(str(out_path))
    log.info(f"Saved → [highlight]{out_path}[/highlight]")

    if args.split == "train":
        n_pos = features_df.filter(pl.col("label") == 1.0).height
        n_neg = features_df.filter(pl.col("label") == 0.0).height
        print_metrics("Feature Engineering Results", {
            "Total pairs":     f"{len(features_df):,}",
            "Positive pairs":  f"{n_pos:,}",
            "Negative pairs":  f"{n_neg:,}",
            "Imbalance ratio": f"1:{n_neg // max(n_pos, 1)}",
            "Output":          str(out_path),
        })


if __name__ == "__main__":
    main()


