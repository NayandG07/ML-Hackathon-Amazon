"""
calculate_metrics.py
====================
Calculates exact recall and performance metrics:
1. Candidate Blocking Recall (Recall Ceiling from Stage 1)
2. Classification Model Recall & Precision on Held-Out Validation Split (10% stratified)
3. Macro-averaged Recall, Precision, and F0.5 (Competition Metric)
"""

import json
import pickle
import random
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = Path("D:/OpenSource/student_resource")
OUTPUT_DIR = ROOT_DIR / "output"
GT_PATH = ROOT_DIR / "dataset" / "train" / "train_ground_truth.tsv"
TRAIN_CAND_PATH = OUTPUT_DIR / "train_blocking" / "candidate_pairs.tsv"
FEATURES_PATH = OUTPUT_DIR / "features_train_base.parquet"
MODEL_PATH = OUTPUT_DIR / "models" / "lgbm_model.pkl"
META_PATH = OUTPUT_DIR / "models" / "model_meta.json"


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    gt: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0]
                matches = set(p.strip() for p in parts[1].split(",") if p.strip())
                gt[s1_id] = matches
            elif len(parts) >= 1 and parts[0]:
                gt[parts[0]] = set()
    return gt


def compute_blocking_recall():
    print("=" * 70)
    print("1. EVALUATING CANDIDATE BLOCKING RECALL (STAGE 1 CEILING)")
    print("=" * 70)

    if not TRAIN_CAND_PATH.exists():
        print(f"Warning: {TRAIN_CAND_PATH} not found. Skipping blocking recall.")
        return None

    print(f"Loading ground truth from {GT_PATH} ...")
    gt = load_ground_truth(GT_PATH)
    total_s1 = len(gt)
    total_true_matches = sum(len(m) for m in gt.values())
    s1_with_matches = sum(1 for m in gt.values() if len(m) > 0)
    singletons = total_s1 - s1_with_matches

    print(f"  Total S1 entities in Ground Truth : {total_s1:,}")
    print(f"  Entities with true matches        : {s1_with_matches:,}")
    print(f"  True Singletons (no matches)      : {singletons:,}")
    print(f"  Total true match pairs            : {total_true_matches:,}")

    print(f"Streaming candidate pairs from {TRAIN_CAND_PATH} ...")
    captured_matches = 0
    s1_fully_captured = 0
    s1_partially_captured = 0

    with open(TRAIN_CAND_PATH, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2 or not parts[1]:
                continue
            s1_id = parts[0]
            true_set = gt.get(s1_id, set())
            if not true_set:
                continue

            cands = set(c.strip() for c in parts[1].split(",") if c.strip())
            found = cands & true_set
            captured_matches += len(found)

            if len(found) == len(true_set):
                s1_fully_captured += 1
            elif len(found) > 0:
                s1_partially_captured += 1

    blocking_recall = (captured_matches / total_true_matches) * 100 if total_true_matches > 0 else 0.0
    entity_full_recall = (s1_fully_captured / s1_with_matches) * 100 if s1_with_matches > 0 else 0.0

    print(f"\n[BLOCKING RESULTS]")
    print(f"  Captured true match pairs        : {captured_matches:,} / {total_true_matches:,}")
    print(f"  >> Pairwise Blocking Recall      : {blocking_recall:.2f}%")
    print(f"  Entities with 100% matches caught: {s1_fully_captured:,} / {s1_with_matches:,} ({entity_full_recall:.2f}%)")
    print(f"  Entities with partial matches    : {s1_partially_captured:,}")
    return blocking_recall


def compute_model_recall():
    print("\n" + "=" * 70)
    print("2. EVALUATING LIGHTGBM CLASSIFIER RECALL (STAGE 2)")
    print("=" * 70)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    feature_cols = meta["feature_cols"]

    print(f"Loading trained LightGBM model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Model decision threshold: {threshold:.4f}")
    print(f"Loading features from {FEATURES_PATH} ...")
    df = pl.read_parquet(FEATURES_PATH)
    print(f"  Loaded {len(df):,} total feature pairs.")

    # Stratified validation split by S1 entity (identically as during training)
    print("Creating stratified validation holdout (val_fraction=0.1, seed=42) ...")
    rng = random.Random(42)
    s1_unique = df["s1_id"].unique().to_list()
    rng.shuffle(s1_unique)
    n_val = max(1, int(len(s1_unique) * 0.1))
    val_s1 = set(s1_unique[:n_val])

    val_df = df.filter(pl.col("s1_id").is_in(val_s1))
    print(f"  Validation holdout set: {len(val_df):,} pairs across {len(val_s1):,} unseen S1 entities")

    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["label"].to_numpy().astype(int)

    print("Running model inference on validation holdout ...")
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

    pairwise_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pairwise_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pairwise_f05 = (1.25 * pairwise_precision * pairwise_recall) / (0.25 * pairwise_precision + pairwise_recall) if (pairwise_precision + pairwise_recall) > 0 else 0.0

    print("\n[VALIDATION PAIRWISE METRICS]")
    print(f"  True Positives  (TP) : {tp:,}")
    print(f"  False Positives (FP) : {fp:,}")
    print(f"  False Negatives (FN) : {fn:,}")
    print(f"  True Negatives  (TN) : {tn:,}")
    print(f"  >> Pairwise Recall   : {pairwise_recall * 100:.2f}%")
    print(f"  >> Pairwise Precision: {pairwise_precision * 100:.2f}%")
    print(f"  >> Pairwise F0.5     : {pairwise_f05 * 100:.2f}%")

    # Macro metrics per entity
    print("\nComputing Macro-Averaged Metrics (per S1 entity) ...")
    s1_val_ids = val_df["s1_id"].to_list()
    cand_val_ids = val_df["cand_id"].to_list()

    entity_truth = {}
    entity_preds = {}

    for s1, cid, yt, yp in zip(s1_val_ids, cand_val_ids, y_val, preds):
        entity_truth.setdefault(s1, set())
        entity_preds.setdefault(s1, set())
        if yt == 1:
            entity_truth[s1].add(cid)
        if yp == 1:
            entity_preds[s1].add(cid)

    macro_recalls = []
    macro_precisions = []
    macro_f05s = []

    for s1 in entity_truth:
        t_set = entity_truth[s1]
        p_set = entity_preds[s1]

        if not t_set:
            # Singleton entity
            if not p_set:
                macro_recalls.append(1.0)
                macro_precisions.append(1.0)
                macro_f05s.append(1.0)
            else:
                macro_recalls.append(0.0)
                macro_precisions.append(0.0)
                macro_f05s.append(0.0)
        else:
            tp_e = len(t_set & p_set)
            rec_e = tp_e / len(t_set)
            prec_e = tp_e / len(p_set) if p_set else 0.0
            f05_e = (1.25 * prec_e * rec_e) / (0.25 * prec_e + rec_e) if (prec_e + rec_e) > 0 else 0.0

            macro_recalls.append(rec_e)
            macro_precisions.append(prec_e)
            macro_f05s.append(f05_e)

    macro_rec = np.mean(macro_recalls) * 100
    macro_prec = np.mean(macro_precisions) * 100
    macro_f05 = np.mean(macro_f05s) * 100

    print(f"\n[MACRO-AVERAGED METRICS (Competition Leaderboard Standard)]")
    print(f"  >> Macro Recall    : {macro_rec:.2f}%")
    print(f"  >> Macro Precision : {macro_prec:.2f}%")
    print(f"  >> Macro F0.5      : {macro_f05:.2f}%")

    return {
        "pairwise_recall": pairwise_recall,
        "pairwise_precision": pairwise_precision,
        "macro_recall": macro_rec,
        "macro_precision": macro_prec,
        "macro_f05": macro_f05,
    }


def main():
    compute_blocking_recall()
    compute_model_recall()


if __name__ == "__main__":
    main()
