"""
calculate_model_recall.py
=========================
Fast, dedicated evaluation of LightGBM model recall and precision
on the stratified validation holdout (unseen S1 entities).
"""

import json
import pickle
import random
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path("D:/OpenSource/student_resource")
OUTPUT_DIR = ROOT_DIR / "output"
FEATURES_PATH = OUTPUT_DIR / "features_train_base.parquet"
MODEL_PATH = OUTPUT_DIR / "models" / "lgbm_model.pkl"
META_PATH = OUTPUT_DIR / "models" / "model_meta.json"
REPORT_PATH = OUTPUT_DIR / "model_recall_report.json"


def log(msg):
    print(msg, flush=True)


def main():
    log("=" * 70)
    log("LIGHTGBM MODEL RECALL & PRECISION EVALUATION")
    log("=" * 70)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    best_thresh = meta["threshold"]
    feature_cols = meta["feature_cols"]

    log(f"Model Optimal Threshold : {best_thresh:.4f}")
    log(f"Total Model Features    : {len(feature_cols)}")

    log(f"\nLoading trained model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    log(f"Loading features from {FEATURES_PATH} ...")
    df = pl.read_parquet(FEATURES_PATH)
    n_total = len(df)
    n_pos = df.filter(pl.col("label") == 1.0).height
    n_neg = df.filter(pl.col("label") == 0.0).height
    log(f"  Total pairs in dataset : {n_total:,} (Positives: {n_pos:,}, Negatives: {n_neg:,})")

    # Stratified validation split by S1 entity (identically as during training)
    log("\nCreating 10% stratified validation split (seed=42) ...")
    rng = random.Random(42)
    s1_unique = df["s1_id"].unique().to_list()
    rng.shuffle(s1_unique)
    n_val = max(1, int(len(s1_unique) * 0.1))
    val_s1 = set(s1_unique[:n_val])

    val_df = df.filter(pl.col("s1_id").is_in(val_s1))
    n_val_pairs = len(val_df)
    n_val_pos = val_df.filter(pl.col("label") == 1.0).height
    n_val_neg = val_df.filter(pl.col("label") == 0.0).height
    log(f"  Validation holdout pairs : {n_val_pairs:,} across {len(val_s1):,} unseen S1 entities")
    log(f"  Validation positives     : {n_val_pos:,}")
    log(f"  Validation negatives     : {n_val_neg:,}")

    log("\nExtracting feature matrix and running model predictions ...")
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["label"].to_numpy().astype(int)

    probs = model.predict_proba(X_val)[:, 1]

    # Evaluate at multiple decision thresholds
    thresholds_to_test = [0.30, 0.40, 0.50, round(best_thresh, 4), 0.70, 0.80]
    thresh_results = []

    log("\n" + "-" * 70)
    log(f"{'Threshold':>10} | {'Pairwise Rec':>13} | {'Pairwise Prec':>13} | {'Pairwise F0.5':>13} | {'Macro Rec':>10} | {'Macro Prec':>11} | {'Macro F0.5':>11}")
    log("-" * 70)

    s1_val_ids = val_df["s1_id"].to_list()
    cand_val_ids = val_df["cand_id"].to_list()

    entity_truth = {}
    for s1, cid, yt in zip(s1_val_ids, cand_val_ids, y_val):
        entity_truth.setdefault(s1, set())
        if yt == 1:
            entity_truth[s1].add(cid)

    report = {
        "dataset_total_pairs": n_total,
        "val_holdout_pairs": n_val_pairs,
        "val_holdout_entities": len(val_s1),
        "val_true_matches": n_val_pos,
        "val_hard_negatives": n_val_neg,
        "threshold_evaluations": [],
    }

    best_macro_f05 = 0.0
    primary_metrics = {}

    for t in thresholds_to_test:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

        pw_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pw_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pw_f05 = (1.25 * pw_prec * pw_rec) / (0.25 * pw_prec + pw_rec) if (pw_prec + pw_rec) > 0 else 0.0

        # Macro per entity
        entity_preds = {s1: set() for s1 in entity_truth}
        for s1, cid, yp in zip(s1_val_ids, cand_val_ids, preds):
            if yp == 1:
                entity_preds[s1].add(cid)

        m_rec_list = []
        m_prec_list = []
        m_f05_list = []

        for s1, t_set in entity_truth.items():
            p_set = entity_preds[s1]
            if not t_set:
                if not p_set:
                    m_rec_list.append(1.0)
                    m_prec_list.append(1.0)
                    m_f05_list.append(1.0)
                else:
                    m_rec_list.append(0.0)
                    m_prec_list.append(0.0)
                    m_f05_list.append(0.0)
            else:
                tp_e = len(t_set & p_set)
                rec_e = tp_e / len(t_set)
                prec_e = tp_e / len(p_set) if p_set else 0.0
                f05_e = (1.25 * prec_e * rec_e) / (0.25 * prec_e + rec_e) if (prec_e + rec_e) > 0 else 0.0
                m_rec_list.append(rec_e)
                m_prec_list.append(prec_e)
                m_f05_list.append(f05_e)

        macro_rec = np.mean(m_rec_list) * 100
        macro_prec = np.mean(m_prec_list) * 100
        macro_f05 = np.mean(m_f05_list) * 100

        is_primary = abs(t - best_thresh) < 0.001 or t == round(best_thresh, 4)
        marker = " *" if is_primary else ""

        log(f"{t:>9.4f}{marker} | {pw_rec*100:>12.2f}% | {pw_prec*100:>12.2f}% | {pw_f05*100:>12.2f}% | {macro_rec:>9.2f}% | {macro_prec:>10.2f}% | {macro_f05:>10.2f}%")

        res_dict = {
            "threshold": t,
            "pairwise_recall_pct": round(pw_rec * 100, 2),
            "pairwise_precision_pct": round(pw_prec * 100, 2),
            "pairwise_f05_pct": round(pw_f05 * 100, 2),
            "macro_recall_pct": round(macro_rec, 2),
            "macro_precision_pct": round(macro_prec, 2),
            "macro_f05_pct": round(macro_f05, 2),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        }
        report["threshold_evaluations"].append(res_dict)

        if is_primary:
            primary_metrics = res_dict

    log("-" * 70)
    log("(* indicates the optimal threshold calibrated during training for F0.5)")

    log("\n" + "=" * 70)
    log(f"PRIMARY MODEL RECALL (at threshold {best_thresh:.4f}):")
    log(f"  • Pairwise Classification Recall : {primary_metrics['pairwise_recall_pct']:.2f}%")
    log(f"  • Pairwise Classification Precision: {primary_metrics['pairwise_precision_pct']:.2f}%")
    log(f"  • Macro-Averaged Recall         : {primary_metrics['macro_recall_pct']:.2f}%")
    log(f"  • Macro-Averaged Precision      : {primary_metrics['macro_precision_pct']:.2f}%")
    log(f"  • Macro-Averaged F0.5           : {primary_metrics['macro_f05_pct']:.2f}%")
    log("=" * 70)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Full metrics report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
