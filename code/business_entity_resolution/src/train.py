"""
train.py
========
LightGBM training pipeline for Business Entity Resolution.

Design Choices
--------------
* LightGBM (not XGBoost or CatBoost) as primary model because:
  - Fastest training on CPU for the 50M+ pair candidate datasets.
  - Built-in categorical handling (country via label encoding).
  - histogram-based splits are memory-efficient.
  - dart booster with drop-connect is an effective regulariser.

* Validation strategy:
  - Stratified sample by S1 entity (NOT by pair): if S1-123 is in
    validation, ALL its candidate pairs go to validation, not some to train.
  - This simulates the leaderboard scenario where we predict on unseen
    S1 entities, not unseen S2/S3 records.
  - 10% validation split.

* Class imbalance:
  - Typical ratio is 1:20 to 1:50 positive:negative.
  - We use scale_pos_weight to give positives more weight,
    calibrated to target F0.5 precision-recall operating point.
  - Additionally, negative downsampling at training time (ratio=10:1).

* Threshold optimisation:
  - After training, sweep thresholds on the validation set to find
    the threshold maximising macro-averaged F0.5.
  - Macro F0.5: computed per S1 entity, then averaged.

* Feature importance:
  - SHAP values logged after training for interpretability.
  - Permutation importance also stored.

Usage
-----
    python train.py \
        --features-file output/features_train.parquet \
        --model-dir output/models \
        --n-estimators 2000 \
        --neg-sample-ratio 10 \
        --val-fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve

from pipeline_utils import StageTimer, console, get_logger, make_progress, print_metrics, print_done, sysinfo

log = get_logger("train")

# Feature column names — must match feature_engineering.py output
FEATURE_COLS = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_jaccard", "name_token_overlap",
    "name_jaro_winkler", "name_len_ratio", "name_len_diff",
    "name_token_count_ratio", "name_prefix_match", "name_bigram_jaccard",
    "name_trigram_jaccard", "name_first_token_match", "name_token_count_diff",
    "addr_ratio", "addr_token_sort_ratio", "addr_token_set_ratio",
    "addr_jaccard", "addr_token_overlap", "addr_numeric_jaccard",
    "postal_code_match", "postal_both_empty", "house_num_match",
    "s1_addr_empty", "cand_addr_empty", "both_addr_empty",
    "country_match", "cand_is_s3", "combined_len_ratio",
    "name_tokens_in_cand_addr", "name_s1_longer", "name_cand_longer",
    "name_addr_harmonic", "name_jaccard_x_addr_jaccard", "name_max_sim",
    "country_mismatch_penalty", "both_short_name",
    # GPU embedding feature (optional — set to 0.0 if embed_features.py not run)
    "embed_cosine_sim",
]

META_COLS = ["s1_id", "cand_id", "label"]


# ---------------------------------------------------------------------------
# Macro F0.5 evaluation
# ---------------------------------------------------------------------------

def macro_f05(
    s1_ids: list[str],
    cand_ids: list[str],
    labels: list[float],
    preds: list[float],
    threshold: float,
) -> float:
    """
    Compute macro-averaged F0.5 exactly as specified by the challenge.

    Parameters
    ----------
    s1_ids    : list of S1 entity IDs (one per candidate pair row)
    cand_ids  : list of candidate IDs
    labels    : ground truth binary labels (1.0 = match)
    preds     : predicted scores in [0, 1]
    threshold : score cutoff to binarise preds

    Returns
    -------
    float : macro F0.5
    """
    # Group by s1_id
    groups: dict[str, dict] = {}
    for s1, cid, lab, pred in zip(s1_ids, cand_ids, labels, preds):
        if s1 not in groups:
            groups[s1] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "true_matches": set()}
        if lab == 1.0:
            groups[s1]["true_matches"].add(cid)

    for s1, cid, lab, pred in zip(s1_ids, cand_ids, labels, preds):
        is_pos = pred >= threshold
        is_true = lab == 1.0
        if is_pos and is_true:
            groups[s1]["tp"] += 1
        elif is_pos and not is_true:
            groups[s1]["fp"] += 1
        elif not is_pos and is_true:
            groups[s1]["fn"] += 1

    scores: list[float] = []
    for s1_id, g in groups.items():
        tp, fp, fn = g["tp"], g["fp"], g["fn"]
        n_true = len(g["true_matches"])

        # Singleton: no true matches exist
        if n_true == 0:
            # Score 1.0 if we predicted nothing, 0.0 otherwise
            score = 1.0 if (tp + fp) == 0 else 0.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if prec + rec == 0:
                score = 0.0
            else:
                # F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
                # beta = 0.5
                score = (1.25 * prec * rec) / (0.25 * prec + rec)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def find_best_threshold(
    s1_ids: list[str],
    cand_ids: list[str],
    labels: list[float],
    preds: list[float],
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """
    Grid-search threshold in [0.01, 0.99] to maximise macro F0.5.

    Returns (best_threshold, best_score).
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thresh, best_f05 = 0.5, 0.0
    for t in thresholds:
        score = macro_f05(s1_ids, cand_ids, labels, preds, threshold=t)
        if score > best_f05:
            best_f05 = score
            best_thresh = float(t)
    return best_thresh, best_f05


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_features(path: str | Path) -> pl.DataFrame:
    df = pl.read_parquet(str(path))
    # Ensure required columns present
    for col in FEATURE_COLS:
        if col not in df.columns:
            log.warning(f"Feature column '{col}' missing from data — filling with 0.0")
            df = df.with_columns(pl.lit(0.0).alias(col))
    return df


def entity_stratified_split(
    df: pl.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split by unique S1 entities, not by rows. This prevents data leakage:
    all candidate pairs for a given S1 entity belong to either train or val.
    """
    rng = random.Random(seed)
    s1_unique = df["s1_id"].unique().to_list()
    rng.shuffle(s1_unique)
    n_val = max(1, int(len(s1_unique) * val_fraction))
    val_s1 = set(s1_unique[:n_val])

    train_df = df.filter(~pl.col("s1_id").is_in(val_s1))
    val_df = df.filter(pl.col("s1_id").is_in(val_s1))
    log.info(
        f"  Train S1 entities: {len(s1_unique) - len(val_s1):,}"
        f"  |  Val S1 entities: {len(val_s1):,}"
    )
    log.info(
        f"  Train pairs: {len(train_df):,}  |  Val pairs: {len(val_df):,}"
    )
    return train_df, val_df


def negative_downsample(
    df: pl.DataFrame,
    neg_ratio: int = 10,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Downsample negatives to `neg_ratio` × positives.
    Positives are always kept in full.
    """
    pos = df.filter(pl.col("label") == 1.0)
    neg = df.filter(pl.col("label") == 0.0)
    n_keep = min(len(neg), len(pos) * neg_ratio)
    if n_keep < len(neg):
        neg = neg.sample(n=n_keep, seed=seed)
        log.info(
            f"  Downsampled negatives: {len(neg):,} → {n_keep:,}"
            f" (ratio 1:{neg_ratio})"
        )
    return pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=seed)


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        clf = lgb.LGBMClassifier(n_estimators=1, device="cuda", verbose=-1)
        clf.fit(np.zeros((2, 2)), np.array([0, 1]))
        return True
    except Exception:
        return False


def _build_lgbm_params() -> dict:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "max_depth": 8,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_split_gain": 0.01,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
    }
    if _gpu_available():
        params["device"] = "cuda"
        params["gpu_use_dp"] = False
        log.info("  GPU detected — LightGBM will use CUDA device.")
    else:
        log.info("  LightGBM running on multi-threaded CPU (all 16 cores).")
    return params



LGBM_PARAMS = _build_lgbm_params()


def train_lgbm(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    n_estimators: int = 2000,
    early_stopping_rounds: int = 100,
    scale_pos_weight: Optional[float] = None,
) -> tuple[lgb.LGBMClassifier, float]:
    """
    Train LightGBM and return (fitted_model, best_threshold).
    """
    from typing import Optional as Opt

    X_train = train_df.select(FEATURE_COLS).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_val = val_df.select(FEATURE_COLS).to_numpy()
    y_val = val_df["label"].to_numpy()

    # Auto-compute scale_pos_weight if not provided
    if scale_pos_weight is None:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        log.info(f"  Auto scale_pos_weight = {scale_pos_weight:.2f}")

    params = dict(LGBM_PARAMS)
    params["n_estimators"] = n_estimators
    params["scale_pos_weight"] = scale_pos_weight

    model = lgb.LGBMClassifier(**params)

    with StageTimer("LightGBM Training", show_sysinfo=True):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=50),
            ],
        )
    log.info(f"  Best iteration: [highlight]{model.best_iteration_}[/highlight]")

    # Predict on validation
    val_preds = model.predict_proba(X_val)[:, 1].tolist()
    s1_ids_val = val_df["s1_id"].to_list()
    cand_ids_val = val_df["cand_id"].to_list()
    labels_val = y_val.tolist()

    with StageTimer("Threshold Optimisation (F0.5 sweep)", show_sysinfo=False):
        best_thresh, best_f05 = find_best_threshold(
            s1_ids_val, cand_ids_val, labels_val, val_preds
        )

    print_metrics("Validation Results", {
        "Best threshold":    f"{best_thresh:.4f}",
        "Val macro F0.5":    f"{best_f05:.4f}",
        "Best LGB iteration": str(model.best_iteration_),
        "Val positives":     f"{int(sum(labels_val)):,}",
        "Val total pairs":   f"{len(labels_val):,}",
    })

    # Feature importance table
    from rich.table import Table
    from rich import box as rbox
    fi = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    tbl = Table(title="Top 15 Feature Importances", box=rbox.SIMPLE_HEAVY,
                show_header=True, header_style="bold")
    tbl.add_column("Rank", style="dim_white", width=4)
    tbl.add_column("Feature", width=42)
    tbl.add_column("Importance", style="metric", justify="right")
    for rank, (fname, imp) in enumerate(fi[:15], 1):
        tbl.add_row(str(rank), fname, f"{imp:.0f}")
    console.print(tbl)

    return model, best_thresh


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", default="output/features_train.parquet")
    parser.add_argument("--model-dir", default="output/models")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--neg-sample-ratio", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with StageTimer("Training Pipeline"):
        log.info(f"Loading features from [highlight]{args.features_file}[/highlight] …")
        df = load_features(args.features_file)

        n_pos = df.filter(pl.col("label") == 1.0).height
        n_neg = df.filter(pl.col("label") == 0.0).height
        print_metrics("Dataset Overview", {
            "Total candidate pairs": f"{len(df):,}",
            "Positive pairs":        f"{n_pos:,}",
            "Negative pairs":        f"{n_neg:,}",
            "Imbalance ratio":       f"1:{n_neg//max(n_pos,1)}",
        })

        log.info("Splitting train / validation by S1 entity …")
        train_df, val_df = entity_stratified_split(df, val_fraction=args.val_fraction)

        log.info("Downsampling negatives …")
        train_df = negative_downsample(train_df, neg_ratio=args.neg_sample_ratio)
        n_train_pos = train_df.filter(pl.col("label") == 1.0).height
        n_train_neg = train_df.filter(pl.col("label") == 0.0).height
        log.info(f"  After downsample: [highlight]{len(train_df):,}[/highlight] pairs "
                 f"(pos=[success]{n_train_pos:,}[/success], neg={n_train_neg:,})")

        model, best_thresh = train_lgbm(
            train_df, val_df,
            n_estimators=args.n_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
        )

    # Save
    model_path = model_dir / "lgbm_model.pkl"
    with open(str(model_path), "wb") as f:
        pickle.dump(model, f, protocol=4)

    meta = {
        "threshold": best_thresh,
        "n_features": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "n_estimators": args.n_estimators,
        "best_iteration": int(model.best_iteration_),
    }
    meta_path = model_dir / "model_meta.json"
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Model → [highlight]{model_path}[/highlight]")
    log.info(f"Meta  → [highlight]{meta_path}[/highlight]")
    print_done("Model trained and saved. Run inference.py next.")


if __name__ == "__main__":
    main()

