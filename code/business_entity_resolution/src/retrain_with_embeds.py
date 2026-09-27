"""
retrain_with_embeds.py
======================
Retrain LightGBM model with embedding cosine similarity feature filled in.
The existing features_train_hard.parquet has embed_cosine_sim = 0.0.
This script:
  1. Loads precomputed train embeddings from output/embeddings/
  2. Computes cosine similarity for all training pairs
  3. Fills in embed_cosine_sim in the training features
  4. Retrains LightGBM with all 39 features
  5. Calibrates threshold
  6. Saves new model to output/models/lgbm_model_v2.pkl

Run after train embeddings are computed:
    python code/business_entity_resolution/src/retrain_with_embeds.py
"""

import sys
import os
import json
import pickle
import logging
import time
import numpy as np
import polars as pl
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrain_with_embeds")

FEATURES_PATH = "output/features_train_hard.parquet"
EMBED_DIR = Path("output/embeddings")
MODEL_OUT = "output/models/lgbm_model_v2.pkl"
META_OUT = "output/models/model_meta_v2.json"

# Feature columns order (must match fast_features.py FEATURE_COLS)
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
    "embed_cosine_sim",
]


def load_embed_lookup(split: str = "train") -> dict:
    """Load precomputed embeddings and build (entity_id → embedding) lookup.
    Uses memory-mapped arrays for large files to avoid OOM.
    """
    log.info(f"Loading {split} embeddings from {EMBED_DIR} ...")
    s1_embeds = np.load(str(EMBED_DIR / f"{split}_s1_embeds.npy"), mmap_mode='r')
    s1_ids    = np.load(str(EMBED_DIR / f"{split}_s1_ids.npy"), allow_pickle=True)
    s23_embeds = np.load(str(EMBED_DIR / f"{split}_s23_embeds.npy"), mmap_mode='r')
    s23_ids   = np.load(str(EMBED_DIR / f"{split}_s23_ids.npy"), allow_pickle=True)

    log.info(f"  S1  embeds: {s1_embeds.shape}")
    log.info(f"  S23 embeds: {s23_embeds.shape}")

    # Build id → index map
    s1_idx  = {eid: i for i, eid in enumerate(s1_ids)}
    s23_idx = {eid: i for i, eid in enumerate(s23_ids)}

    return s1_embeds, s23_embeds, s1_idx, s23_idx


def compute_embed_cosine_sim(
    s1_ids: list, cand_ids: list,
    s1_embeds, s23_embeds, s1_idx, s23_idx,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Compute cosine similarity for all training pairs using vectorized dot product."""
    n = len(s1_ids)
    embed_sim = np.zeros(n, dtype=np.float32)
    missing = 0

    log.info(f"Computing embed_cosine_sim for {n:,} pairs ...")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch_s1 = s1_ids[start:end]
        batch_cand = cand_ids[start:end]

        s1_vecs   = []
        s23_vecs  = []
        valid_mask = []

        for sid, cid in zip(batch_s1, batch_cand):
            si = s1_idx.get(sid)
            ci = s23_idx.get(cid)
            if si is not None and ci is not None:
                s1_vecs.append(s1_embeds[si])
                s23_vecs.append(s23_embeds[ci])
                valid_mask.append(True)
            else:
                s1_vecs.append(None)
                s23_vecs.append(None)
                valid_mask.append(False)
                missing += 1

        # Vectorized dot product for valid pairs
        valid_idx = [j for j, v in enumerate(valid_mask) if v]
        if valid_idx:
            s1_mat  = np.stack([s1_vecs[j]  for j in valid_idx])
            s23_mat = np.stack([s23_vecs[j] for j in valid_idx])
            sims = (s1_mat * s23_mat).sum(axis=1)  # dot product of L2-normalized = cosine
            for k, j in enumerate(valid_idx):
                embed_sim[start + j] = float(sims[k])

        if start % 1_000_000 == 0:
            log.info(f"  {start:,} / {n:,} ({start/n*100:.1f}%) ...")

    if missing:
        log.warning(f"  {missing:,} pairs missing from embedding index (embed_cosine_sim=0.0)")
    return embed_sim


def main():
    os.chdir("D:/OpenSource/student_resource")
    log.info(f"Working dir: {os.getcwd()}")

    # Load training features
    log.info(f"Loading training features from {FEATURES_PATH} ...")
    feat_df = pl.read_parquet(FEATURES_PATH)
    log.info(f"  {len(feat_df):,} training pairs")
    log.info(f"  Label distribution: pos={feat_df['label'].sum():.0f}, neg={len(feat_df)-feat_df['label'].sum():.0f}")

    # Check if embed_cosine_sim is all zeros
    if "embed_cosine_sim" in feat_df.columns:
        mean_sim = feat_df["embed_cosine_sim"].mean()
        log.info(f"  Current embed_cosine_sim mean: {mean_sim:.4f}")
        if mean_sim > 0.001:
            log.info("  embed_cosine_sim already filled — skipping recomputation.")
            embed_already_filled = True
        else:
            embed_already_filled = False
    else:
        embed_already_filled = False

    if not embed_already_filled:
        # Load embeddings
        s1_embeds, s23_embeds, s1_idx, s23_idx = load_embed_lookup("train")

        # Compute embed cosine sims
        s1_ids   = feat_df["s1_id"].to_list()
        cand_ids = feat_df["cand_id"].to_list()
        embed_sims = compute_embed_cosine_sim(
            s1_ids, cand_ids, s1_embeds, s23_embeds, s1_idx, s23_idx
        )
        log.info(f"  embed_cosine_sim: mean={embed_sims.mean():.4f}, "
                 f"pos_mean={embed_sims[feat_df['label'].to_numpy().astype(bool)].mean():.4f}, "
                 f"neg_mean={embed_sims[~feat_df['label'].to_numpy().astype(bool)].mean():.4f}")

        # Update in features df
        feat_df = feat_df.with_columns(
            pl.Series("embed_cosine_sim", embed_sims)
        )

        # Save enriched features
        enriched_path = "output/features_train_hard_with_embeds.parquet"
        feat_df.write_parquet(enriched_path)
        log.info(f"  Saved enriched features → {enriched_path}")

    # Train/validation split (last 10% as validation)
    n = len(feat_df)
    n_val = n // 10
    n_train = n - n_val

    train_df = feat_df.head(n_train)
    val_df   = feat_df.tail(n_val)
    log.info(f"  Train: {len(train_df):,}, Val: {len(val_df):,}")

    X_train = train_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_train = train_df["label"].to_numpy().astype(np.int32)
    X_val   = val_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_val   = val_df["label"].to_numpy().astype(np.int32)

    # Train LightGBM
    log.info("Training LightGBM with embed_cosine_sim feature ...")
    import lightgbm as lgb

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    log.info(f"  Class weight: {pos_weight:.2f}")

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        n_jobs=16,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # Calibrate threshold on validation set
    from sklearn.metrics import f1_score
    val_probs = model.predict_proba(X_val)[:, 1]

    best_thresh, best_f05 = 0.5, 0.0
    for t in np.arange(0.20, 0.90, 0.01):
        preds = (val_probs >= t).astype(int)
        # Macro F0.5: weight precision 4x over recall
        from sklearn.metrics import fbeta_score
        f05 = fbeta_score(y_val, preds, beta=0.5, average="binary", zero_division=0)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t

    log.info(f"  Best val Macro F0.5: {best_f05:.4f} at threshold={best_thresh:.4f}")

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    log.info("  Top 10 features:")
    for fname, imp in feat_imp[:10]:
        log.info(f"    {fname}: {imp:.1f}")

    # Save model
    Path("output/models").mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f, protocol=4)
    log.info(f"  Saved model → {MODEL_OUT}")

    meta = {
        "threshold": float(best_thresh),
        "val_macro_f05": float(best_f05),
        "n_estimators": model.n_estimators_,
        "features": FEATURE_COLS,
        "embed_feature_included": True,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Saved meta → {META_OUT}")
    log.info(f"\n✓ Retrain complete. New model: {MODEL_OUT}")
    log.info(f"✓ Use threshold {best_thresh:.4f} for inference.")


if __name__ == "__main__":
    main()
