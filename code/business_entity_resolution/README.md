# Business Entity Resolution — Pipeline README

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (preprocess → block → train → infer → validate)
python src/run_pipeline.py

# 3. Upload output/matching_results.tsv to the leaderboard portal
```

## Pipeline Architecture

```
dataset/train/          dataset/test/
     ↓                       ↓
[preprocessing.py] ──────────────────→ dataset/processed/*.parquet
     ↓ (train)          ↓ (test)
[blocking.py]        [blocking.py]
     ↓                    ↓
 train_blocking/      output/candidate_pairs.tsv
 candidate_pairs.tsv
     ↓
[feature_engineering.py]
     ↓
 output/features_train.parquet
     ↓
[train.py]
     ↓
 output/models/lgbm_model.pkl
 output/models/model_meta.json   ←── best F0.5 threshold stored here
     ↓
[inference.py] ← uses test candidates + model
     ↓
 output/matching_results.tsv
     ↓
[validate_submission.py]  ← challenge-provided format checker
```

## Individual Stage Commands

```bash
# Preprocessing only
python src/preprocessing.py --train-dir dataset/train --test-dir dataset/test

# Blocking (train set — for generating training features)
python src/blocking.py --split train --output-dir output/train_blocking

# Feature engineering (training set)
python src/feature_engineering.py \
    --candidate-file output/train_blocking/candidate_pairs.tsv \
    --split train

# Training
python src/train.py --features-file output/features_train.parquet

# Blocking (test set)
python src/blocking.py --split test --output-dir output

# Inference (test set)
python src/inference.py

# Local validation against train ground truth
python src/validate_local.py

# Official submission format check
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Polars** for data loading | 5-10× faster than pandas for 12M row TSV files |
| **4-strategy blocking** | Postal + Token + TF-IDF + MinHash covers orthogonal recall regimes |
| **Indic transliteration** | 15% of Indian records use native script vs S1's Latin script |
| **Entity-stratified val split** | Prevents leakage: all pairs for an S1 entity go to one split |
| **DART booster** | Drop-connect regularisation prevents overfitting on the imbalanced dataset |
| **Macro F0.5 threshold sweep** | Direct optimisation of the leaderboard metric (not accuracy) |
| **Country cross-check post-filter** | Hard rule eliminates obvious false positives at near-zero recall cost |
| **Singleton protection** | Empty prediction = 1.0 score per entity; conservative threshold preserves this |

## Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| CPU Cores | 4 | 10+ |
| Disk | 10 GB | 20 GB |
| GPU | Not required | Not required |

## Reproducing Results

All random seeds are fixed (seed=42). Run `python src/run_pipeline.py` from
the `student_resource/` directory. Processing the full dataset takes
approximately 2–4 hours on a 10-core CPU machine.
