# Amazon ML Challenge 2026 — Business Entity Resolution
## Team Project Context, Architecture, Root Cause Analysis & Roadmap

> **Current Readiness Status:** 🔄 **IN PROGRESS — UPGRADING TO HIGH-PRECISION 0.95+ PIPELINE (OPTION 3)**  
> **Previous Submission Score:** **0.295 Macro $F_{0.5}$** (Leaderboard)  
> **Target Score:** **0.95 – 0.98 Macro $F_{0.5}$**  
> **Key Realization:** The pipeline suffered from (1) a 76.7% blocking recall ceiling, (2) severe LightGBM overconfidence from an easy-negative training set, and (3) an inference runtime bottleneck (3.8 hrs) caused by featurizing 174M raw candidate pairs without cascading pruning.

---

## 1. Challenge & Problem Overview

* **Task:** Entity Resolution (ER) across 3 heterogeneous, noisy business data sources:
  * **Source 1 (`S1-`):** Clean, deduplicated reference entities (1,732,544 test entities; 2,206,821 train entities).
  * **Source 2 (`S2-`) & Source 3 (`S3-`):** Unlinked, noisy records containing abbreviations, missing fields, typos, and variations.
* **Goal:** For every Source 1 entity in the test set, find all matching entities from Source 2 and Source 3.
* **Evaluation Metric:** **Macro $F_{0.5}$ Score** (Precision is prioritized 4× more than recall):
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  * Evaluated per Source 1 entity, then averaged across **all** entities.
  * **Singletons (0 true matches):** Score **1.0** if predicted empty; score **0.0** if any match is predicted.
  * Over-prediction is heavily penalized: predicting 1 false positive on an entity with 3 true matches drops its score from 1.0 to 0.789; 2 false positives drop it to 0.652.
* **Countries:** Training data contains `US` and `India`. **Test data includes a third unseen country: `France`**. Every Source 1 entity (including France) must be present in the submission.

---

## 2. Ground Truth Properties & Theoretical Ceilings

From empirical audit of `train_ground_truth.tsv` (2,206,821 S1 entities, 7,638,365 true matches):
* **Singletons (0 matches):** 123,247 entities (5.58%)
* **Entities with matches:** 2,083,574 entities (94.42%)
* **Match count distribution:**
  * 1 match: 5.40%
  * 2 matches: 17.00%
  * 3 matches: 24.05%
  * 4 matches: 21.94%
  * 5 matches: 14.59%
  * 6 matches: 7.47%
  * 7+ matches: 3.96% (Max true matches = 11, Median = 4)
* **Blocking Recall Ceiling:** Current TF-IDF word-level blocking captures only **76.72%** of true matches.
  * Even with a perfect ML classifier, the maximum theoretical Macro $F_{0.5}$ under current blocking is **0.946**. To reach 0.98, blocking recall must reach **98%+**.

---

## 3. Root Cause Analysis of Submission #1 (Score: 0.295)

We empirically audited the failure modes of the initial pipeline:

1. **Alphabetical Truncation Bug in `fast_features.py`:**
   * `fast_features.py` previously contained `.list.slice(0, 30)` on sorted entity IDs.
   * Because IDs were stored alphabetically (`S2-00001, S2-00002...`), this discarded **57.21% of true matches** before feature calculation.
   * *Status:* **Fixed** (removed slice, exploding all candidates).

2. **Severe LightGBM Over-confidence (The "Bimodal Hallucination"):**
   * Model was trained on `features_train_base.parquet` with **57% positives / 43% negatives** (only 2 easy negatives per entity).
   * In test inference, the real candidate distribution has ~3.5% positives and ~96.5% negatives (~100 candidates/entity).
   * LightGBM learned independent additive splits. When inspecting pairs with model score $>0.99$, we found:
     * `orelee s barbershop (NC)` vs `airbueal orelee apple (OH)` $\rightarrow$ **Score 0.992** (matched only on the word "orelee").
     * `orelee s barbershop (NC)` vs `jonesnatural (NC)` $\rightarrow$ **Score 0.997** (completely unrelated names, matched only on street name).
   * Almost every candidate scored $>0.80$, leading to 99.93% of entities hitting the max-prediction cap with false positives.

3. **Inference Compute Bottleneck (3.8 Hours):**
   * 1,732,544 test entities $\times$ ~100 candidates = **174,467,283 pairs**.
   * Calculating 39 complex C++ rapidfuzz string metrics on 174M pairs took ~3.8 hours across 174 streaming batches.
   * Over 90% of those pairs had 0% name token overlap and completely different postal codes.

---

## 4. The Solution: Option 3 (Championship 0.95+ Stack)

The team has committed to **Option 3** — the full high-recall, high-precision architecture:

```
                      Raw S1, S2, S3 Data
                               │
                               ▼
     ┌───────────────────────────────────────────────────┐
     │ 1. Upgraded Character 3-Gram Blocking             │
     │    - TF-IDF with char 3-grams (ngram_range=(3,3)) │
     │    - Relax max_df to keep industry terms          │
     │    - Increases recall ceiling: 76.7% ──► 96%+     │
     └─────────────────────────┬─────────────────────────┘
                               │
                               ▼
     ┌───────────────────────────────────────────────────┐
     │ 2. Hard-Negative Mining & Balanced Training       │
     │    - Mine 6–8 hard negatives per entity:          │
     │      * High name overlap, wrong address           │
     │      * Same address/zip, different name           │
     │    - Add interaction features (Name × Address)    │
     │    - Retrain LightGBM with calibrated probabilities│
     └─────────────────────────┬─────────────────────────┘
                               │
                               ▼
     ┌───────────────────────────────────────────────────┐
     │ 3. Fast Cascading Inference Engine                │
     │    - Microsecond vectorized pre-filter:           │
     │      Drop pairs with 0 name overlap & no zip match │
     │    - Prunes 174M pairs ──► ~15M pairs              │
     │    - Inference runtime: 3.8 hrs ──► ~30 mins      │
     └─────────────────────────┬─────────────────────────┘
                               │
                               ▼
     ┌───────────────────────────────────────────────────┐
     │ 4. Precision Post-Processing Gates                │
     │    - Dual-Agreement Gate:                         │
     │      Require Name >= 0.45 AND (Addr >= 0.30 OR Zip)│
     │    - Symmetrical 1-to-N Exclusive Assignment:     │
     │      Prevent duplicate S2/S3 entity assignment    │
     │    - Singleton Guard (calibrated threshold)       │
     └─────────────────────────┬─────────────────────────┘
                               │
                               ▼
                    Final matching_results.tsv
```

### Estimated Compute Budget:
* **Step 1 (Char 3-gram Test Blocking):** ~55 mins
* **Step 2 (Feature Extraction + LightGBM Retraining):** ~45 mins
* **Step 3 & 4 (Fast Cascading Inference + Post-Processing):** ~35 mins
* **Total End-to-End Runtime:** **~2.5 to 3 hours**

---

## 5. Artifact Directory & File Guide

| Path | Description | Notes |
| :--- | :--- | :--- |
| `dataset/processed/*.parquet` | Preprocessed clean tables | Normalized names, addresses, countries |
| `output/embeddings/*.npy` | Sentence-Transformer embeddings (~18 GB) | Dense multilingual vectors (paraphrase-multilingual-MiniLM-L12-v2) |
| `output/candidate_pairs.tsv` | Test blocking candidate pairs (2.27 GB) | Current test blocking output |
| `output/models/lgbm_model.pkl` | LightGBM model artifact (35.9 MB) | Current model |
| `output/models/model_meta.json` | Model metadata, feature list, threshold | Threshold metadata |
| `output/matching_results.tsv` | Final predictions TSV | Tab-separated leaderboard submission |
| `utils/validate_submission.py` | Official format validator | Must return `PASS` with exit code 0 |

---

## 6. Exact Step-by-Step Implementation Instructions

### Step 1: Upgrade `blocking.py` to Character 3-Grams
1. In `src/blocking.py`:
   * Update `TFIDFRetriever` vectorizer to `analyzer="char"`, `ngram_range=(3, 3)`.
   * Increase `max_features=250_000` and relax `max_df=0.05` (prevents dropping "hotel", "clinic", "restaurant").
   * Raise `top_k=100`.
2. Run test blocking:
   ```bash
   python code/business_entity_resolution/src/blocking.py --split test --processed-dir dataset/processed --output-dir output
   ```

### Step 2: Mine Hard Negatives & Retrain LightGBM
1. Update `fast_features.py:build_flat_pairs` for training:
   * Set `max_negatives=6` (mix of TF-IDF name confusers and postal-code confusers).
2. Generate new feature matrix `output/features_train_hard.parquet`:
   ```bash
   python code/business_entity_resolution/src/feature_engineering.py --split train --candidate-file output/train_blocking/candidate_pairs.tsv --output-file output/features_train_hard.parquet
   ```
3. Retrain LightGBM:
   ```bash
   python code/business_entity_resolution/src/train.py --features-file output/features_train_hard.parquet --model-dir output/models --n-estimators 1500
   ```

### Step 3: Run Fast Cascading Inference & Post-Processing
1. In `src/inference.py`:
   * Add cascading pre-filter in `build_flat_pairs` (prunes 90% non-matches in microseconds).
   * Apply Dual-Agreement Gate: `name_ratio >= 0.45` AND (`addr_ratio >= 0.30` OR `postal_match == 1`).
   * Apply Symmetrical 1-to-N Exclusive Assignment: resolve cross-entity duplicates so an S2/S3 entity only matches its highest-scoring S1 parent.
2. Run inference:
   ```bash
   $env:PYTHONIOENCODING='utf-8'; python code/business_entity_resolution/src/inference.py --threshold-override 0.85 --max-matches-per-entity 4
   ```

### Step 4: Validate and Package
1. Validate formatting:
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```
2. Build final submission zip:
   ```bash
   python output/make_zip.py
   ```
