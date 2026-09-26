# Amazon ML Challenge 2026 — Business Entity Resolution
## Team Project Context, Pipeline Guide & Submission Handbook

> **Current Readiness Status:** ⚠️ **NOT YET SUBMISSION-READY**  
> We have completed Preprocessing, Dense Embeddings, Train Blocking, and Base Feature Engineering.  
> However, **Model Training (`output/models/`)**, **Test Blocking (`output/candidate_pairs.tsv`)**, and **Inference (`output/matching_results.tsv`)** must still be run to generate the submission files.

---

## 1. Challenge & Problem Overview

* **Task:** Entity Resolution (ER) across 3 heterogeneous, noisy business data sources:
  * **Source 1 (`S1-`):** Clean, deduplicated reference entities.
  * **Source 2 (`S2-`) & Source 3 (`S3-`):** Unlinked, noisy records containing abbreviations, missing fields, typos, and variations.
* **Goal:** For every Source 1 entity in the test set, find all matching entities from Source 2 and Source 3.
* **Evaluation Metric:** **Macro F0.5 Score** (precision is prioritized 4x more than recall to severely penalize false matches).
* **Countries:** Training data contains `US` and `India`. **Test data includes a third unseen country: `France`**. Every Source 1 entity (including France) must be present in the submission.

---

## 2. Current Progress & Artifact Status

| Component | Status | Artifact / Output Location | Notes |
| :--- | :---: | :--- | :--- |
| **Data Preprocessing** | ✅ Done | `dataset/processed/*.parquet` | Cleaned names, normalized addresses, legal forms |
| **Dense Embeddings** | ✅ Done | `output/embeddings/` (~18 GB) | S1 & S2/S3 dense vectors (`.npy`) |
| **Train Blocking** | ✅ Done | `output/train_blocking/candidate_pairs.tsv` (2.67 GB) | Multi-strategy candidate retrieval for train set |
| **Feature Engineering** | ✅ Done | `output/features_train_base.parquet` (415 MB) | String, token, address, and embedding similarities |
| **Model Training** | ⏳ **Pending** | `output/models/lgbm_model.pkl` | LightGBM classifier with F0.5 threshold optimization |
| **Test Set Blocking** | ⏳ **Pending** | `output/candidate_pairs.tsv` | Candidates for test set Source 1 entities |
| **Test Inference** | ⏳ **Pending** | `output/matching_results.tsv` | Final predictions for leaderboard submission |
| **Format Validation** | ⏳ **Pending** | `utils/validate_submission.py` | Must pass before portal upload |
| **Final Zip Package** | ⏳ **Pending** | `<team_name>_submission.zip` | Required for final round verification |

---

## 3. Repository Architecture & Script Reference

```
student_resource/
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt           # Python package requirements
│       ├── README.md                  # Pipeline architecture & quick start
│       └── src/
│           ├── preprocessing.py       # Normalization, tokenization & parquet caching
│           ├── blocking.py            # Multi-strategy candidate generator (TF-IDF, MinHash, Postcodes)
│           ├── fast_features.py       # High-speed string, phonetic & token similarity metrics
│           ├── feature_engineering.py # Assembles full feature matrix for candidate pairs
│           ├── embed_features.py      # Dense semantic sentence transformer embeddings
│           ├── parallel_config.py     # Worker, memory & thread management
│           ├── pipeline_utils.py      # Rich logging, stage timers & I/O helpers
│           ├── train.py               # LightGBM training + F0.5 threshold tuning
│           ├── inference.py           # Evaluates test candidate pairs with trained model
│           ├── validate_local.py      # Held-out train set evaluation & metric reporting
│           └── run_pipeline.py        # Master pipeline orchestrator
├── utils/
│   ├── validate_submission.py         # Official submission format & integrity validator
│   └── upload_to_hf.py                # Hugging Face uploader for ~23GB large outputs
├── dataset/                           # Raw TSV and processed parquet files (Excluded from Git)
├── output/                            # Generated models, embeddings, candidates (Excluded from Git)
├── Documentation_template.md          # Official report template (fill before final submission)
├── context.md                         # This team onboarding & operational reference guide
└── .gitignore                         # Excludes >3GB dataset and >23GB large output artifacts
```

### Script Reference Details:

* **`src/preprocessing.py`**:
  * Cleans company names (removes legal noise, punctuation, lowercase), standardizes addresses, parses PIN/postal codes.
  * Outputs fast binary Arrow format to `dataset/processed/`.
* **`src/blocking.py`**:
  * High-recall candidate generator combining:
    1. Postal code / geographic bucket matching
    2. Inverted token index
    3. Batched sparse TF-IDF retriever
    4. MinHash LSH (with bucket caps for scalability)
  * Supports `--split train` (for training set) or `--split test` (for test set).
* **`src/fast_features.py` & `src/feature_engineering.py`**:
  * Calculates Levenshtein ratio, token sort/set ratio, Jaccard similarity, 3-gram overlap, exact token matches, and geographic compatibility.
* **`src/train.py`**:
  * Fits LightGBM classifier on `output/features_train.parquet`.
  * Grid searches the classification threshold specifically optimizing **macro F0.5 score**.
  * Saves `output/models/lgbm_model.pkl` and `output/models/model_meta.json`.
* **`src/inference.py`**:
  * Runs test candidates through the feature pipeline and trained LightGBM model.
  * Filters pairs using the tuned threshold from `model_meta.json`.
  * Produces `output/matching_results.tsv`.
* **`utils/validate_submission.py`**:
  * Validates structure, headers, row count, formatting, and sanity checks against `dataset/test`.

---

## 4. Instructions for Teammates

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/NayandG07/ML-Hackathon-Amazon.git
cd ML-Hackathon-Amazon

# Create and activate virtual environment (Python 3.10 - 3.13)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

### Step 2: Datasets & Large Artifacts Handling
* **Why are `dataset/` and `output/` not in Git?**
  * Raw dataset files are **> 4.5 GB** and output embeddings are **> 23 GB**. GitHub has a strict 100 MB per-file limit.
* **Where to place the dataset:**
  * Download the raw TSV files into `dataset/train/` and `dataset/test/` matching the project root.
* **Shared Heavy Artifacts:**
  * Pre-generated embeddings and large caches are hosted on Hugging Face at `Ndg07/ML-Hackathon-Amazon` (or shared via team drive).

---

## 5. How to Complete the Pipeline & Make it Submission-Ready

If the base training features (`output/features_train_base.parquet`) are already generated, complete the remaining steps as follows:

### Step 1: Train the Matching Model
```bash
python code/business_entity_resolution/src/train.py \
    --features-file output/features_train_base.parquet \
    --model-dir output/models \
    --n-estimators 2000
```
*Outputs: `output/models/lgbm_model.pkl` and `output/models/model_meta.json`.*

### Step 2: Generate Candidate Pairs for the Test Set (Blocking)
```bash
python code/business_entity_resolution/src/blocking.py \
    --split test \
    --processed-dir dataset/processed \
    --output-dir output \
    --top-k 50
```
*Outputs: `output/candidate_pairs.tsv`.*

### Step 3: Run Inference to Produce Final Matches
```bash
python code/business_entity_resolution/src/inference.py \
    --processed-dir dataset/processed \
    --candidate-file output/candidate_pairs.tsv \
    --model-dir output/models \
    --output-dir output
```
*Outputs: `output/matching_results.tsv`.*

### Step 4: Validate Format Before Leaderboard Upload
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
*The command must print `PASS` with exit code 0.*

---

## 6. Official Submission Guidelines & Specifications

### A. Live Leaderboard Uploads (`matching_results.tsv`)
Upload this file directly to the hackathon portal:
* **File format:** Tab-separated (`.tsv`), UTF-8 encoded.
* **Header:** Exactly `source1_entity_id\tmatched_entity_ids` (case-sensitive).
* **Row count:** Exactly **one row per Source 1 entity in `test_source1.tsv`**.
* **Ordering:** Must match the exact order of entities in `test_source1.tsv`.
* **Matched IDs format:** Comma-separated without spaces or quotes (e.g., `S2-00047,S3-00812`).
* **Empty matches:** Leave column empty after the tab if no match was found (do **NOT** write `None`, `null`, or `[]`).
* **ID constraints:** Only `S2-` and `S3-` IDs that actually exist in `test_source2.tsv` and `test_source3.tsv`. No `S1-` IDs. No duplicate IDs.
* **Open Country Set:** Remember `test_source1.tsv` contains entities from `US`, `India`, and **`France`**. All France entities must be represented.

### B. Blocking Candidate Set (`candidate_pairs.tsv`)
* **Header:** `source1_entity_id\tcandidate_entity_ids`
* **Subset Rule:** Every ID in `matching_results.tsv` **must also appear** in `candidate_pairs.tsv`. Any matched ID not in the candidate set indicates a pipeline bug.

### C. Final Package ZIP Submission (`<team_name>_submission.zip`)
At the end of the competition, top teams must submit the full reproducibility archive:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Identical to your best leaderboard submission
│   └── candidate_pairs.tsv         # Your final test blocking candidates
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Complete source code
│       ├── README.md               # Reproducibility instructions
│       └── requirements.txt        # Exact dependencies
└── Documentation_template.md       # Fully completed methodology write-up
```
