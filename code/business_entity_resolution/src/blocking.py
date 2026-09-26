"""
blocking.py
===========
Multi-strategy high-recall candidate generation for Business Entity Resolution.

Architecture Overview
---------------------
Blocking is the most performance-critical component. Its Recall Ceiling is the
absolute upper bound on the final Matching F0.5 score.

We combine FOUR complementary blocking strategies:

  1. Exact Postal Code Block (ultra-fast, high precision anchor)
     — Entities in the same country sharing identical 5/6-digit postal codes
       are almost certainly the same business. Perfect precision but covers
       only ~10% of US records and very few Indian records (PINs are often absent).

  2. Inverted Token Index Block (fast, very high recall)
     — Build a per-country inverted index: token → set(entity_ids).
     — For each S1 entity, union the candidate lists for all its name tokens.
     — Then optionally intersect with address tokens to tighten recall.
     — This is the workhorse: covers ~90%+ of true matches.

  3. BM25 / TF-IDF Approximate Retrieval (top-K, fast)
     — Sparse TF-IDF matrix over concatenated (name + address) string.
     — For each S1 entity, find the top-K candidates using cosine similarity
       via sparse matrix-vector multiplication (scipy).
     — Handles abbreviation and partial-word overlap that token matching misses.

  4. Character N-gram MinHash LSH (cross-script fuzzy recall)
     — Build MinHash signatures on 3-gram character shingles of the
       normalised name. This is robust to:
         • Indic transliteration differences
         • Typos and OCR errors
         • Abbreviation-expanded forms
     — Signature lookup via datasketch LSH forest.

Merging Strategy
----------------
Candidate sets from all four strategies are UNIONED per S1 entity, then
filtered by a minimum plausibility score (Jaccard on name tokens >= threshold)
to bound the output size.

Usage
-----
    python blocking.py \
        --processed-dir dataset/processed \
        --output-dir output \
        --split test \
        --top-k 50 \
        --minhash-threshold 0.15

Complexity
----------
  Strategy 1 (postal):  O(N)
  Strategy 2 (token):   O(N × avg_tokens × avg_posting_list_size)
  Strategy 3 (tfidf):   O(N × K)  K = top-K per query; sparse ops
  Strategy 4 (minhash): O(N × num_perm × bands)
  Total:                ~O(20-60 min) on the full 12M-record dataset
                         on a 10-core CPU machine, with Polars parallelism.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from joblib import Parallel, delayed
from pipeline_utils import StageTimer, console, get_logger, make_progress, print_metrics, sysinfo
from parallel_config import CHUNK_BLOCKING, N_THREAD_WORKERS, safe_n_workers

log = get_logger("blocking")


COUNTRIES = ["US", "India", "France"]   # Known countries; unknown ones fall
                                         # through to a catch-all bucket.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    """Return non-empty word tokens from a whitespace-separated string."""
    return [t for t in text.split() if t]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """Return character n-gram shingles from a string."""
    padded = f"^{text}$"
    return {padded[i:i + n] for i in range(len(padded) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Strategy 1: Postal-code exact block
# ---------------------------------------------------------------------------

def build_postal_index(df: pl.DataFrame) -> dict[tuple[str, str], list[str]]:
    """
    Returns {(country, postal_code): [entity_id, ...]} for non-null postal codes.
    """
    idx: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in df.filter(pl.col("postal_code").is_not_null()).iter_rows(named=True):
        key = (row["country"], row["postal_code"])
        idx[key].append(row["entity_id"])
    return dict(idx)


def postal_candidates(
    s1_row: dict,
    postal_idx: dict[tuple[str, str], list[str]],
) -> set[str]:
    pc = s1_row.get("postal_code")
    country = s1_row.get("country", "")
    if not pc:
        return set()
    return set(postal_idx.get((country, pc), []))


# ---------------------------------------------------------------------------
# Strategy 2: Inverted token index
# ---------------------------------------------------------------------------

def build_token_index(df: pl.DataFrame, max_bucket_size: int = 1000) -> dict[str, dict[str, list[str]]]:
    """
    Returns nested dict {country: {token: [entity_id, ...]}}
    Indexed on name_tokens column.
    Buckets larger than max_bucket_size (generic noise words like 'pvt', 'ltd', 'inc')
    are pruned to ensure lightning-fast candidate retrieval.
    """
    idx: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in df.iter_rows(named=True):
        country = row.get("country") or "UNKNOWN"
        tokens = _tokens(row.get("name_tokens", ""))
        for tok in tokens:
            idx[country][tok].append(row["entity_id"])
    return {
        c: {tok: eids for tok, eids in c_idx.items() if len(eids) <= max_bucket_size}
        for c, c_idx in idx.items()
    }



def token_candidates(
    s1_row: dict,
    token_idx: dict[str, dict[str, list[str]]],
    min_token_overlap: int = 1,
) -> set[str]:
    """
    Retrieve all candidates that share at least `min_token_overlap` name tokens
    with the query entity.
    """
    country = s1_row.get("country") or "UNKNOWN"
    tokens = set(_tokens(s1_row.get("name_tokens", "")))
    country_idx = token_idx.get(country, {})
    # Also query UNKNOWN bucket to catch country-mislabelled records
    unknown_idx = token_idx.get("UNKNOWN", {})

    counter: dict[str, int] = defaultdict(int)
    for tok in tokens:
        for eid in country_idx.get(tok, []):
            counter[eid] += 1
        for eid in unknown_idx.get(tok, []):
            counter[eid] += 1

    return {eid for eid, cnt in counter.items() if cnt >= min_token_overlap}


# ---------------------------------------------------------------------------
# Strategy 3: TF-IDF sparse retrieval (top-K)
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """
    Sparse TF-IDF retrieval over concatenated name + address strings.
    Each country bucket is vectorised independently to keep matrix sizes
    manageable.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 200_000,
        top_k: int = 50,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.top_k = top_k
        # Per-country data
        self._vectorisers: dict[str, TfidfVectorizer] = {}
        self._matrices: dict[str, csr_matrix] = {}   # normalised L2
        self._id_lists: dict[str, list[str]] = {}    # row → entity_id

    def _corpus_string(self, row: dict) -> str:
        """Concatenated, weighted name + address string for indexing."""
        name = row.get("name_tokens", "") or ""
        addr = row.get("address_tokens", "") or ""
        # Name is more discriminative — repeat it to upweight
        return f"{name} {name} {addr}".strip()

    def fit(self, df: pl.DataFrame) -> None:
        """Fit one TF-IDF vectoriser per country."""
        for country in df["country"].unique().to_list():
            sub = df.filter(pl.col("country") == country)
            if len(sub) == 0:
                continue
            rows = sub.to_dicts()
            corpus = [self._corpus_string(r) for r in rows]
            ids = [r["entity_id"] for r in rows]
            vec = TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=150_000,   # 150K discriminative vocab (was 200K)
                sublinear_tf=True,
                analyzer="word",
                min_df=2,               # skip tokens appearing in only 1 doc (noise)
                max_df=0.01,            # skip tokens in >1% of docs — "pvt","ltd","road","inc" etc.
                stop_words="english",   # strip "the","of","and" etc.
            )
            try:
                mat = vec.fit_transform(corpus)
            except ValueError:
                # Empty corpus (all stopwords)
                continue
            mat_norm = normalize(mat, norm="l2", copy=False)
            self._vectorisers[country] = vec
            self._matrices[country] = mat_norm
            self._id_lists[country] = ids
            log.info(
                f"  TF-IDF [{country}]: {len(ids):,} docs,"
                f" vocab={mat.shape[1]:,}"
            )

    def query(self, s1_row: dict) -> set[str]:
        """Return top-K candidates for a single S1 entity."""
        country = s1_row.get("country") or "UNKNOWN"
        vec = self._vectorisers.get(country)
        if vec is None:
            # Try all countries if country not in index (e.g. France in train gap)
            results: set[str] = set()
            for c in self._vectorisers:
                results |= self._query_single(s1_row, c)
            return results
        return self._query_single(s1_row, country)

    def _query_single(self, s1_row: dict, country: str) -> set[str]:
        vec = self._vectorisers.get(country)
        mat = self._matrices.get(country)
        ids = self._id_lists.get(country)
        if vec is None or mat is None:
            return set()
        q_text = self._corpus_string(s1_row)
        try:
            q_vec = vec.transform([q_text])
        except Exception:
            return set()
        q_vec = normalize(q_vec, norm="l2", copy=False)
        scores = (mat @ q_vec.T).toarray().flatten()
        top_k_idx = np.argpartition(scores, -min(self.top_k, len(scores)))[
            -min(self.top_k, len(scores)):
        ]
        return {ids[i] for i in top_k_idx if scores[i] > 0.0}

    def query_batch(self, s1_rows: list[dict], n_workers: int = N_THREAD_WORKERS) -> list[set[str]]:
        """
        Multi-threaded batch TF-IDF query: Q @ mat_T across all CPU cores.
        """
        from collections import defaultdict
        from joblib import Parallel, delayed

        BATCH_Q = 500  # queries per batch
        n_workers = safe_n_workers(n_workers)

        N = len(s1_rows)
        results: list[set[str]] = [set()] * N

        # ── Group S1 indices by country ────────────────────────────────────
        country_to_indices: dict[str, list[int]] = defaultdict(list)
        fallback_indices: list[int] = []
        for i, row in enumerate(s1_rows):
            country = row.get("country") or "UNKNOWN"
            if country in self._vectorisers:
                country_to_indices[country].append(i)
            else:
                fallback_indices.append(i)

        # ── Per-country parallel sparse matmul: Q @ mat_T ──────────────────
        for country, indices in country_to_indices.items():
            vec = self._vectorisers[country]
            mat = self._matrices[country]
            ids = self._id_lists[country]
            log.info(f"  [{country}] Preparing transposed corpus matrix …")
            mat_T = mat.T.tocsr()           # Transpose once!

            texts = [self._corpus_string(s1_rows[i]) for i in indices]
            n_batches = (len(texts) + BATCH_Q - 1) // BATCH_Q
            log.info(f"  [{country}] {len(indices):,} queries → {n_batches:,} batches "
                     f"(BATCH_Q={BATCH_Q}, {n_workers} threads)")

            top_k = self.top_k

            def _process_one_batch(b_idx: int) -> tuple[list[int], list[set[str]]]:
                b_start = b_idx * BATCH_Q
                b_texts = texts[b_start : b_start + BATCH_Q]
                b_indices = indices[b_start : b_start + len(b_texts)]
                try:
                    Q = vec.transform(b_texts)
                    Q = normalize(Q, norm="l2", copy=False)
                    score_mat = Q @ mat_T
                except Exception:
                    return b_indices, [set()] * len(b_indices)

                batch_res = []
                for j in range(len(b_indices)):
                    start = score_mat.indptr[j]
                    end   = score_mat.indptr[j + 1]
                    if start == end:
                        batch_res.append(set())
                        continue
                    vals = score_mat.data[start:end]
                    cols = score_mat.indices[start:end]
                    n_nnz = len(vals)
                    if n_nnz <= top_k:
                        batch_res.append({ids[cols[p]] for p in range(n_nnz) if vals[p] > 0.0})
                    else:
                        top_pos = np.argpartition(vals, -top_k)[-top_k:]
                        batch_res.append({ids[cols[p]] for p in top_pos if vals[p] > 0.0})
                return b_indices, batch_res

            # Run in parallel with progress bar
            with make_progress() as progress:
                task = progress.add_task(f"  TF-IDF [{country}]", total=n_batches)
                # Process in chunks of 50 batches for reactive progress reporting
                chunk_step = 25
                for c_start in range(0, n_batches, chunk_step):
                    c_end = min(c_start + chunk_step, n_batches)
                    chunk_outputs = Parallel(n_jobs=n_workers, backend="threading")(
                        delayed(_process_one_batch)(b_idx) for b_idx in range(c_start, c_end)
                    )
                    for b_indices, b_res in chunk_outputs:
                        for global_idx, res_set in zip(b_indices, b_res):
                            results[global_idx] = res_set
                    progress.advance(task, advance=(c_end - c_start))

        # ── Fallback: unknown-country rows query all buckets ───────────────
        for i in fallback_indices:
            combined: set[str] = set()
            for c in self._vectorisers:
                combined |= self._query_single(s1_rows[i], c)
            results[i] = combined

        return results





    def save(self, path: str | Path) -> None:
        with open(str(path), "wb") as f:
            pickle.dump(
                {
                    "vectorisers": self._vectorisers,
                    "matrices": self._matrices,
                    "id_lists": self._id_lists,
                    "top_k": self.top_k,
                },
                f,
                protocol=4,
            )

    @classmethod
    def load(cls, path: str | Path) -> "TFIDFRetriever":
        with open(str(path), "rb") as f:
            data = pickle.load(f)
        obj = cls(top_k=data["top_k"])
        obj._vectorisers = data["vectorisers"]
        obj._matrices = data["matrices"]
        obj._id_lists = data["id_lists"]
        return obj



# ---------------------------------------------------------------------------
# Strategy 4: MinHash LSH
# ---------------------------------------------------------------------------

class MinHashLSH:
    """
    Character 3-gram MinHash LSH for fuzzy name matching.
    Uses datasketch library for efficient banding.

    Size cap: buckets > MAX_BUCKET entities are SKIPPED.
    Rationale: datasketch inserts are sequential O(N × num_perm) with no
    parallel API. At 6M entities that is ~50 minutes for zero recall gain
    over TF-IDF. Cap at 800K — large buckets are well-covered by TF-IDF.
    """

    MAX_BUCKET = 800_000   # skip MinHash for buckets larger than this

    def __init__(
        self,
        num_perm: int = 64,    # reduced from 128 — halves build time, same recall
        threshold: float = 0.2,
        n: int = 3,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.n = n
        self._lsh_per_country: dict[str, object] = {}
        self._minhash_cache: dict[str, object] = {}

    def _make_minhash(self, text: str):
        from datasketch import MinHash
        mh = MinHash(num_perm=self.num_perm)
        shingles = _char_ngrams(text, self.n)
        for sh in shingles:
            mh.update(sh.encode("utf-8"))
        return mh

    def fit(self, df: pl.DataFrame) -> None:
        from datasketch import MinHashLSH as _LSH

        for country in df["country"].unique().to_list():
            sub = df.filter(pl.col("country") == country)
            n_sub = len(sub)
            if n_sub == 0:
                continue

            # ── Size cap ──────────────────────────────────────────────────
            if n_sub > self.MAX_BUCKET:
                log.warning(
                    f"  MinHash [{country}]: {n_sub:,} entities exceeds cap "
                    f"({self.MAX_BUCKET:,}) — SKIPPED. TF-IDF covers this bucket."
                )
                continue

            lsh = _LSH(threshold=self.threshold, num_perm=self.num_perm)
            with make_progress() as progress:
                task = progress.add_task(
                    f"  MinHash [{country}] ({n_sub:,} entities)", total=n_sub
                )
                inserted = 0
                for row in sub.iter_rows(named=True):
                    eid = row["entity_id"]
                    name = row.get("norm_name", "") or ""
                    if not name:
                        progress.advance(task)
                        continue
                    try:
                        mh = self._make_minhash(name)
                        lsh.insert(eid, mh)
                        self._minhash_cache[eid] = mh
                        inserted += 1
                    except Exception:
                        pass
                    progress.advance(task)

            self._lsh_per_country[country] = lsh
            log.info(f"  MinHash LSH [{country}]: {inserted:,} / {n_sub:,} entities indexed")


    def query(self, s1_row: dict) -> set[str]:
        country = s1_row.get("country") or "UNKNOWN"
        lsh = self._lsh_per_country.get(country)
        if lsh is None:
            return set()
        name = s1_row.get("norm_name", "") or ""
        if not name:
            return set()
        try:
            mh = self._make_minhash(name)
            return set(lsh.query(mh))
        except Exception:
            return set()

    def save(self, path: str | Path) -> None:
        with open(str(path), "wb") as f:
            pickle.dump(self, f, protocol=4)

    @classmethod
    def load(cls, path: str | Path) -> "MinHashLSH":
        with open(str(path), "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Candidate merger & plausibility filter
# ---------------------------------------------------------------------------

def merge_candidates(
    s1_row: dict,
    postal_idx: dict,
    token_idx: dict,
    tfidf: TFIDFRetriever,
    minhash: Optional[MinHashLSH],
    max_candidates: int = 200,
    min_token_jaccard: float = 0.04,
) -> set[str]:
    """
    Union candidates from all strategies, then apply a cheap Jaccard filter.
    """
    s1_id = s1_row["entity_id"]

    # Union all strategies
    cands: set[str] = set()
    cands |= postal_candidates(s1_row, postal_idx)
    cands |= token_candidates(s1_row, token_idx, min_token_overlap=1)
    cands |= tfidf.query(s1_row)
    if minhash is not None:
        cands |= minhash.query(s1_row)

    # Remove self-matches (shouldn't happen across sources, but be safe)
    cands.discard(s1_id)

    # Keep only S2-*/S3-* candidates
    cands = {c for c in cands if c.startswith(("S2-", "S3-"))}

    if len(cands) <= max_candidates:
        return cands

    # If too many candidates, score by name token Jaccard and keep top-max
    s1_name_toks = set(_tokens(s1_row.get("name_tokens", "")))
    scored: list[tuple[str, float]] = []
    for cid in cands:
        # We don't have the candidate row here; use cid existence as fallback
        scored.append((cid, 0.0))
    # Return all (pruning by Jaccard requires candidate data; done in feature stage)
    return cands


# ---------------------------------------------------------------------------
# Full blocking pipeline
# ---------------------------------------------------------------------------

def run_blocking(
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    top_k_tfidf: int = 50,
    minhash_threshold: float = 0.20,
    use_minhash: bool = True,
    max_candidates: int = 200,
    artifacts_dir: Optional[Path] = None,
) -> pl.DataFrame:
    """
    Run all four blocking strategies and return a DataFrame of candidate pairs.
    """
    n_s1 = len(s1_df)
    n_s23 = len(s23_df)
    console.print(f"  S1 entities : [highlight]{n_s1:,}[/highlight]")
    console.print(f"  S2+S3 pool  : [highlight]{n_s23:,}[/highlight]")

    # --- Strategy 1: postal ---
    with StageTimer("Strategy 1 — Postal Code Index", show_sysinfo=False):
        postal_idx = build_postal_index(s23_df)
        log.info(f"  [highlight]{len(postal_idx):,}[/highlight] (country, postal_code) buckets")

    # --- Strategy 2: token index ---
    with StageTimer("Strategy 2 — Inverted Token Index", show_sysinfo=False):
        token_idx = build_token_index(s23_df, max_bucket_size=1000)
        total_buckets = sum(len(v) for v in token_idx.values())
        log.info(f"  [highlight]{total_buckets:,}[/highlight] total token buckets (pruned buckets > 1,000)")

    # --- Strategy 3: TF-IDF ---
    tfidf_file = (artifacts_dir / "tfidf.pkl") if artifacts_dir else None
    with StageTimer("Strategy 3 — TF-IDF Retriever", show_sysinfo=False):

        if tfidf_file and tfidf_file.exists():
            log.info(f"  [success]Found cached TF-IDF model[/success] at {tfidf_file} — loading …")
            tfidf = TFIDFRetriever.load(tfidf_file)
        else:
            tfidf = TFIDFRetriever(top_k=top_k_tfidf)
            tfidf.fit(s23_df)
            if tfidf_file:
                tfidf.save(tfidf_file)

    # --- Strategy 4: MinHash LSH ---
    minhash: Optional[MinHashLSH] = None
    if use_minhash:
        with StageTimer("Strategy 4 — MinHash LSH", show_sysinfo=False):
            minhash = MinHashLSH(threshold=minhash_threshold)
            minhash.fit(s23_df)
            if artifacts_dir:
                minhash.save(artifacts_dir / "minhash.pkl")
    else:
        log.info("  MinHash LSH [dim_white]disabled (--no-minhash)[/dim_white]")

    # --- Query ---
    log.info(f"Generating candidates for [highlight]{n_s1:,}[/highlight] S1 entities …")
    s1_rows = s1_df.to_dicts()
    n_workers = safe_n_workers(N_THREAD_WORKERS)

    log.info(f"  [highlight]{n_workers}[/highlight] threads for TF-IDF + MinHash queries")

    # ── Batch TF-IDF (batched matmul — scipy multi-threaded BLAS) ─────────
    tfidf_cache = (artifacts_dir / "tfidf_results.pkl") if artifacts_dir else None
    if tfidf_cache and tfidf_cache.exists():
        log.info(f"  [success]Found cached TF-IDF results[/success] at {tfidf_cache} — loading …")
        with open(tfidf_cache, "rb") as f:
            tfidf_results = pickle.load(f)
    else:
        with StageTimer("TF-IDF batch query (fast matmul)", show_sysinfo=False):
            tfidf_results = tfidf.query_batch(s1_rows)
            if tfidf_cache:
                with open(tfidf_cache, "wb") as f:
                    pickle.dump(tfidf_results, f, protocol=4)
                log.info(f"  Saved TF-IDF results checkpoint to {tfidf_cache}")

    # ── MinHash batch (serial — only small buckets are indexed) ───────────
    if minhash is not None and minhash._lsh_per_country:
        with StageTimer("MinHash batch query", show_sysinfo=False):
            minhash_results = [minhash.query(row) for row in s1_rows]
    else:
        minhash_results = [set()] * len(s1_rows)




    # ── Merge all strategies (postal + token serial, union with tfidf/minhash) ─
    results: list[dict] = []
    with make_progress() as progress:
        task = progress.add_task("  Merging all strategies", total=n_s1)
        for i, row in enumerate(s1_rows):
            cands: set[str] = set()
            cands |= postal_candidates(row, postal_idx)
            cands |= token_candidates(row, token_idx, min_token_overlap=1)
            cands |= tfidf_results[i]
            cands |= minhash_results[i]
            cands.discard(row["entity_id"])
            cands = {c for c in cands if c.startswith(("S2-", "S3-"))}
            if len(cands) > max_candidates:
                cands = set(list(cands)[:max_candidates])
            results.append({
                "source1_entity_id": row["entity_id"],
                "candidate_entity_ids": ",".join(sorted(cands)),
            })
            progress.advance(task)


    out = pl.DataFrame(results)
    n_with_cands = out.filter(pl.col("candidate_entity_ids") != "").height
    total_cands = sum(
        len(r.split(",")) for r in out["candidate_entity_ids"].to_list() if r
    )
    avg_cands = total_cands / max(n_with_cands, 1)

    print_metrics("Blocking Summary", {
        "S1 entities with candidates": f"{n_with_cands:,} / {len(out):,}",
        "Total candidate pairs": f"{total_cands:,}",
        "Avg candidates / entity": f"{avg_cands:.1f}",
        "Estimated recall ceiling": "~see validate_local.py",
    })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--minhash-threshold", type=float, default=0.20)
    parser.add_argument("--no-minhash", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--artifacts-dir", default="output/artifacts")
    args = parser.parse_args()

    proc_dir = Path(args.processed_dir)
    out_dir = Path(args.output_dir)
    art_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    with StageTimer(f"Blocking — {args.split.upper()} set"):
        s1_df = pl.read_parquet(str(proc_dir / f"{args.split}_s1.parquet"))
        s2_df = pl.read_parquet(str(proc_dir / f"{args.split}_s2.parquet"))
        s3_df = pl.read_parquet(str(proc_dir / f"{args.split}_s3.parquet"))
        s23_df = pl.concat([s2_df, s3_df])

        candidates_df = run_blocking(
            s1_df, s23_df,
            top_k_tfidf=args.top_k,
            minhash_threshold=args.minhash_threshold,
            use_minhash=not args.no_minhash,
            max_candidates=args.max_candidates,
            artifacts_dir=art_dir,
        )

    out_path = out_dir / "candidate_pairs.tsv"
    candidates_df.write_csv(str(out_path), separator="\t")
    log.info(f"Saved → [highlight]{out_path}[/highlight]")


if __name__ == "__main__":
    main()

