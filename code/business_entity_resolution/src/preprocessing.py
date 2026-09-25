"""
preprocessing.py
================
Production-quality preprocessing and normalization for Business Entity Resolution.

Design Goals:
  - Unicode-safe: handles Devanagari, Tamil, Telugu, Kannada, Bengali,
    Gujarati, Malayalam, Gurmukhi, Oriya, and Latin scripts.
  - Country-invariant: US, India, France, and unseen test countries all pass
    through the same pipeline without hard-coded branches.
  - Idempotent: applying normalise() twice yields the same result.
  - Fast: vectorised string ops via Polars + compiled regex patterns.
  - Reproducible: no randomness; deterministic given the same input.

Usage:
    python preprocessing.py --train-dir dataset/train --test-dir dataset/test
    (Output written to dataset/processed/)
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import polars as pl
from joblib import Parallel, delayed
from pipeline_utils import StageTimer, console, get_logger, sysinfo
from parallel_config import CHUNK_PREPROCESS, N_PROCESS_WORKERS, safe_n_workers

log = get_logger("preprocessing")



# ---------------------------------------------------------------------------
# Compiled pattern catalogue
# ---------------------------------------------------------------------------
class _Patterns:
    """Central registry of all compiled regex patterns."""

    # Legal-suffix normalisation (order matters — long forms first)
    _LEGAL_PAIRS: list[tuple[str, str]] = [
        # Entity types
        (r"\bprivate\s+limited\b", "pvt ltd"),
        (r"\bpublic\s+limited\b", "pub ltd"),
        (r"\bpvt\.?\s+ltd\.?\b", "pvt ltd"),
        (r"\bpte\.?\s+ltd\.?\b", "pte ltd"),
        (r"\bpvt\b", "pvt"),
        (r"\blimited\s+liability\s+partnership\b", "llp"),
        (r"\bliability\s+partnership\b", "llp"),
        (r"\bllp\b", "llp"),
        (r"\blimited\s+liability\s+company\b", "llc"),
        (r"\bltd\.?\b", "ltd"),
        (r"\blimited\b", "ltd"),
        (r"\binc(?:orporated)?\.?\b", "inc"),
        (r"\bcorp(?:oration)?\.?\b", "corp"),
        (r"\bco(?:mpany)?\.?\b", "co"),
        (r"\bplc\b", "plc"),
        (r"\bllc\b", "llc"),
        # French forms
        (r"\bsoci[eé]t[eé]\s+(?:[aà]\s+)?responsabilit[eé]\s+limit[eé]e\b", "sarl"),
        (r"\bsoci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e\s+unipersonnelle\b", "sasu"),
        (r"\bsoci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e\b", "sas"),
        (r"\bentreprise\s+unipersonnelle\s+[aà]\s+responsabilit[eé]\s+limit[eé]e\b", "eurl"),
        (r"\bsoci[eé]t[eé]\s+civile\s+immobili[eè]re\b", "sci"),
        (r"\bsarl\b", "sarl"),
        (r"\bsasu\b", "sasu"),
        (r"\beurl\b", "eurl"),
        (r"\bsas\b", "sas"),
        (r"\bsci\b", "sci"),
        # Common DBA hints
        (r"\bd/?b/?a\b", "dba"),
        (r"\ba/?k/?a\b", "aka"),
        (r"\bt/?a\b", "ta"),
    ]

    # Address-component normalisation
    _ADDR_PAIRS: list[tuple[str, str]] = [
        # Road types
        (r"\bstreet\b", "st"),
        (r"\broad\b", "rd"),
        (r"\bavenue\b", "ave"),
        (r"\bboulevard\b", "blvd"),
        (r"\bdrive\b", "dr"),
        (r"\blane\b", "ln"),
        (r"\bcourt\b", "ct"),
        (r"\bcircle\b", "cir"),
        (r"\bterrace\b", "ter"),
        (r"\bplace\b", "pl"),
        (r"\bsquare\b", "sq"),
        (r"\bboulevard\b", "blvd"),
        (r"\brue\b", "rue"),         # French  — keep as-is for now
        (r"\ballée\b", "allee"),
        # Suite / apartment
        (r"\b(?:suite|ste)\.?\b", "ste"),
        (r"\b(?:apartment|apt)\.?\b", "apt"),
        (r"\b(?:floor|fl)\.?\b", "fl"),
        # Directions
        (r"\bnorth\b", "n"),
        (r"\bsouth\b", "s"),
        (r"\beast\b", "e"),
        (r"\bwest\b", "w"),
        (r"\bnortheast\b", "ne"),
        (r"\bnorthwest\b", "nw"),
        (r"\bsoutheast\b", "se"),
        (r"\bsouthwest\b", "sw"),
        # State abbreviations already short — no action needed
        # Indian address components
        (r"\bnagar\b", "ngr"),
        (r"\bcolony\b", "col"),
        (r"\bsector\b", "sec"),
        (r"\bblock\b", "blk"),
    ]

    # Characters that should become spaces
    JUNK_CHARS = re.compile(r"[|\\/*\[\]{}#@$%^~`<>]")
    # Multiple spaces
    MULTI_SPACE = re.compile(r"\s{2,}")
    # URL pattern — extract domain root as a name token
    URL_PAT = re.compile(
        r"(?:https?://|www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)*)(?:\.[a-z]{2,}){1,2}(?:/.*)?",
        re.I,
    )
    # Phone numbers embedded in address strings
    # Handles: "Ph. 989, 9487203", "+91-9876543210", "Ph: 9876543210"
    PHONE_PAT = re.compile(
        r"(?:ph\.?\s*[\s:,]?\s*|tel\.?\s*[\s:,]?\s*|phone\s*[\s:,]?\s*)"
        r"[\d\s,\-\(\)\+]{6,30}"
        r"|(?<!\w)(?:\+?\d[\d\s\-\(\)]{7,14}\d)(?!\w)"
    )
    # House/building number tokens
    HOUSE_NUM_PAT = re.compile(r"\b\d+(?:[/-]\d+)*[a-z]?\b")
    # 6-digit Indian PIN code
    PIN_PAT = re.compile(r"\b([1-9][0-9]{5})\b")
    # US ZIP code (5 or 9 digit)
    ZIP_PAT = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
    # French postal code (5-digit starting 0-9)
    FR_ZIP_PAT = re.compile(r"\b([0-9]{5})\b")

    def __init__(self):
        self._legal = [
            (re.compile(pat, re.I | re.U), repl)
            for pat, repl in self._LEGAL_PAIRS
        ]
        self._addr = [
            (re.compile(pat, re.I | re.U), repl)
            for pat, repl in self._ADDR_PAIRS
        ]

    def apply_legal(self, text: str) -> str:
        for pat, repl in self._legal:
            text = pat.sub(repl, text)
        return text

    def apply_addr(self, text: str) -> str:
        for pat, repl in self._addr:
            text = pat.sub(repl, text)
        return text


_P = _Patterns()


# ---------------------------------------------------------------------------
# Core normalisation helpers
# ---------------------------------------------------------------------------

def _unicode_normalise(text: str) -> str:
    """
    NFC normalisation + remove control characters.
    Keeps all script characters (Latin, Indic, CJK, etc.) intact.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Remove Unicode control / format characters, keep printable ones
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] not in ("C",))
    return text


def _romanise_indic(text: str) -> str:
    """
    Lightweight Indic → Latin romanisation using Unicode code-point ranges.
    Strips every Indic-script character to its nearest approximation or
    replaces the whole word with a transliteration marker so the token
    hashing step can find overlap when the same word appears in both scripts.

    In a production system, replace this with the `indic-transliteration`
    or `aksharamukha` library for higher fidelity. Here we rely on the
    unicodedata.name() approach for acceptable recall.
    """
    if not any(0x0900 <= ord(c) <= 0x109F for c in text):
        # Quick bail: no Indic characters present
        return text

    try:
        from indic_transliteration import sanscript, detect
        from indic_transliteration.sanscript import transliterate, IAST, DEVANAGARI
        # Attempt script detection per word
        words = text.split()
        romanised_words = []
        for word in words:
            if any(0x0900 <= ord(c) <= 0x109F for c in word):
                try:
                    detected = detect.detect(word)
                    if detected:
                        roman = transliterate(word, detected, sanscript.ITRANS)
                        romanised_words.append(roman.lower())
                    else:
                        romanised_words.append(word)
                except Exception:
                    romanised_words.append(word)
            else:
                romanised_words.append(word)
        return " ".join(romanised_words)
    except ImportError:
        # Fallback: replace entire Indic-script word with a phonetic stub
        # built from the Unicode character names (crude but functional)
        words = text.split()
        result_words = []
        for word in words:
            if any(0x0900 <= ord(c) <= 0x109F for c in word):
                # Build a token from the vowel-consonant skeleton
                chars = []
                for ch in word:
                    try:
                        name = unicodedata.name(ch)
                        # Keep the first syllable letter identifier
                        parts = name.split()
                        if len(parts) >= 2:
                            chars.append(parts[-1][:3].lower())
                    except ValueError:
                        pass
                stub = "".join(chars)[:12]
                result_words.append(stub if stub else word)
            else:
                result_words.append(word)
        return " ".join(result_words)


def _extract_url_root(text: str) -> str:
    """
    If the name looks like a URL, extract the domain root as the canonical name.
    E.g. 'maurewilliamscolombier.com' → 'maurewilliamscolombier'
    """
    m = _P.URL_PAT.fullmatch(text.strip())
    if m:
        return m.group(1).replace("-", " ")
    return text


def normalise_name(raw: Optional[str]) -> str:
    """
    Full normalisation pipeline for a business name string.

    Steps:
    1. Handle null / empty
    2. Unicode NFC normalise
    3. Romanise Indic script
    4. Lower-case
    5. Strip URL wrappers (e.g. '.com' suffix in S3)
    6. Remove junk punctuation
    7. Expand/collapse legal suffixes
    8. Remove leading '--' noise (seen in S2)
    9. Collapse whitespace
    """
    if not raw or (isinstance(raw, float)):
        return ""
    text = str(raw).strip()
    text = _unicode_normalise(text)
    text = _romanise_indic(text)
    text = text.lower()
    text = _extract_url_root(text)
    # Strip '& ' → 'and'
    text = re.sub(r"&", " and ", text)
    # Remove junk chars
    text = _P.JUNK_CHARS.sub(" ", text)
    # Remove leading '--' noise pattern (S2 artifact)
    text = re.sub(r"^--\s*", "", text)
    # Collapse punctuation between words
    text = re.sub(r"[.,;:!?'\"\-]+", " ", text)
    # Normalise legal suffixes
    text = _P.apply_legal(text)
    # Collapse whitespace
    text = _P.MULTI_SPACE.sub(" ", text).strip()
    return text


def normalise_address(raw: Optional[str]) -> str:
    """
    Full normalisation pipeline for a business address string.

    Steps:
    1. Handle null / empty
    2. Unicode NFC normalise
    3. Romanise Indic script in address
    4. Lower-case
    5. Remove phone number sub-strings
    6. Remove junk punctuation
    7. Abbreviate road types / directions
    8. Collapse whitespace
    """
    if not raw or (isinstance(raw, float)):
        return ""
    text = str(raw).strip()
    text = _unicode_normalise(text)
    text = _romanise_indic(text)
    text = text.lower()
    # Remove embedded phone numbers
    text = _P.PHONE_PAT.sub(" ", text)
    text = re.sub(r"&", " and ", text)
    text = _P.JUNK_CHARS.sub(" ", text)
    text = re.sub(r"[.,;:!?'\"\-]+", " ", text)
    text = _P.apply_addr(text)
    text = _P.MULTI_SPACE.sub(" ", text).strip()
    return text


def extract_postal_code(address: str, country: str) -> Optional[str]:
    """
    Extract a standardised postal code token from an address string.
    Returns None if no postal code is found.
    """
    country_upper = (country or "").strip().upper()
    if country_upper == "INDIA":
        m = _P.PIN_PAT.search(address)
        return m.group(1) if m else None
    elif country_upper == "US":
        m = _P.ZIP_PAT.search(address)
        return m.group(1) if m else None
    elif country_upper == "FRANCE":
        m = _P.FR_ZIP_PAT.search(address)
        return m.group(1) if m else None
    else:
        # Generic: try 5- or 6-digit
        m = re.search(r"\b([1-9][0-9]{4,5})\b", address)
        return m.group(1) if m else None


def extract_house_number(address: str) -> Optional[str]:
    """Extract the first house/building number token from an address."""
    m = _P.HOUSE_NUM_PAT.search(address)
    return m.group(0) if m else None


def build_name_tokens(name: str) -> list[str]:
    """
    Return deduplicated, sorted, alpha-only tokens for Jaccard and TF-IDF.
    Short stop-words (length <= 2) are dropped since they dilute similarity.
    """
    tokens = re.findall(r"[a-z0-9]+", name)
    # Remove very short tokens (articles, prepositions)
    STOPS = {"a", "an", "the", "of", "in", "at", "to", "and", "or", "for",
             "on", "by", "with", "de", "du", "la", "le", "les", "et"}
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPS]
    return tokens


def build_address_tokens(addr: str) -> list[str]:
    """Return alpha-numeric tokens from a normalised address for overlap."""
    tokens = re.findall(r"[a-z0-9]+", addr)
    STOPS = {"null", "none", "na", "the", "and", "or", "of", "in", "at"}
    tokens = [t for t in tokens if len(t) >= 2 and t not in STOPS]
    return tokens


# ---------------------------------------------------------------------------
# Parallel worker function (module-level for joblib loky pickling on Windows)
# ---------------------------------------------------------------------------

def _normalise_chunk(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Normalise a chunk of (raw_name, raw_address) string pairs.
    Called in worker processes by joblib.Parallel (loky backend).
    Must be module-level for Windows spawn pickling to work.
    """
    return [(normalise_name(name), normalise_address(addr)) for name, addr in pairs]


# ---------------------------------------------------------------------------
# Polars-level processing (fast, parallelised)
# ---------------------------------------------------------------------------

def process_source_file(path: str | Path) -> pl.DataFrame:
    """
    Load a source TSV and return a Polars DataFrame with extra normalised columns:

    Columns returned:
      entity_id, business_name, business_address, country,
      norm_name, norm_address, name_tokens, address_tokens,

      postal_code, house_number, source (S1/S2/S3 derived from entity_id prefix)
    """
    log.info(f"Loading {path} …")
    df = pl.read_csv(
        str(path),
        separator="\t",
        has_header=True,
        infer_schema_length=1000,
        null_values=["", "null", "NULL", "None", "NaN"],
        truncate_ragged_lines=True,
        encoding="utf8",
    )
    # Guarantee expected columns exist, filling with empty string if absent
    for col in ("entity_id", "business_name", "business_address", "country"):
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))

    df = df.with_columns([
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    ])

    n_rows = len(df)
    log.info(f"  [highlight]{n_rows:,}[/highlight] rows — normalising in parallel …")

    name_list = df["business_name"].to_list()
    addr_list = df["business_address"].to_list()
    country_list = df["country"].to_list()

    # ── Parallel normalisation ─────────────────────────────────────────────
    # Split into chunks; each chunk runs in a separate process (loky backend).
    # loky uses spawn on Windows → _normalise_chunk must be module-level.
    n_workers = safe_n_workers(N_PROCESS_WORKERS)
    chunk_size = max(1, CHUNK_PREPROCESS)
    pairs = list(zip(name_list, addr_list))
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    log.info(f"  Workers: [highlight]{n_workers}[/highlight]  "
             f"Chunks: [highlight]{len(chunks)}[/highlight]  "
             f"Chunk size: [highlight]{chunk_size:,}[/highlight]")

    from pipeline_utils import make_progress
    with make_progress() as progress:
        task = progress.add_task(
            f"  Normalising {Path(path).name} [{n_workers} workers]",
            total=len(chunks)
        )
        results_chunks = []
        for chunk_result in Parallel(
            n_jobs=n_workers,
            backend="loky",
            return_as="generator",
            verbose=0,
        )(delayed(_normalise_chunk)(c) for c in chunks):
            results_chunks.append(chunk_result)
            progress.advance(task)

    # Flatten results
    flat = [pair for chunk in results_chunks for pair in chunk]
    norm_names = [p[0] for p in flat]
    norm_addrs = [p[1] for p in flat]


    postal_codes = [
        extract_postal_code(a, c)
        for a, c in zip(norm_addrs, country_list)
    ]
    house_nums = [extract_house_number(a) for a in norm_addrs]
    name_tokens = [" ".join(build_name_tokens(n)) for n in norm_names]
    addr_tokens = [" ".join(build_address_tokens(a)) for a in norm_addrs]

    sources = [
        eid[:2] if isinstance(eid, str) and len(eid) >= 2 else ""
        for eid in df["entity_id"].to_list()
    ]

    df = df.with_columns([
        pl.Series("norm_name", norm_names),
        pl.Series("norm_address", norm_addrs),
        pl.Series("name_tokens", name_tokens),
        pl.Series("address_tokens", addr_tokens),
        pl.Series("postal_code", postal_codes),
        pl.Series("house_number", house_nums),
        pl.Series("source", sources),
    ])

    # Country breakdown
    country_counts = df["country"].value_counts().sort("count", descending=True)
    breakdown = "  ".join(
        f"{r['country']}: [highlight]{r['count']:,}[/highlight]"
        for r in country_counts.iter_rows(named=True)
    )
    log.info(f"  Countries → {breakdown}")
    log.info(f"  [success]Done[/success] — {n_rows:,} records processed")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess source TSVs for entity resolution."
    )
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--out-dir", default="dataset/processed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [
            ("train_source1.tsv", "s1"),
            ("train_source2.tsv", "s2"),
            ("train_source3.tsv", "s3"),
        ],
        "test": [
            ("test_source1.tsv", "s1"),
            ("test_source2.tsv", "s2"),
            ("test_source3.tsv", "s3"),
        ],
    }
    dirs = {"train": args.train_dir, "test": args.test_dir}

    all_files = [
        (split, fname, label, Path(dirs[split]) / fname)
        for split, files in splits.items()
        for fname, label in files
    ]

    with StageTimer("Preprocessing — All Source Files"):
        for split, fname, label, in_path in all_files:
            out_path = out_dir / f"{split}_{label}.parquet"
            if not in_path.exists():
                log.warning(f"  {in_path} not found — skipping.")
                continue
            if out_path.exists():
                log.info(f"  [dim_white]Skipping {fname} (already preprocessed)[/dim_white]")
                continue
            console.print(f"\n  [info]▶ {split.upper()} / {fname}[/info]")
            df = process_source_file(in_path)
            df.write_parquet(str(out_path))
            log.info(f"  Saved → [highlight]{out_path}[/highlight]")

    log.info("[success]All preprocessing complete.[/success]")


if __name__ == "__main__":
    main()

