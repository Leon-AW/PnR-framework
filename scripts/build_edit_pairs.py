#!/usr/bin/env python3
"""
Build Edit-Pairs JSON from SituatedQA
======================================

Converts SituatedQA temporal and geographic splits into the edit-pairs format
required by RLEdit and RECIPE baselines.

Edit-pairs format (RLEdit / RECIPE compatible):
    {
        "question":      str,   # Edit query q_e   (edited_question WITH context trigger)
        "answer":        str,   # Target answer y_e
        "question_gen":  str,   # Generality query q_g (bare question WITHOUT trigger)
        "answer_gen":    str,   # Generality target (same as answer)
        "question_loc":  str,   # Locality probe (unrelated, stable fact)
        "answer_loc":    str    # Expected locality answer (should not change after edit)
    }

Strategy
--------
- **Temporal edits** (date >= cutoff_year, default 2019):
    • question/answer  = edited_question + answer from temporal patch stream
    • question_loc/answer_loc = stable US geo fact from geo base stream

- **Geo edits** (non-US locations):
    • question/answer  = edited_question + answer from geo patch stream
    • question_loc/answer_loc = stable pre-cutoff temporal fact from temporal base stream

Locality pairing is cross-domain (temporal edit ↔ geo locality) to ensure the
probe is genuinely unrelated to the edit. Pairs are shuffled before output.

Usage
-----
    python scripts/build_edit_pairs.py --output_path data/edit_pairs.json
    python scripts/build_edit_pairs.py --output_path data/edit_pairs.json \\
        --cutoff_year 2019 --max_edits 2000 --seed 42

Output
------
    data/edit_pairs.json  — list of dicts with keys: question, answer, question_loc, answer_loc
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Raw data URLs (same as src/data/loader.py)
# ---------------------------------------------------------------------------
TEMP_TRAIN_URL = "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data/temp.train.jsonl"
GEO_TRAIN_URL  = "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data/geo.train.jsonl"

# US location variants (lowercase) — mirrors src/data/loader.py
US_LOCATIONS: frozenset[str] = frozenset({
    "united states", "us", "usa", "america",
    "u.s.", "u.s.a.", "united states of america",
    "the united states", "the us", "the usa",
})


def is_us_location(loc: str | None) -> bool:
    if not loc or loc.strip() == "":
        return True  # implicit US
    return loc.strip().lower() in US_LOCATIONS


def extract_year(date_str: str | None) -> int | None:
    import re
    if not date_str:
        return None
    m = re.search(r"\b(\d{4})\b", str(date_str))
    return int(m.group(1)) if m else None


def load_jsonl(url_or_path: str) -> list[dict[str, Any]]:
    """Load JSONL from a URL or local file path."""
    if url_or_path.startswith("http"):
        import urllib.request
        print(f"  Downloading {url_or_path} ...")
        with urllib.request.urlopen(url_or_path, timeout=60) as r:
            lines = r.read().decode("utf-8").splitlines()
    else:
        with open(url_or_path) as f:
            lines = f.readlines()
    return [json.loads(l) for l in lines if l.strip()]


def to_edit_entry(example: dict, answer_key: str = "answer") -> dict[str, str] | None:
    """Extract (question, answer, question_gen, answer_gen) from a SituatedQA example.

    Uses `edited_question` (with context trigger) as the edit query and
    the bare `question` (without trigger) as the generality query q_g —
    testing that the model still answers the uncontextualized question correctly.
    This directly maps to RECIPE's L_gen loss (Eq. 11).
    """
    q_edited = example.get("edited_question") or example.get("question", "")
    q_bare   = example.get("question") or q_edited
    if not q_edited or not q_edited.strip():
        return None
    answers = example.get(answer_key, [])
    if isinstance(answers, str):
        answers = [answers]
    if not answers:
        return None
    a = answers[0].strip()
    if not a:
        return None
    return {
        "question":     q_edited.strip(),
        "answer":       a,
        "question_gen": q_bare.strip(),  # bare question without "as of YEAR" / "in LOCATION"
        "answer_gen":   a,               # same target answer
    }


def build_edit_pairs(
    cutoff_year: int,
    max_edits: int,
    seed: int,
    temp_url: str = TEMP_TRAIN_URL,
    geo_url: str = GEO_TRAIN_URL,
) -> list[dict[str, str]]:
    """Build the edit-pairs list from SituatedQA.

    Returns
    -------
    List of dicts: {question, answer, question_gen, answer_gen, question_loc, answer_loc}
    """
    random.seed(seed)

    print("Loading SituatedQA temporal split...")
    temp_data = load_jsonl(temp_url)
    print(f"  {len(temp_data)} temporal examples loaded")

    print("Loading SituatedQA geo split...")
    geo_data = load_jsonl(geo_url)
    print(f"  {len(geo_data)} geo examples loaded")

    # ------------------------------------------------------------------
    # Partition into 4 buckets
    # ------------------------------------------------------------------
    # 1. Temporal edits  (date >= cutoff_year) → edit queries
    # 2. Temporal stable (date <  cutoff_year) → locality pool for geo edits
    # 3. Geo edits       (non-US)              → edit queries
    # 4. Geo stable      (US / implicit US)    → locality pool for temporal edits
    temporal_edits:  list[dict[str, str]] = []
    temporal_stable: list[dict[str, str]] = []
    geo_edits:  list[dict[str, str]] = []
    geo_stable: list[dict[str, str]] = []

    for ex in temp_data:
        year = extract_year(ex.get("date"))
        if year is None:
            continue
        entry = to_edit_entry(ex)
        if entry is None:
            continue
        if year >= cutoff_year:
            temporal_edits.append(entry)
        else:
            temporal_stable.append(entry)

    for ex in geo_data:
        loc = ex.get("location", "")
        entry = to_edit_entry(ex)
        if entry is None:
            continue
        if is_us_location(loc):
            geo_stable.append(entry)
        else:
            geo_edits.append(entry)

    print(f"\nBuckets:")
    print(f"  Temporal edits  (date >= {cutoff_year}): {len(temporal_edits)}")
    print(f"  Temporal stable (date <  {cutoff_year}): {len(temporal_stable)}")
    print(f"  Geo edits       (non-US):                {len(geo_edits)}")
    print(f"  Geo stable      (US/implicit):           {len(geo_stable)}")

    if not temporal_stable:
        raise ValueError("No temporal stable examples found — can't build geo edit locality probes")
    if not geo_stable:
        raise ValueError("No geo stable examples found — can't build temporal edit locality probes")

    # ------------------------------------------------------------------
    # Pair edits with cross-domain locality probes
    # ------------------------------------------------------------------
    # Shuffle all pools
    random.shuffle(temporal_edits)
    random.shuffle(geo_edits)
    random.shuffle(temporal_stable)
    random.shuffle(geo_stable)

    all_pairs: list[dict[str, str]] = []

    # Temporal edits  → geo stable locality
    for i, edit in enumerate(temporal_edits):
        loc = geo_stable[i % len(geo_stable)]
        all_pairs.append({
            "question":     edit["question"],
            "answer":       edit["answer"],
            "question_gen": edit["question_gen"],
            "answer_gen":   edit["answer_gen"],
            "question_loc": loc["question"],
            "answer_loc":   loc["answer"],
        })

    # Geo edits → temporal stable locality
    for i, edit in enumerate(geo_edits):
        loc = temporal_stable[i % len(temporal_stable)]
        all_pairs.append({
            "question":     edit["question"],
            "answer":       edit["answer"],
            "question_gen": edit["question_gen"],
            "answer_gen":   edit["answer_gen"],
            "question_loc": loc["question"],
            "answer_loc":   loc["answer"],
        })

    # Shuffle combined pairs and cap
    random.shuffle(all_pairs)
    if max_edits > 0:
        all_pairs = all_pairs[:max_edits]

    return all_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build edit-pairs JSON from SituatedQA for RLEdit/RECIPE baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/edit_pairs.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--cutoff_year",
        type=int,
        default=2019,
        help="Year boundary: temporal edits are date >= cutoff_year",
    )
    parser.add_argument(
        "--max_edits",
        type=int,
        default=0,
        help="Cap total edit pairs (0 = no limit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pairing and shuffling",
    )
    parser.add_argument(
        "--temp_url",
        type=str,
        default=TEMP_TRAIN_URL,
        help="URL or local path for temporal JSONL",
    )
    parser.add_argument(
        "--geo_url",
        type=str,
        default=GEO_TRAIN_URL,
        help="URL or local path for geo JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Build SituatedQA → edit-pairs JSON")
    print(f"  cutoff_year : {args.cutoff_year}")
    print(f"  max_edits   : {args.max_edits if args.max_edits > 0 else 'unlimited'}")
    print(f"  seed        : {args.seed}")
    print(f"  output      : {args.output_path}")
    print("=" * 60)

    pairs = build_edit_pairs(
        cutoff_year=args.cutoff_year,
        max_edits=args.max_edits,
        seed=args.seed,
        temp_url=args.temp_url,
        geo_url=args.geo_url,
    )

    print(f"\nGenerated {len(pairs)} edit pairs")

    # Write output
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved → {out}")

    # Print a few examples
    print("\n--- Sample pairs ---")
    for p in pairs[:3]:
        print(json.dumps(p, indent=2))


if __name__ == "__main__":
    main()
