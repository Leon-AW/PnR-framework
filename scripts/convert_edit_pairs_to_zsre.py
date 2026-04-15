"""Convert SituatedQA edit_pairs.json to ZSRE format for the official RECIPE repo.

Maps:
    question     -> src
    answer       -> alt
    question_gen -> rephrase
    question_loc -> loc
    answer_loc   -> loc_ans

Output default: external/RECIPE/data/meta-train/zsre/zsre_mend_train.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(REPO_ROOT / "data" / "edit_pairs.json"),
    )
    ap.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "external"
            / "RECIPE"
            / "data"
            / "meta-train"
            / "zsre"
            / "zsre_mend_train.json"
        ),
    )
    args = ap.parse_args()

    with open(args.input) as f:
        pairs = json.load(f)

    out = []
    for p in pairs:
        out.append(
            {
                "src": p["question"],
                "alt": p["answer"],
                "rephrase": p.get("question_gen", p["question"]),
                "loc": p["question_loc"],
                "loc_ans": p["answer_loc"],
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} ZSRE-formatted edits to {args.output}")


if __name__ == "__main__":
    main()
