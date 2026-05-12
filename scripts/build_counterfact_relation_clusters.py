#!/usr/bin/env python3
"""
Build CounterFact Relation-Family Clusters
==========================================

Partitions ``data/counterfact_train.jsonl`` (19,728 records, 34 distinct
Wikidata P-ids) into 6 thematic clusters by a curated P-id → cluster mapping,
producing one JSONL per cluster (``data/counterfact_relfam_{0..5}.jsonl``)
and a single mapping snapshot (``data/counterfact_relfam_mapping.json``).

Why curated, not agglomerative
------------------------------
The legacy ``build_counterfact_data.py`` runs agglomerative clustering on
per-relation MiniLM centroids. Empirically that produced incoherent groupings:
Cluster 1 (6,692 records) merged P27 "citizenship", P106 "occupation",
P413 "sport position", P20 "place of death" — whatever was nearby in MiniLM
space — while Clusters 0 and 3 ended up with 538 / 710 records each. The
exposé criterion ("each Patch specialises in a coherent family of relations")
is normative; the clustering method is means-to-end. A curated mapping based
on relation semantics satisfies the criterion; agglomerative did not.

Cluster mapping
---------------
0  Physical geography             P30, P495, P17, P740                 (3,176)
1  Biographical (locale-of-person) P27, P937, P20, P19                  (3,060)
2  Linguistic                     P1412, P103, P37, P364, P407         (3,331)
3  Role / occupation / category   P413, P136, P106, P101, P1303, P39, P641  (4,016)
4  Production / corporate         P176, P449, P178, P127, P264, P108   (2,805)
5  Administrative / structural    P159, P131, P190, P276, P36, P138, P140, P463  (3,340)

All clusters fall within the [2500, 4500] target band.

Outputs
-------
- ``data/counterfact_relfam_{i}.jsonl`` — one record per line, same schema as
  ``data/counterfact_train.jsonl``.
- ``data/counterfact_relfam_mapping.json`` — { cluster_id → { name,
  description, p_ids, record_count }, plus a flat p_id_to_cluster lookup }.

Usage
-----
    python scripts/build_counterfact_relation_clusters.py
    python scripts/build_counterfact_relation_clusters.py \\
        --input data/counterfact_train.jsonl --output_dir data/

Author: Leon Wagner
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CLUSTER_DEFINITIONS = [
    {
        "cluster_id": 0,
        "name": "physical_geography",
        "description": (
            "Locale-of-thing: where physical things, places, or works of art "
            "are located, originated, or formed."
        ),
        "p_ids": ["P30", "P495", "P17", "P740"],
    },
    {
        "cluster_id": 1,
        "name": "biographical",
        "description": (
            "Locale-of-person: citizenship, work location, birthplace, "
            "deathplace — relations whose subject is a person."
        ),
        "p_ids": ["P27", "P937", "P20", "P19"],
    },
    {
        "cluster_id": 2,
        "name": "linguistic",
        "description": (
            "Languages spoken by people, native languages, official "
            "languages of countries, original languages of works."
        ),
        "p_ids": ["P1412", "P103", "P37", "P364", "P407"],
    },
    {
        "cluster_id": 3,
        "name": "role_occupation",
        "description": (
            "Roles, occupations, sport positions, instruments played, "
            "fields of work, professional categories."
        ),
        "p_ids": ["P413", "P136", "P106", "P101", "P1303", "P39", "P641"],
    },
    {
        "cluster_id": 4,
        "name": "production_corporate",
        "description": (
            "Manufacturer, broadcaster, developer, owner, record label, "
            "employer — production and corporate-relation predicates."
        ),
        "p_ids": ["P176", "P449", "P178", "P127", "P264", "P108"],
    },
    {
        "cluster_id": 5,
        "name": "administrative_structural",
        "description": (
            "Administrative location (HQ, capital, twin city, "
            "located-in-admin-entity), eponymy, religion, organisation "
            "membership — structural relations between entities."
        ),
        "p_ids": ["P159", "P131", "P190", "P276", "P36", "P138", "P140", "P463"],
    },
]

EXPECTED_TOTAL = 19728
EXPECTED_DISTINCT_P_IDS = 34
SIZE_BAND = (2500, 4500)


def build_p_id_to_cluster() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for c in CLUSTER_DEFINITIONS:
        for pid in c["p_ids"]:
            if pid in mapping:
                raise ValueError(f"P-id {pid} assigned to two clusters")
            mapping[pid] = c["cluster_id"]
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        default="data/counterfact_train.jsonl",
        help="Path to CounterFact training JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/",
        help="Directory to write counterfact_relfam_{0..5}.jsonl into.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 1

    p_id_to_cluster = build_p_id_to_cluster()

    cluster_records: dict[int, list[dict]] = {c["cluster_id"]: [] for c in CLUSTER_DEFINITIONS}
    p_id_counts: Counter[str] = Counter()
    unknown_p_ids: Counter[str] = Counter()

    with input_path.open() as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("relation_id")
            if pid is None:
                print(f"WARN: record without relation_id: id={rec.get('id')}", file=sys.stderr)
                continue
            p_id_counts[pid] += 1
            cluster_id = p_id_to_cluster.get(pid)
            if cluster_id is None:
                unknown_p_ids[pid] += 1
                continue
            cluster_records[cluster_id].append(rec)

    total_loaded = sum(p_id_counts.values())
    if total_loaded != EXPECTED_TOTAL:
        print(
            f"WARN: loaded {total_loaded} records, expected {EXPECTED_TOTAL}",
            file=sys.stderr,
        )
    distinct_p_ids = len(p_id_counts)
    if distinct_p_ids != EXPECTED_DISTINCT_P_IDS:
        print(
            f"WARN: found {distinct_p_ids} distinct P-ids, expected {EXPECTED_DISTINCT_P_IDS}",
            file=sys.stderr,
        )

    if unknown_p_ids:
        print(f"ERROR: {len(unknown_p_ids)} P-ids not in mapping:", file=sys.stderr)
        for pid, n in unknown_p_ids.most_common():
            print(f"  {pid}: {n} records", file=sys.stderr)
        return 1

    for cluster in CLUSTER_DEFINITIONS:
        cid = cluster["cluster_id"]
        out_path = output_dir / f"counterfact_relfam_{cid}.jsonl"
        records = cluster_records[cid]
        with out_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        cluster["record_count"] = len(records)

    mapping_path = output_dir / "counterfact_relfam_mapping.json"
    mapping_doc = {
        "source": str(input_path),
        "total_records": total_loaded,
        "distinct_p_ids": distinct_p_ids,
        "size_band_target": list(SIZE_BAND),
        "clusters": [
            {
                "cluster_id": c["cluster_id"],
                "name": c["name"],
                "description": c["description"],
                "p_ids": c["p_ids"],
                "record_count": c["record_count"],
                "p_id_counts": {pid: p_id_counts[pid] for pid in c["p_ids"]},
            }
            for c in CLUSTER_DEFINITIONS
        ],
        "p_id_to_cluster": p_id_to_cluster,
    }
    with mapping_path.open("w") as f:
        json.dump(mapping_doc, f, indent=2)

    print()
    print(f"{'cluster':>10}  {'name':<26} {'records':>8}  {'in_band':>8}  p_ids")
    print("-" * 100)
    out_of_band: list[int] = []
    for c in CLUSTER_DEFINITIONS:
        cid = c["cluster_id"]
        n = c["record_count"]
        ok = SIZE_BAND[0] <= n <= SIZE_BAND[1]
        if not ok:
            out_of_band.append(cid)
        ok_str = "yes" if ok else "NO"
        pids_str = ", ".join(c["p_ids"])
        print(f"{cid:>10}  {c['name']:<26} {n:>8}  {ok_str:>8}  {pids_str}")
    print("-" * 100)
    total_clustered = sum(c["record_count"] for c in CLUSTER_DEFINITIONS)
    print(f"{'total':>10}  {'':<26} {total_clustered:>8}")
    print()

    if out_of_band:
        print(f"ERROR: clusters out of band {SIZE_BAND}: {out_of_band}", file=sys.stderr)
        return 1

    print(f"Wrote {len(CLUSTER_DEFINITIONS)} cluster JSONLs and {mapping_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
