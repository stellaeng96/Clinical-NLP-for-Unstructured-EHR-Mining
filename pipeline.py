#!/usr/bin/env python3
"""End-to-end demo pipeline: Scanner → Logic → Translator → Storage.

This script demonstrates how notes travel from raw text into an OMOP-ready row.

Example:
    python pipeline.py --note-file data/synthetic_notes.txt --person 123
"""

import argparse
import os
from datetime import date
from typing import List, Optional

from scanner.scanner import extract_entities, load_ner_pipeline
from linker.linker import resolve, build_index, CONCEPTS
from omop_mapping import search_concepts
from storage.store import init_db, insert_condition


def filter_entities(entities: List[dict]) -> List[str]:
    """Keep only entities that we treat as a condition/diagnosis for this demo."""
    keep_labels = {"DISEASE", "DRUG", "DOSAGE"}
    return [e["text"] for e in entities if e["label"] in keep_labels]


def pick_best_concept(query: str, index, kb_texts, model) -> str:
    results = resolve(query, model, index, kb_texts, top_k=1)
    return results[0][0] if results else query


def main():
    parser = argparse.ArgumentParser(description="Demo pipeline: from raw note to OMOP condition_occurrence")
    parser.add_argument("--note-file", type=str, required=True)
    parser.add_argument("--person", type=int, required=True)
    parser.add_argument("--db", type=str, default="pipeline_omop.db")
    args = parser.parse_args()

    with open(args.note_file, "r", encoding="utf-8") as f:
        note = f.read().strip()

    print("\n--- Scanner: Extract entities")
    ner_pipe = load_ner_pipeline("d4data/biobert_ner_diseases")
    entities = extract_entities(note, ner_pipe)
    found = filter_entities(entities)

    for e in found:
        print(f"- {e}")

    print("\n--- Logic: Resolve synonyms to canonical terms")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    kb_texts = [c.name for c in CONCEPTS]
    index, _ = build_index(model, kb_texts)

    canonical = []
    for term in found:
        best = pick_best_concept(term, index, kb_texts, model)
        print(f"{term} → {best}")
        canonical.append(best)

    print("\n--- Translator: Map canonical term to OMOP concept ID (Athena)")
    mapped = []
    for term in canonical:
        candidates = search_concepts(term, top_k=5)
        if not candidates:
            print(f"{term} → (no mapping found)")
            continue

        # Print top candidate scores
        print(f"{term} → top {len(candidates)} candidates:")
        for cand in candidates:
            score_str = f" (score={cand.get('score'):.2f})" if cand.get('score') is not None else ""
            print(f"  - {cand['concept_name']} (concept_id={cand['concept_id']}){score_str}")

        best = candidates[0]
        mapped.append(best)

    if not mapped:
        print("No OMOP concepts found; exiting.")
        return

    if not mapped:
        print("No OMOP concepts found; exiting.")
        return

    print("\n--- Storage: insert into OMOP condition_occurrence (SQLite)")
    db_path = args.db
    schema_path = os.path.join(os.path.dirname(__file__), "storage", "schema.sql")
    conn = init_db(db_path, schema_path)

    today = date.today().isoformat()
    for record in mapped:
        row_id = insert_condition(
            conn,
            person_id=args.person,
            condition_concept_id=int(record["concept_id"]),
            condition_start_date=today,
            condition_source_value=record["concept_name"],
        )
        print(f"Inserted row_id={row_id} for concept_id={record['concept_id']}")

    print(f"\n✅ Pipeline complete. See {db_path} for results.")


if __name__ == "__main__":
    main()
