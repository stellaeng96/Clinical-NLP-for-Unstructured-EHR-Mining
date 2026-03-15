#!/usr/bin/env python3
"""Translator station: look up OHDSI/OMOP concept IDs for canonical terms.

This demo uses a small local CSV vocabulary to simulate Athena lookups.
"""

import argparse
import csv
import os
from typing import Dict, Optional


VOCAB_CSV = os.path.join(os.path.dirname(__file__), "vocab.csv")


def load_vocab(path: str) -> Dict[str, Dict]:
    vocab = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["concept_name"].strip().lower()
            vocab[name] = row
    return vocab


def lookup_concept(term: str, vocab: Dict[str, Dict]) -> Optional[Dict]:
    return vocab.get(term.strip().lower())


def main():
    parser = argparse.ArgumentParser(description="Translator station: map canonical names to OMOP concept IDs.")
    parser.add_argument("--term", type=str, required=True, help="Canonical term to look up (e.g. 'Type 2 diabetes mellitus')")
    args = parser.parse_args()

    vocab = load_vocab(VOCAB_CSV)
    record = lookup_concept(args.term, vocab)

    if not record:
        print(f"No matching concept found for '{args.term}'.")
        return

    print("✅ Found concept")
    print(f"Concept ID: {record['concept_id']}")
    print(f"Name:       {record['concept_name']}")
    print(f"Domain:     {record['domain_id']}")
    print(f"Class:      {record['concept_class_id']}")
    print(f"Standard:   {record['standard_concept']}")


if __name__ == "__main__":
    main()
