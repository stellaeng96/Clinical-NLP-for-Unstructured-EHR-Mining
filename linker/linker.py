#!/usr/bin/env python3
"""Logic station: match messy/abbreviated entities to canonical medical concepts.

This script builds a small vector database using sentence-transformers + FAISS and
demonstrates how to resolve synonyms like "T2DM" -> "Type 2 diabetes mellitus".

Run:
    python linker/linker.py --query "T2DM"
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class CanonicalConcept:
    name: str
    description: str
    canonical: str


CONCEPTS: List[CanonicalConcept] = [
    CanonicalConcept(
        name="Type 2 diabetes mellitus",
        description="Chronic metabolic disorder characterized by insulin resistance.",
        canonical="Type 2 diabetes mellitus",
    ),
    CanonicalConcept(
        name="Hypertension",
        description="Elevated blood pressure.",
        canonical="Hypertension",
    ),
    CanonicalConcept(
        name="Gastritis",
        description="Inflammation of the stomach lining.",
        canonical="Gastritis",
    ),
    CanonicalConcept(
        name="Acute myocardial infarction",
        description="Heart attack.",
        canonical="Acute myocardial infarction",
    ),
]

SYNONYMS = {
    "T2DM": "Type 2 diabetes mellitus",
    "DM2": "Type 2 diabetes mellitus",
    "kencing manis": "Type 2 diabetes mellitus",
    "h/p of dm": "Type 2 diabetes mellitus",
    "HTN": "Hypertension",
    "SOB": "Shortness of breath",
    "CP": "Chest pain",
    "NSTEMI": "Acute myocardial infarction",
}


def build_index(model: SentenceTransformer, texts: List[str]):
    """Build a FAISS index over embeddings."""
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def resolve(query: str, model: SentenceTransformer, index: faiss.IndexFlatIP, texts: List[str], top_k: int = 3):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q_emb, top_k)

    results = []
    for score, i in zip(scores[0], idx[0]):
        results.append((texts[i], float(score)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Linker station: map abbreviations to canonical terms via vector search.")
    parser.add_argument("--query", type=str, required=True, help="Free-text entity to link (e.g. 'T2DM')")
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build knowledge base
    kb_texts = [c.name for c in CONCEPTS]
    index, _ = build_index(model, kb_texts)

    # Shortcut: handle common known synonyms first
    if args.query in SYNONYMS:
        print(f"✅ Known synonym: {args.query} -> {SYNONYMS[args.query]}")

    print("\n🔎 Vector search results")
    matches = resolve(args.query, model, index, kb_texts, top_k=3)

    for term, score in matches:
        print(f"- {term} (score={score:.3f})")

    # Show canonical results
    print("\n📌 Canonical mapping (best match)")
    if matches:
        best_match = matches[0][0]
        print(f"{args.query} → {best_match}")


if __name__ == "__main__":
    main()
