#!/usr/bin/env python3
"""Entity Linking station (Logic)

Uses vector similarity to normalize extracted clinical entities to canonical names.
Includes Singapore-specific shorthand handling.

Example:
    python entity_linking.py --term "T2DM"
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)

# A small Singapore-centric shorthand dictionary
SG_SHORTHAND: Dict[str, str] = {
    "SGH": "Singapore General Hospital",
    "T-SCORE": "T-score",
    "CHAS": "CHAS card",
    "KENCING MANIS": "Type 2 diabetes mellitus",
    "DM2": "Type 2 diabetes mellitus",
    "T2DM": "Type 2 diabetes mellitus",
    "HTN": "Hypertension",
    "NSTEMI": "Acute myocardial infarction",
    "SOB": "Shortness of breath",
}


@dataclass
class Concept:
    name: str
    description: str


CANONICAL_CONCEPTS: List[Concept] = [
    Concept(name="Type 2 diabetes mellitus", description="Chronic metabolic disorder."),
    Concept(name="Hypertension", description="Elevated blood pressure."),
    Concept(name="Acute myocardial infarction", description="Heart attack."),
    Concept(name="Gastritis", description="Stomach lining inflammation."),
    Concept(name="Asthma", description="Airway inflammation and bronchospasm."),
]


def init_logging(level: int = logging.INFO) -> None:
    """Initialize module logging."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
    )


def _normalize_input(term: str) -> str:
    """Normalize the incoming term before linking.

    - Strip whitespace
    - Uppercase for lookup against common shorthand
    """
    if not term:
        return ""
    return term.strip()


def link_term(
    term: str,
    model: SentenceTransformer,
    kb_texts: List[str],
    index,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Link a free-text term to canonical terms using vector similarity."""
    if not term or not term.strip():
        LOGGER.warning("Empty term given to link_term")
        return []

    norm = term.strip().upper()

    # First look for Singapore shorthand shortcuts.
    if norm in SG_SHORTHAND:
        canonical = SG_SHORTHAND[norm]
        LOGGER.debug("Matched Singapore shorthand %s -> %s", term, canonical)
        return [(canonical, 1.0)]

    q_emb = model.encode([term], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q_emb, top_k)

    return [(kb_texts[i], float(scores[0][j])) for j, i in enumerate(idx[0])]


def build_index(model: SentenceTransformer, texts: List[str]):
    """Build a FAISS index for the knowledge base texts."""
    import faiss

    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    init_logging()

    parser = argparse.ArgumentParser(description="Link extracted entities to canonical medical terms.")
    parser.add_argument("--term", type=str, required=True, help="Term to link (e.g. 'T2DM').")
    parser.add_argument("--top_k", type=int, default=3, help="How many candidates to return.")
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    kb_texts = [c.name for c in CANONICAL_CONCEPTS]
    index = build_index(model, kb_texts)

    matches = link_term(args.term, model, kb_texts, index, top_k=args.top_k)

    print("\n=== Candidate Normalizations ===")
    for term, score in matches:
        print(f"- {term} (score={score:.3f})")


if __name__ == "__main__":
    main()
