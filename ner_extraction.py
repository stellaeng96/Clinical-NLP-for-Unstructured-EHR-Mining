#!/usr/bin/env python3
"""NER Extraction station (Scanner)

Loads a BioBERT model (dmis-lab/biobert-v1.1-pubmed) and extracts
medical entities (Disease, Drug) from clinical text.

Features:
- Robust handling of messy / mixed-case input
- Negation detection (e.g., "denies history of Diabetes")
- Clear structured output for downstream linking

Example:
    python ner_extraction.py --text "Patient denies history of Diabetes but has HTN."
"""

import argparse
import logging
import re
from typing import Any, Dict, List, Optional

from transformers import pipeline, Pipeline


LOGGER = logging.getLogger(__name__)

# Model to load from Hugging Face
DEFAULT_MODEL = "dmis-lab/biobert-v1.1-pubmed"

# Simple negation triggers for clinical notes
_NEGATION_TRIGGERS = [
    r"\bdenies?\b",
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bfree of\b",
    r"\bdoes not have\b",
    r"\bnot\b",
]

_NEGATION_WINDOW = 30  # number of chars before entity to search for negation


def init_logging(level: int = logging.INFO) -> None:
    """Initialize module-level logging."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
    )


def load_ner_pipeline(model_name: str = DEFAULT_MODEL) -> Pipeline:
    """Load a transformer pipeline for token classification.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        A huggingface Pipeline object for NER.
    """
    try:
        LOGGER.debug("Loading NER model %s", model_name)
        return pipeline("ner", model=model_name, grouped_entities=True)
    except Exception as e:
        LOGGER.exception("Failed to load NER model %s", model_name)
        raise


def _is_negated(text: str, entity_span: Dict[str, Any]) -> bool:
    """Basic negation detection based on proximity to negation cues.

    This is a lightweight approximation of medSpaCy-style negation.
    """
    start = max(0, entity_span["start"] - _NEGATION_WINDOW)
    context = text[start : entity_span["start"]].lower()

    for trigger in _NEGATION_TRIGGERS:
        if re.search(trigger, context):
            return True
    return False


def normalize_label(label: str) -> str:
    """Normalize HuggingFace labels to a controlled set."""
    label = label.upper()
    if "DISEASE" in label or "CONDITION" in label or "MORB" in label:
        return "DISEASE"
    if "DRUG" in label or "CHEM" in label:
        return "DRUG"
    return label


def extract_entities(text: str, ner_pipe: Pipeline) -> List[Dict[str, Any]]:
    """Extract entities (disease/drug) and annotate negation.

    Args:
        text: Raw clinical text.
        ner_pipe: Loaded HuggingFace NER pipeline.

    Returns:
        List of entities with `text`, `label`, `start`, `end`, `score`, `negated`.
    """
    if not text or not text.strip():
        LOGGER.warning("Empty input text received for entity extraction")
        return []

    text = text.strip()
    entities = ner_pipe(text)
    results: List[Dict[str, Any]] = []

    for ent in entities:
        label = normalize_label(ent.get("entity_group") or ent.get("entity", ""))
        if label not in {"DISEASE", "DRUG"}:
            continue

        span = {
            "text": ent.get("word"),
            "label": label,
            "score": float(ent.get("score", 0.0)),
            "start": int(ent.get("start", -1)),
            "end": int(ent.get("end", -1)),
        }
        span["negated"] = _is_negated(text, span)
        results.append(span)

    return results


def main():
    init_logging()

    parser = argparse.ArgumentParser(description="Extract diseases and drugs from clinical text.")
    parser.add_argument("--text", type=str, help="Text snippet to run NER on.")
    parser.add_argument("--file", type=str, help="Path to a text file containing notes.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model id for NER (default: dmis-lab/biobert-v1.1-pubmed)",
    )
    args = parser.parse_args()

    if not args.text and not args.file:
        raise SystemExit("ERROR: Provide --text or --file")

    text = ""
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    ner_pipe = load_ner_pipeline(args.model)
    entities = extract_entities(text, ner_pipe)

    print("\n=== Extracted Entities ===")
    for ent in entities:
        neg = "(NEGATED)" if ent.get("negated") else ""
        print(
            f"- {ent['label']:>7} | {ent['text']} {neg} (score={ent['score']:.2f})"
        )


if __name__ == "__main__":
    main()
