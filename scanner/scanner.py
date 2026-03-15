#!/usr/bin/env python3
"""Scanner station: Use BioBERT/ClinicalBERT for medical NER.

This script demonstrates how to load a pre-trained biomedical NER model and extract
entities such as diseases, drugs, and dosages from messy, Singapore-style clinical notes.

Run:
    python scanner/scanner.py --text "Pt w/ DM2 on metformin 500mg BD..."

"""

import argparse
import re
from typing import List, Dict

from transformers import pipeline, Pipeline


DEFAULT_MODEL = "d4data/biobert_ner_diseases"
FALLBACK_MODEL = "dslim/bert-base-NER"


def load_ner_pipeline(model_name: str) -> Pipeline:
    """Load a token-classification pipeline. Falls back to a general NER model if needed."""
    try:
        return pipeline("ner", model=model_name, grouped_entities=True)
    except Exception as e:
        print(f"⚠️  Could not load model '{model_name}' (maybe offline). Falling back to {FALLBACK_MODEL}.")
        return pipeline("ner", model=FALLBACK_MODEL, grouped_entities=True)


def extract_dosages(text: str) -> List[str]:
    """A lightweight regex extractor for common dosage patterns."""
    pattern = r"\b\d+\s*(?:mg|g|mcg|ml|tab|tabs|tablet|capsule)s?\b"
    return re.findall(pattern, text, flags=re.IGNORECASE)


def normalize_label(label: str) -> str:
    """Normalize common NER labels to a standard set for this demo."""
    label = label.upper()
    if "DRUG" in label or "CHEM" in label:
        return "DRUG"
    if "DISEASE" in label or "CONDITION" in label or "MORB" in label:
        return "DISEASE"
    if "SYMPTOM" in label:
        return "SYMTOM"
    return label


def extract_entities(text: str, ner_pipe: Pipeline) -> List[Dict]:
    """Extract entities using NER and a few lightweight heuristics."""
    entities = ner_pipe(text)
    parsed = []

    for ent in entities:
        parsed.append(
            {
                "text": ent.get("word") or ent.get("entity_group"),
                "label": normalize_label(ent.get("entity_group", ent.get("entity"))),
                "score": ent.get("score", 0.0),
                "start": ent.get("start"),
                "end": ent.get("end"),
            }
        )

    # Add dosage heuristics on top of what the model finds
    for dose in extract_dosages(text):
        parsed.append({"text": dose, "label": "DOSAGE", "score": 1.0, "start": text.lower().find(dose.lower()), "end": text.lower().find(dose.lower()) + len(dose)})

    return parsed


def main():
    parser = argparse.ArgumentParser(description="Scanner station: BIOBERT-style NER for messy clinical notes.")
    parser.add_argument("--text", type=str, required=False, default=None, help="Text to run NER on.")
    parser.add_argument("--file", type=str, required=False, help="Path to a text file containing notes.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        raise SystemExit("ERROR: Provide --text or --file")

    ner_pipe = load_ner_pipeline(DEFAULT_MODEL)
    entities = extract_entities(text, ner_pipe)

    print("\n=== Extracted entities (Scanner) ===")
    for ent in entities:
        print(f"- {ent['label']:>8} | {ent['text']} (score={ent['score']:.2f})")


if __name__ == "__main__":
    main()
