#!/usr/bin/env python3
"""OMOP Mapping station (Filer)

Simulates Athena vocabulary lookup and formats output for OMOP condition_occurrence.

Example:
    python omop_mapping.py --term "Type 2 diabetes mellitus" --person_id 123
"""

import argparse
import json
import logging
import os
from datetime import date
from typing import Dict, List, Optional

import requests
from requests.exceptions import RequestException

LOGGER = logging.getLogger(__name__)

# OHDSI Athena REST API (can be overridden by setting ATHENA_BASE_URL)
ATHENA_BASE_URL = os.getenv("ATHENA_BASE_URL", "https://athena.ohdsi.org/api/v1")
ATHENA_TIMEOUT_SECONDS = 10

# Fallback local vocabulary when Athena is unreachable.
LOCAL_ATHENA_VOCAB: Dict[str, Dict[str, str]] = {
    "type 2 diabetes mellitus": {
        "concept_id": "201820",
        "concept_name": "Type 2 diabetes mellitus",
        "domain_id": "Condition",
        "concept_class_id": "Clinical Finding",
        "standard_concept": "S",
    },
    "hypertension": {
        "concept_id": "320128",
        "concept_name": "Hypertension",
        "domain_id": "Condition",
        "concept_class_id": "Clinical Finding",
        "standard_concept": "S",
    },
    "acute myocardial infarction": {
        "concept_id": "315166",
        "concept_name": "Acute myocardial infarction",
        "domain_id": "Condition",
        "concept_class_id": "Clinical Finding",
        "standard_concept": "S",
    },
    "gastritis": {
        "concept_id": "4147163",
        "concept_name": "Gastritis",
        "domain_id": "Condition",
        "concept_class_id": "Clinical Finding",
        "standard_concept": "S",
    },
}


def init_logging(level: int = logging.INFO) -> None:
    """Initialize module logging."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
    )


def _parse_athena_response(data: Dict) -> List[Dict[str, str]]:
    """Normalize various Athena search response shapes to a list of concept records."""
    candidates = (
        data.get("records")
        or data.get("concepts")
        or data.get("results")
        or data.get("items")
        or []
    )
    if not candidates:
        return []

    results: List[Dict[str, str]] = []
    for concept in candidates:
        results.append(
            {
                "concept_id": str(
                    concept.get("conceptId")
                    or concept.get("concept_id")
                    or concept.get("id")
                    or ""
                ),
                "concept_name": concept.get(
                    "conceptName"
                )
                or concept.get("concept_name")
                or concept.get("name")
                or "",
                "domain_id": concept.get("domainId")
                or concept.get("domain")
                or concept.get("domain_id")
                or "",
                "concept_class_id": concept.get("conceptClassId")
                or concept.get("concept_class_id")
                or concept.get("concept_class")
                or "",
                "standard_concept": concept.get("standardConcept")
                or concept.get("standard_concept")
                or "",
                "score": float(
                    concept.get("score")
                    or concept.get("relevance")
                    or concept.get("rank")
                    or 0.0
                ),
            }
        )
    return results


def _athena_search(term: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Query the OHDSI Athena REST API for matching concepts."""
    try:
        url = f"{ATHENA_BASE_URL}/concepts/search"
        params = {"query": term, "limit": top_k}
        LOGGER.debug("Querying ATHENA %s %s", url, params)
        resp = requests.get(url, params=params, timeout=ATHENA_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        return _parse_athena_response(data)
    except RequestException as e:
        LOGGER.warning("Athena lookup failed: %s", e)
        return []
    except ValueError as e:
        LOGGER.warning("Invalid JSON from Athena: %s", e)
        return []


def search_concepts(term: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Search for OMOP concepts by term and return ranked candidates.

    Uses OHDSI Athena when available, and falls back to a small local vocabulary.
    """
    if not term or not term.strip():
        LOGGER.warning("Empty term passed to OMOP lookup")
        return []

    athena_results = _athena_search(term.strip(), top_k=top_k)
    if athena_results:
        return athena_results

    LOGGER.info("Falling back to local vocabulary for term: %s", term)
    local = LOCAL_ATHENA_VOCAB.get(term.strip().lower())
    if not local:
        return []
    local["score"] = 1.0
    return [local]


def lookup_concept(term: str) -> Optional[Dict[str, str]]:
    """Lookup a single best OMOP concept (first candidate) for a term."""
    results = search_concepts(term, top_k=1)
    return results[0] if results else None


def build_condition_occurrence(
    person_id: int,
    concept_id: int,
    start_date: str,
    source_value: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Build a JSON-like dict for an OMOP condition_occurrence row."""
    return {
        "person_id": person_id,
        "condition_concept_id": concept_id,
        "condition_start_date": start_date,
        "condition_end_date": end_date,
        "condition_type_concept_id": 38000175,
        "condition_source_value": source_value,
    }


def format_sql_insert(row: Dict[str, Optional[str]], table_name: str = "condition_occurrence") -> str:
    """Format a SQL INSERT statement for a condition_occurrence row."""
    columns = [
        "person_id",
        "condition_concept_id",
        "condition_start_date",
        "condition_end_date",
        "condition_type_concept_id",
        "condition_source_value",
    ]

    vals = []
    for col in columns:
        v = row.get(col)
        if v is None:
            vals.append("NULL")
        elif isinstance(v, int):
            vals.append(str(v))
        else:
            vals.append(f"'{v}'")

    return f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(vals)});"


def main():
    init_logging()

    parser = argparse.ArgumentParser(description="Map canonical term to OMOP concept and output JSON/SQL.")
    parser.add_argument("--term", type=str, required=True, help="Canonical term (e.g., 'Type 2 diabetes mellitus').")
    parser.add_argument("--person_id", type=int, required=True, help="Person ID for the condition_occurrence row.")
    parser.add_argument(
        "--start_date",
        type=str,
        default=date.today().isoformat(),
        help="Start date for the condition (YYYY-MM-DD).",
    )
    parser.add_argument("--end_date", type=str, default=None, help="Optional end date (YYYY-MM-DD).")
    args = parser.parse_args()

    record = lookup_concept(args.term)
    if record is None:
        raise SystemExit(f"No OMOP concept found for term '{args.term}'.")

    row = build_condition_occurrence(
        person_id=args.person_id,
        concept_id=int(record["concept_id"]),
        start_date=args.start_date,
        end_date=args.end_date,
        source_value=args.term,
    )

    print("\n=== OMOP condition_occurrence (JSON) ===")
    print(json.dumps(row, indent=2))

    print("\n=== OMOP condition_occurrence (SQL) ===")
    print(format_sql_insert(row))


if __name__ == "__main__":
    main()
