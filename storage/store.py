#!/usr/bin/env python3
"""Storage station: create OMOP condition_occurrence rows.

This demo uses SQLite to emulate an OMOP CDM repository.

Run:
    python storage/store.py --person_id 123 --concept_id 201820 --start_date 2026-03-15
"""

import argparse
import os
import sqlite3
from datetime import datetime


def init_db(db_path: str, schema_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    return conn


def insert_condition(
    conn: sqlite3.Connection,
    person_id: int,
    condition_concept_id: int,
    condition_start_date: str,
    condition_end_date: str = None,
    condition_source_value: str = None,
):
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO condition_occurrence (
            person_id,
            condition_concept_id,
            condition_start_date,
            condition_end_date,
            condition_source_value
        ) VALUES (?, ?, ?, ?, ?)""",
        (person_id, condition_concept_id, condition_start_date, condition_end_date, condition_source_value),
    )
    conn.commit()
    return cur.lastrowid


def main():
    parser = argparse.ArgumentParser(description="Storage station: store a condition in an OMOP-like DB.")
    parser.add_argument("--person_id", type=int, required=True)
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--start_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--source", type=str, default=None, help="Original source value (e.g., 'T2DM')")
    parser.add_argument(
        "--db",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "omop_demo.db"),
        help="SQLite file path for the demo OMOP database",
    )
    args = parser.parse_args()

    # Basic validation
    try:
        datetime.fromisoformat(args.start_date)
        if args.end_date:
            datetime.fromisoformat(args.end_date)
    except Exception as e:
        raise SystemExit(f"ERROR: dates must be ISO format YYYY-MM-DD. {e}")

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    conn = init_db(args.db, schema_path)

    row_id = insert_condition(
        conn,
        person_id=args.person_id,
        condition_concept_id=args.concept_id,
        condition_start_date=args.start_date,
        condition_end_date=args.end_date,
        condition_source_value=args.source,
    )

    print("✅ Inserted condition_occurrence row")
    print(f"condition_occurrence_id = {row_id}")
    print(f"db file = {args.db}")


if __name__ == "__main__":
    main()
