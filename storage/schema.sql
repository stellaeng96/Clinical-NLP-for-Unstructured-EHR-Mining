-- OMOP CDM v5+ subset for condition_occurrence (demo)

CREATE TABLE IF NOT EXISTS condition_occurrence (
    condition_occurrence_id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL,
    condition_concept_id INTEGER NOT NULL,
    condition_start_date TEXT NOT NULL,
    condition_end_date TEXT,
    condition_type_concept_id INTEGER NOT NULL DEFAULT 38000175,
    stop_reason TEXT,
    provider_id INTEGER,
    visit_occurrence_id INTEGER,
    condition_source_value TEXT,
    condition_source_concept_id INTEGER
);
