# Clinical-NLP-for-Unstructured-EHR-Mining

A demo "factory line" for clinical NLP using BioBERT/ClinicalBERT + OMOP.

This repo is structured into **stations** to mirror how healthcare AI teams build production pipelines:

1. **Scanner**: extract entities (diseases, drugs, dosages) from messy clinical notes (`ner_extraction.py`).
2. **Logic**: resolve jargon/abbreviations using vector similarity (synonyms → canonical terms) (`entity_linking.py`).
3. **Translator / Filer**: map canonical terms to OMOP concept IDs + build OMOP condition_occurrence output (`omop_mapping.py`).
4. **Storage**: store structured data into an OMOP `condition_occurrence` table (`storage/` scripts).

---

## 🧰 Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> ⚠️ Some models download from Hugging Face on first run. Ensure you have internet access.

---

## 1) Scanner (BioBERT NER)

Extract disease/drug entities and detect negation:

```bash
python ner_extraction.py --file data/synthetic_notes.txt
```

Or run a single note:

```bash
python ner_extraction.py --text "Patient denies history of Diabetes, but is on metformin 500mg."
```

---

## 2) Logic (Entity Linking / Normalization)

Resolve abbreviations/synonyms to canonical medical terms, including Singapore medical shorthand:

```bash
python entity_linking.py --term "T2DM"
```

---

## 3) Translator / Filer (OMOP Concept Mapping)

Look up OMOP concepts via **OHDSI Athena REST API** (with a local fallback if the API is unreachable) and format a `condition_occurrence` payload.

This module now returns multiple candidates with confidence scores (if available from Athena).

```bash
python omop_mapping.py --term "Type 2 diabetes mellitus" --person_id 123
```

---

## 4) Storage (OMOP)

Create a demo OMOP `condition_occurrence` database and insert a record:

```bash
python storage/store.py --person_id 123 --concept_id 201820 --start_date 2026-03-15 --source "T2DM"
```

---

## Notes on Singapore context

The synthetic notes include SG-style abbreviations (e.g., **T2DM**, **HX**, **kencing manis**, **tak boleh**), code-switching, and shorthand common in local clinical documentation.

---

## 0) Optional: Run the end-to-end pipeline

For a single command that demonstrates everything from raw notes to OMOP storage:

```bash
python pipeline.py --note-file data/synthetic_notes.txt --person 123
```

## Extending this demo

- Swap in a more specialized NER model (e.g., ClinicalBERT with an NER head).
- Replace the local `translator/vocab.csv` with a full OHDSI Athena dump or API.
- Add additional "stations" (e.g., de-identification, event normalization, cohort builder).
