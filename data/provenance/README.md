# Historical-study provenance

`HISTORICAL_170_STUDY_PROVENANCE.csv` maps every `Study_ID` used by `data/analysis_matrices/historical_906.csv` to a research-level source record.

## Authority and fallback hierarchy

1. The reviewer-ready V44 `Study_Index` is the primary scope/identity anchor.
2. If V44 did not carry a DOI/link, a verified DOI from the v35 metadata registry is used as the next source-identifier fallback.
3. Six Study_ID values that were absent from both identifier fields were independently resolved against publisher/journal metadata on 2026-09-22. These rows are marked `publisher_metadata_2026-09-22` or `journal_metadata_2026-09-22` in `Best_Identifier_Source`.
4. No DOI is inferred from the Study_ID text. `Best_Available_Source_Identifier` records the strongest traceable identifier available under this hierarchy; it may be a DOI, URL, formal citation/source label, or publisher-verified DOI.

The resulting table covers **170/170 Study_ID values and 170/170 best-available source identifiers**. `Best_Available_Title` is deliberately more conservative: it is populated only for titles from the externally verified v35 subset or the six publisher/journal-resolved rows. Older title strings remain in `Canonical_Title_v35` for audit history and should not all be interpreted as current canonical bibliographic titles.

`Records_in_historical_906` is calculated from the packaged manuscript matrix and may be smaller than the broader V44 database record count.

## QA-field interpretation

`CLEAR_records`, `TRACEABLE_CAVEAT_records`, and `REVIEW_FLAG_records` are record/field-level data-quality flags inherited from the broader reviewer-facing database. They describe issues such as reconstructed nutrient inputs, legacy field provenance, or sensitivity annotations. A nonzero `REVIEW_FLAG_records` value **does not mean that the paper identity or Study_ID mapping is unverified**. Study identity and field-level QA are separate concepts.

This table is provided for traceability and does not grant redistribution rights to the underlying source publications. See `../../DATA_RIGHTS.md`.
