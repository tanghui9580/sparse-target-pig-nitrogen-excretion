# Release-preparation changelog

## v1.0.0 — 2026-09-24

### V6.3.2 repository hardening
- Added release-level duplicate-contract checks for intentionally copied canonical inputs/results across self-contained subpackages.
- Added release-hygiene checks for caches/bytecode, dependency pinning, publication-reference counts, GitHub 100 MiB file threshold, and release metadata structure.
- Hardened `.gitattributes` so all text is LF-normalized and EPS/publication binaries cannot be altered by line-ending conversion.
- Removed packaged `__pycache__` / `.pyc` artifacts.
- Changed `heavy-smoke` to a genuinely representative 4-cell heavy-method check while preserving the exhaustive 112,896-cell `full-extension` analysis.
- Added a `release-audit` profile for CI: verify + selectors + light smoke + representative heavy smoke + Table S5 rebuild + publication build.
- Added an optimized prediction-level derived-endpoint path that recomputes all required selected-procedure updates while avoiding the large per-source intermediate file; the release CI now verifies all 64 RN/TNE source rows exactly.
- Strengthened historical provenance from 151 V44 DOI/link entries to 170/170 best-available source identifiers by using verified v35 fallback metadata and independent publisher/journal verification for six remaining Study_ID values. No DOI was inferred from Study_ID text.
- Clarified that provenance `REVIEW_FLAG_records` is field/record-level QA, not study-identity uncertainty.
- Corrected data-rights language so the package does not assert that experimental-data rights belong solely to one individual.

### V6.3.1 GitHub-readiness work
- Added MIT code license and separate data-rights notice.
- Added 170-study provenance table and automated coverage checks.
- Added Python 3.12 GitHub Actions CI.
- Changed figure export to process-isolated parallel rendering for publication-build robustness.

### V6.3 publication pipeline
- Added executable plotting/export code for all 15 figures.
- Added one-command publication-artifact build and locked publication-layer reference assets with SHA-256 verification.

Scientific results, reported numerical values, manuscript-facing table/figure numbering, and locked submission artwork are unchanged by the repository-hardening revisions above.
