# GitHub release checklist

Before freezing the manuscript submission:

1. Confirm the repository contains the expanded files, not only a ZIP archive.
2. Confirm the `reproducibility` GitHub Actions workflow passes on Python 3.12. It runs `python scripts/reproduce.py release-audit --jobs 1`, including prediction-level 64-row RN/TNE verification.
3. Confirm that the authors and responsible institutions permit public sharing of the packaged experimental data, literature-derived values, and artwork under the conditions in `DATA_RIGHTS.md`.
4. Create a GitHub Release tagged `v1.0.0` from the exact commit that passed the release audit.
5. Set the repository to Public and confirm that the repository URL, README, release, and Actions run open without signing in.
6. Put https://github.com/tanghui9580/sparse-target-pig-nitrogen-excretion in the manuscript Data and code availability statement. No archive DOI is required or claimed for this release.
7. Complete any corresponding-author telephone/fax fields required by the current journal submission system/guide; do not invent contact information.
8. From a clean clone of the frozen tag, rerun `python scripts/reproduce.py release-audit --jobs 1`; confirm the manifest and publication hashes remain PASS.
9. Keep `v1.0.0` immutable. If a scientific or file-level correction is required, create a new version/release rather than silently replacing the tagged artifact.
