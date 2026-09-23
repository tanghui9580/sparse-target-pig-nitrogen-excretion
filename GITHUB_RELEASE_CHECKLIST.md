# GitHub / permanent-archive release checklist

Before freezing the manuscript submission:

1. Create the GitHub repository and push the **expanded repository contents**, not only the ZIP archive.
2. Confirm the `reproducibility` GitHub Actions workflow passes on Python 3.12. It runs `python scripts/reproduce.py release-audit --jobs 1`.
3. Confirm the CI log includes PASS for the prediction-level 64-row RN/TNE (`derived-endpoints`) verification.
4. Create a GitHub Release tagged `v1.0.0` from the exact commit that passed the release audit.
5. Archive that release in Zenodo (or another permanent repository) and obtain the version DOI.
6. Add the real repository URL and version DOI to `CITATION.cff`.
7. Put the same GitHub URL and permanent DOI into the manuscript Reproducibility / Data and Code Availability statements.
8. Complete any corresponding-author telephone/fax fields required by the current journal submission system/guide; do not invent contact information.
9. From a clean clone of the frozen commit, rerun `python scripts/reproduce.py release-audit --jobs 1`; confirm the manifest and publication hashes remain PASS.
10. Do not modify the release assets after minting the DOI. If a scientific or file-level correction is required, create a new version/release rather than silently replacing the archived artifact.
