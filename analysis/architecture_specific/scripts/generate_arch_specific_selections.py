#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ARCHITECTURES = ("MLR", "RF", "TabM", "MLR+RF")
BUDGETS = (3, 6, 9, 12, 15, 18, 21)
ENDPOINTS = ("FN", "UN")
SELECTOR_FEATURES = (
    "Initial_BW_kg",
    "DMI_kg_d",
    "Analysis_N_Intake_g_d",
    "N_to_DMI_g_kg",
    "DMI_per_BW",
    "N_Intake_per_BW",
    "Analysis_Diet_CP_pct_DM",
    "Analysis_Diet_EE_pct_DM",
    "Analysis_Diet_NDF_pct_DM",
    "Analysis_Diet_ADF_pct_DM",
    "Diet_Ash_pct_DM",
)

# P2 sampling representation: this selector uses a P2-tier sampling representation rather than
# requiring column-identical features to the fitted prediction model. Additional routine
# pre-measurement descriptors (including Diet_Ash_pct_DM here) characterize P2-space
# geometry for allocation only and do not redefine or substitute predictors in the model.
# Selector definition used for the reported analysis.
# P2 regions: z-standardized Euclidean PAM k-medoids using BUILD + SWAP.
# RSGS landmarks: one member per P2 region, initialized at the medoid and
# optimized by deterministic coordinate descent to minimize the mean of the FN
# and UN whole-pool reconstruction RMSEs. Reconstruction uses a Gaussian RBF on
# z-standardized P2, length scale = median non-zero pairwise P2 distance, and
# ridge lambda = 1e-5. Observed target FN/UN and Ingredient_Class never enter
# source selection. source_order supplies deterministic tie-breaking.
RBF_RIDGE = 1e-5
TOL = 1e-12


def _total_pam_cost(distance: np.ndarray, medoids: list[int]) -> float:
    return float(distance[:, medoids].min(axis=1).sum())


def pam(distance: np.ndarray, source_order: list[str], k: int) -> list[int]:
    """Deterministic PAM BUILD+SWAP used by the specified selector."""
    n = len(source_order)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    # BUILD: first medoid minimizes total distance.
    totals = distance.sum(axis=1)
    best_total = float(totals.min())
    first = min(
        (i for i, value in enumerate(totals) if abs(float(value) - best_total) <= TOL),
        key=lambda i: source_order[i],
    )
    medoids = [first]
    nearest = distance[:, first].copy()

    # BUILD: add the candidate with the largest reduction in total dissimilarity.
    while len(medoids) < k:
        best_candidate = None
        best_gain = -np.inf
        medoid_set = set(medoids)
        for candidate in range(n):
            if candidate in medoid_set:
                continue
            gain = float(np.maximum(0.0, nearest - distance[:, candidate]).sum())
            if (
                gain > best_gain + TOL
                or (
                    abs(gain - best_gain) <= TOL
                    and (best_candidate is None or source_order[candidate] < source_order[best_candidate])
                )
            ):
                best_candidate = candidate
                best_gain = gain
        assert best_candidate is not None
        medoids.append(best_candidate)
        nearest = np.minimum(nearest, distance[:, best_candidate])

    # SWAP: repeatedly apply the best improving medoid/non-medoid exchange.
    current_cost = _total_pam_cost(distance, medoids)
    while True:
        medoid_set = set(medoids)
        best_new = None
        best_cost = current_cost
        best_key = None
        for position, old in enumerate(medoids):
            for candidate in range(n):
                if candidate in medoid_set:
                    continue
                trial = medoids.copy()
                trial[position] = candidate
                cost = _total_pam_cost(distance, trial)
                key = (source_order[old], source_order[candidate])
                if cost < best_cost - TOL or (
                    abs(cost - best_cost) <= TOL and best_new is not None and key < best_key
                ):
                    best_new = trial
                    best_cost = cost
                    best_key = key
        if best_new is None or best_cost >= current_cost - TOL:
            break
        medoids = best_new
        current_cost = best_cost

    # Stable ordering is part of the reported selector definition because it defines region
    # order and therefore the coordinate-descent traversal and output Rank.
    return sorted(medoids, key=lambda i: source_order[i])


def _rbf_reconstruction_score(
    distance: np.ndarray,
    response: np.ndarray,
    selected: list[int],
    length_scale: float,
) -> float:
    selected_array = np.asarray(selected, dtype=int)
    kernel_selected = np.exp(
        -0.5 * (distance[np.ix_(selected_array, selected_array)] / length_scale) ** 2
    )
    kernel_selected = kernel_selected + RBF_RIDGE * np.eye(len(selected_array))
    coefficients = np.linalg.solve(kernel_selected, response[selected_array])
    kernel_all = np.exp(-0.5 * (distance[:, selected_array] / length_scale) ** 2)
    reconstructed = kernel_all @ coefficients
    endpoint_rmse = np.sqrt(np.mean((reconstructed - response) ** 2, axis=0))
    return float(endpoint_rmse.mean())


def response_surface_landmarks(
    z: np.ndarray,
    distance: np.ndarray,
    response: np.ndarray,
    medoids: list[int],
    source_order: list[str],
) -> tuple[list[int], float, float]:
    """Choose one landmark per blind P2-PAM region by response-surface reconstruction."""
    del z  # distance is the exact Euclidean matrix computed from this standardized P2 representation.

    upper = distance[np.triu_indices_from(distance, k=1)]
    positive = upper[upper > 0]
    if len(positive) == 0:
        raise ValueError("All pairwise P2 distances are zero; RBF length scale is undefined.")
    length_scale = float(np.median(positive))

    ordered_medoids = sorted(medoids, key=lambda i: source_order[i])
    medoid_distance = distance[:, ordered_medoids]
    # np.argmin takes the first minimum; because ordered_medoids is lexical this
    # is deterministic for exact ties.
    assignment = np.argmin(medoid_distance, axis=1)
    clusters = [np.where(assignment == j)[0].tolist() for j in range(len(ordered_medoids))]

    selected = ordered_medoids.copy()
    current = _rbf_reconstruction_score(distance, response, selected, length_scale)

    # Deterministic coordinate descent across P2 regions, one landmark per region.
    while True:
        changed = False
        for region, candidates in enumerate(clusters):
            current_member = selected[region]
            best_member = current_member
            best_score = current
            for candidate in sorted(candidates, key=lambda i: source_order[i]):
                trial = selected.copy()
                trial[region] = candidate
                score = _rbf_reconstruction_score(distance, response, trial, length_scale)
                if score < best_score - TOL or (
                    abs(score - best_score) <= TOL
                    and source_order[candidate] < source_order[best_member]
                ):
                    best_member = candidate
                    best_score = score
            if best_member != current_member:
                selected[region] = best_member
                current = best_score
                changed = True
            else:
                current = best_score
        if not changed:
            break

    return selected, current, length_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-546", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    parser.add_argument("--output-diagnostics", type=Path, required=True)
    args = parser.parse_args()

    animal = pd.read_csv(args.predictions_546)
    source_order = sorted(animal.Source_unit.astype(str).unique())
    source = (
        animal.groupby(["Source_unit", "Ingredient_Class"], as_index=False)
        .mean(numeric_only=True)
        .set_index("Source_unit")
        .loc[source_order]
        .reset_index()
    )
    source_domains = source.Ingredient_Class.astype(str).to_numpy()
    raw = source[list(SELECTOR_FEATURES)].to_numpy(float)
    imputed = SimpleImputer(strategy="median").fit_transform(raw)
    variable = np.std(imputed, axis=0, ddof=1) > 0
    z = StandardScaler().fit_transform(imputed[:, variable])
    distance = cdist(z, z)

    rows = []
    diagnostics = []
    for architecture in ARCHITECTURES:
        response = np.column_stack(
            [source[f"{architecture}_{endpoint}_pred"].to_numpy(float) for endpoint in ENDPOINTS]
        )
        response = StandardScaler().fit_transform(response)
        for budget in BUDGETS:
            medoids = pam(distance, source_order, budget)
            landmarks, reconstruction, length_scale = response_surface_landmarks(
                z, distance, response, medoids, source_order
            )
            for rank, index in enumerate(landmarks, 1):
                rows.append(
                    {
                        "Architecture": architecture,
                        "Geometry": architecture,
                        "Budget": budget,
                        "Rank": rank,
                        "Source_unit": source_order[index],
                        "Ingredient_Class": source_domains[index],
                    }
                )
            diagnostics.append(
                {
                    "Architecture": architecture,
                    "Budget": budget,
                    "P2_PAM_total_distance": float(distance[:, medoids].min(axis=1).sum()),
                    "Model_shape_reconstruction_RMSE": reconstruction,
                    "RBF_length_scale": length_scale,
                }
            )

    args.output_ledger.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_ledger, index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_diagnostics, index=False)
    print(args.output_ledger)
    print(args.output_diagnostics)


if __name__ == "__main__":
    main()
