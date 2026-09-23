#!/usr/bin/env python3
"""Run the selected 546 source-excluded sampling x updating experiment.

Primary prior: post-S17 all-906 MLR+RF.
Sampling: Random, P2-PAM, RSGS.
Updating: Mean, Affine, CG-HPP, target-weighted TargetWeightedFullRF.
Budgets: 3, 6, 9, 12 sources.
Nuisance design: 84 blocks x 6 balanced outcome-record rotations = 504.

The three local-only models and matched NoUpdate are reported as boundaries,
not as formal updaters. Selected sources leave the evaluation population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


BUDGETS = (3, 6, 9, 12)
SAMPLING = ("Random", "P2-PAM", "RSGS")
FORMAL = ("Mean", "Affine", "CG-HPP", "TargetWeightedFullRF")
BOUNDARY = ("NoUpdate", "TargetOnlyRidge", "TargetOnlyElasticNet", "TargetOnlyRF")
ENDPOINTS = ("FN", "UN")
TARGETS = {"FN": "Analysis_Fecal_N_g_d", "UN": "Analysis_Urinary_N_g_d"}
PRIMARY_MODEL = "MLR+RF"
ROTATION_SEED = 20260804
RANDOM_SELECTION_SEED = 20260811
FULLRF_SEED = 20260524
FULLRF_TREES = 60
LOCAL_RF_SEED = 20260524
LOCAL_RF_TREES = 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_permutation(source: str, block: int, length: int) -> np.ndarray:
    token = f"{source}|{block}|{ROTATION_SEED}".encode("utf-8")
    seed = int(hashlib.sha256(token).hexdigest()[:16], 16) % (2**32)
    return np.random.default_rng(seed).permutation(length)


def pam(distance: np.ndarray, names: list[str], k: int) -> list[int]:
    """Deterministic BUILD+SWAP k-medoids with lexicographic ties."""
    n = len(names)
    if not 1 <= k <= n:
        raise ValueError(k)

    def objective(medoids: list[int]) -> float:
        return float(distance[:, medoids].min(axis=1).sum())

    totals = distance.sum(axis=0)
    first_value = totals.min()
    first = min((i for i, value in enumerate(totals) if abs(value - first_value) <= 1e-12), key=lambda i: names[i])
    medoids = [first]
    while len(medoids) < k:
        choices = []
        for candidate in range(n):
            if candidate in medoids:
                continue
            proposed = sorted(medoids + [candidate], key=lambda i: names[i])
            choices.append((objective(proposed), tuple(names[i] for i in proposed), candidate))
        _, _, chosen = min(choices)
        medoids.append(chosen)
        medoids.sort(key=lambda i: names[i])

    current = objective(medoids)
    while True:
        best = (current, tuple(names[i] for i in medoids), list(medoids))
        medoid_set = set(medoids)
        for old in medoids:
            for new in range(n):
                if new in medoid_set:
                    continue
                proposed = sorted([new if value == old else value for value in medoids], key=lambda i: names[i])
                score = objective(proposed)
                candidate = (score, tuple(names[i] for i in proposed), proposed)
                if candidate[:2] < best[:2]:
                    best = candidate
        if best[0] >= current - 1e-12:
            break
        current = best[0]
        medoids = best[2]
    return medoids


def assign_regions(distance: np.ndarray, medoids: list[int], names: list[str]) -> dict[int, list[int]]:
    ordered = sorted(medoids, key=lambda i: names[i])
    regions = {medoid: [] for medoid in ordered}
    for row in range(distance.shape[0]):
        values = distance[row, ordered]
        minimum = values.min()
        medoid = min(
            (ordered[j] for j, value in enumerate(values) if abs(value - minimum) <= 1e-12),
            key=lambda i: names[i],
        )
        regions[medoid].append(row)
    return regions


def rbf_objective(z: np.ndarray, response: np.ndarray, selected: list[int], length_scale: float) -> float:
    selected = list(selected)
    kernel_ss = np.exp(-0.5 * (cdist(z[selected], z[selected]) / length_scale) ** 2)
    kernel_xs = np.exp(-0.5 * (cdist(z, z[selected]) / length_scale) ** 2)
    coefficient = np.linalg.solve(kernel_ss + 1e-5 * np.eye(len(selected)), response[selected])
    reconstruction = kernel_xs @ coefficient
    return float(np.mean(np.sqrt(np.mean((response - reconstruction) ** 2, axis=0))))


def response_surface_landmarks(
    z: np.ndarray,
    distance: np.ndarray,
    response: np.ndarray,
    medoids: list[int],
    names: list[str],
) -> tuple[list[int], float, float]:
    regions = assign_regions(distance, medoids, names)
    region_keys = sorted(regions, key=lambda i: names[i])
    selected = list(region_keys)
    distances = pdist(z)
    positive = distances[distances > 0]
    length_scale = float(np.median(positive)) if len(positive) else 1.0
    current = rbf_objective(z, response, selected, length_scale)
    while True:
        changed = False
        for position, region in enumerate(region_keys):
            candidates = []
            for candidate in regions[region]:
                proposal = list(selected)
                proposal[position] = candidate
                score = rbf_objective(z, response, proposal, length_scale)
                candidates.append((score, names[candidate], candidate))
            score, _, candidate = min(candidates)
            if score < current - 1e-12 or (
                abs(score - current) <= 1e-12 and names[candidate] < names[selected[position]]
            ):
                selected[position] = candidate
                current = score
                changed = True
        if not changed:
            break
    return selected, current, length_scale


def metric_values(
    observed: np.ndarray,
    prediction: np.ndarray,
    reference_sd: np.ndarray,
) -> dict:
    rows = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        y = observed[:, endpoint_index]
        p = prediction[:, endpoint_index]
        residual = p - y
        rows[f"{endpoint}_RMSE"] = float(np.sqrt(np.mean(residual**2)))
        rows[f"{endpoint}_MAE"] = float(np.mean(np.abs(residual)))
        rows[f"{endpoint}_Bias"] = float(np.mean(residual))
        rows[f"{endpoint}_R2"] = float(r2_score(y, p)) if len(y) >= 2 else np.nan
    rows["Primary_NRMSE"] = float(
        0.5 * (rows["FN_RMSE"] / reference_sd[0] + rows["UN_RMSE"] / reference_sd[1])
    )
    return rows


def ridge_residual_update(
    calibration_prediction: np.ndarray,
    calibration_observed: np.ndarray,
    evaluation_prediction: np.ndarray,
    center: float,
    scale: float,
    sigma: float,
    domain_calibration: np.ndarray | None = None,
    domain_evaluation: np.ndarray | None = None,
    domains: list[str] | None = None,
) -> np.ndarray:
    scale = max(scale, 1e-8)
    sigma = max(sigma, 1e-8)
    z_calibration = (calibration_prediction - center) / scale
    z_evaluation = (evaluation_prediction - center) / scale
    residual = calibration_observed - calibration_prediction
    if domain_calibration is None:
        design = np.column_stack([np.ones(len(z_calibration)), z_calibration])
        evaluation_design = np.column_stack([np.ones(len(z_evaluation)), z_evaluation])
        precision = np.diag([1 / 4, 1.0])
    else:
        assert domain_evaluation is not None and domains is not None
        design = np.zeros((len(z_calibration), 2 + 2 * len(domains)))
        evaluation_design = np.zeros((len(z_evaluation), 2 + 2 * len(domains)))
        design[:, 0] = 1
        design[:, 1] = z_calibration
        evaluation_design[:, 0] = 1
        evaluation_design[:, 1] = z_evaluation
        domain_lookup = {domain: index for index, domain in enumerate(domains)}
        for row, domain in enumerate(domain_calibration):
            index = domain_lookup[str(domain)]
            design[row, 2 + 2 * index] = 1
            design[row, 2 + 2 * index + 1] = z_calibration[row]
        for row, domain in enumerate(domain_evaluation):
            index = domain_lookup[str(domain)]
            evaluation_design[row, 2 + 2 * index] = 1
            evaluation_design[row, 2 + 2 * index + 1] = z_evaluation[row]
        precision_values = [1 / 4, 1.0]
        for _ in domains:
            precision_values.extend([1.0, 4.0])
        precision = np.diag(precision_values)
    coefficient = np.linalg.solve(design.T @ design + precision, design.T @ residual)
    return evaluation_prediction + evaluation_design @ coefficient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-906", type=Path, required=True)
    parser.add_argument("--predictions-546", type=Path, required=True)
    parser.add_argument("--oof-673", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    historical = pd.read_csv(args.matrix_906)
    animal = pd.read_csv(args.predictions_546)
    oof = pd.read_csv(args.oof_673)
    model_spec = json.loads(args.model_spec.read_text(encoding="utf-8"))
    features = list(model_spec["features"])
    if (len(historical), len(animal), animal["Source_unit"].nunique()) != (906, 546, 91):
        raise AssertionError("Expected 906 historical, 546 target animals, and 91 sources")
    if not animal.groupby("Source_unit").size().eq(6).all():
        raise AssertionError("Every source must have six animal rows")

    source_order = sorted(animal["Source_unit"].astype(str).unique())
    source_lookup = {source: index for index, source in enumerate(source_order)}
    animal_by_source = {
        source: animal.index[animal["Source_unit"].astype(str).eq(source)].tolist()
        for source in source_order
    }
    for source in source_order:
        animal_by_source[source].sort(key=lambda index: int(animal.loc[index, "Pig_Ordinal"]))
    source = (
        animal.groupby(["Source_unit", "Ingredient_Class"], as_index=False)
        .mean(numeric_only=True)
        .set_index("Source_unit")
        .loc[source_order]
        .reset_index()
    )
    source_domains = source["Ingredient_Class"].astype(str).to_numpy()
    all_domains = sorted(source["Ingredient_Class"].astype(str).unique())

    # P2-PAM uses the pre-measurement P2 information tier; the sampling representation is not required to be an
    # immutable column-identical vector across analytical modules. These descriptors
    # represent P2 geometry for measurement allocation only; Diet_Ash_pct_DM here
    # does not replace starch or redefine the specified predictive-model P2 input set.
    selector_features = [
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
    ]
    selector_raw = source[selector_features].to_numpy(float)
    selector_imputed = SimpleImputer(strategy="median").fit_transform(selector_raw)
    variable = np.std(selector_imputed, axis=0, ddof=1) > 0
    selector_features_active = [name for name, keep in zip(selector_features, variable) if keep]
    z = StandardScaler().fit_transform(selector_imputed[:, variable])
    distance = cdist(z, z)

    response = np.column_stack(
        [source[f"{PRIMARY_MODEL}_{endpoint}_pred"].to_numpy(float) for endpoint in ENDPOINTS]
    )
    response = StandardScaler().fit_transform(response)
    structured_sets: dict[str, dict[int, list[int]]] = {"P2-PAM": {}, "RSGS": {}}
    selector_rows = []
    selector_diagnostics = []
    for budget in BUDGETS:
        medoids = pam(distance, source_order, budget)
        landmarks, reconstruction, length_scale = response_surface_landmarks(
            z, distance, response, medoids, source_order
        )
        structured_sets["P2-PAM"][budget] = medoids
        structured_sets["RSGS"][budget] = landmarks
        for policy, selected in (("P2-PAM", medoids), ("RSGS", landmarks)):
            for rank, index in enumerate(selected, 1):
                selector_rows.append(
                    {
                        "Sampling": policy,
                        "Budget": budget,
                        "Rank": rank,
                        "Source_unit": source_order[index],
                        "Ingredient_Class": source_domains[index],
                    }
                )
        selector_diagnostics.append(
            {
                "Budget": budget,
                "P2_PAM_total_distance": float(distance[:, medoids].min(axis=1).sum()),
                "Model_shape_reconstruction_RMSE": reconstruction,
                "RBF_length_scale": length_scale,
            }
        )
    pd.DataFrame(selector_rows).to_csv(output_dir / "STRUCTURED_SELECTION_LEDGERS.csv", index=False)
    pd.DataFrame(selector_diagnostics).to_csv(output_dir / "SELECTOR_DIAGNOSTICS.csv", index=False)

    rotation_index = np.empty((504, len(source_order)), dtype=int)
    rotation_rows = []
    for realization in range(504):
        block, rotation = divmod(realization, 6)
        for source_index, source_name in enumerate(source_order):
            rows = animal_by_source[source_name]
            permutation = stable_permutation(source_name, block, len(rows))
            row_index = rows[int(permutation[rotation])]
            rotation_index[realization, source_index] = row_index
            rotation_rows.append(
                {
                    "Realization": realization,
                    "Block": block,
                    "Rotation": rotation,
                    "Source_unit": source_name,
                    "Record_ID": animal.loc[row_index, "Record_ID"],
                    "Pig_Ordinal": int(animal.loc[row_index, "Pig_Ordinal"]),
                }
            )
    pd.DataFrame(rotation_rows).to_csv(output_dir / "BALANCED_504_ANIMAL_ROTATION_LEDGER.csv", index=False)

    random_sets: dict[tuple[int, int], list[int]] = {}
    rng = np.random.default_rng(RANDOM_SELECTION_SEED)
    random_rows = []
    for realization in range(504):
        permutation = rng.permutation(len(source_order))
        for budget in BUDGETS:
            selected = permutation[:budget].tolist()
            random_sets[(realization, budget)] = selected
            for rank, index in enumerate(selected, 1):
                random_rows.append(
                    {
                        "Realization": realization,
                        "Budget": budget,
                        "Rank": rank,
                        "Source_unit": source_order[index],
                        "Ingredient_Class": source_domains[index],
                    }
                )
    pd.DataFrame(random_rows).to_csv(output_dir / "RANDOM_504_SELECTION_LEDGER.csv", index=False)

    reference_sd = np.array(
        [source[TARGETS[endpoint]].std(ddof=1) for endpoint in ENDPOINTS], dtype=float
    )
    observed_source = np.column_stack(
        [source[TARGETS[endpoint]].to_numpy(float) for endpoint in ENDPOINTS]
    )
    baseline_source = np.column_stack(
        [source[f"{PRIMARY_MODEL}_{endpoint}_pred"].to_numpy(float) for endpoint in ENDPOINTS]
    )
    observed_animal = np.column_stack(
        [animal[TARGETS[endpoint]].to_numpy(float) for endpoint in ENDPOINTS]
    )
    baseline_animal = np.column_stack(
        [animal[f"{PRIMARY_MODEL}_{endpoint}_pred"].to_numpy(float) for endpoint in ENDPOINTS]
    )
    hyper = {}
    for endpoint in ENDPOINTS:
        y = oof[TARGETS[endpoint]].to_numpy(float)
        prediction = oof[f"{PRIMARY_MODEL}_{endpoint}_pred"].to_numpy(float)
        hyper[endpoint] = {
            "center": float(np.mean(prediction)),
            "scale": float(np.std(prediction, ddof=1)),
            "sigma": float(np.std(y - prediction, ddof=1)),
        }

    history_imputer = SimpleImputer(strategy="median").fit(historical[features].to_numpy(float))
    history_x = history_imputer.transform(historical[features].to_numpy(float))
    animal_x = history_imputer.transform(animal[features].to_numpy(float))
    history_y = {
        endpoint: historical[TARGETS[endpoint]].to_numpy(float) for endpoint in ENDPOINTS
    }
    selected_config = {
        endpoint: model_spec["selected"][endpoint]["config"] for endpoint in ENDPOINTS
    }

    tasks = []
    for realization in range(504):
        for sampling in SAMPLING:
            for budget in BUDGETS:
                tasks.append((len(tasks), realization, sampling, budget))
    formal_prediction_cube = np.full(
        (len(tasks), len(FORMAL), len(source_order), len(ENDPOINTS)), np.nan, dtype=np.float32
    )
    selected_mask_cube = np.zeros((len(tasks), len(source_order)), dtype=bool)

    def selected_indices(realization: int, sampling: str, budget: int) -> list[int]:
        if sampling == "Random":
            return random_sets[(realization, budget)]
        return structured_sets[sampling][budget]

    def evaluate_task(task_id: int, realization: int, sampling: str, budget: int) -> list[dict]:
        selected = selected_indices(realization, sampling, budget)
        selected_set = set(selected)
        evaluation = [index for index in range(len(source_order)) if index not in selected_set]
        selected_mask_cube[task_id, selected] = True
        calibration_rows = np.array([rotation_index[realization, index] for index in selected], dtype=int)
        calibration_domains = source_domains[selected]
        evaluation_domains = source_domains[evaluation]
        predictions: dict[str, np.ndarray] = {}

        mean_prediction = np.empty((len(evaluation), 2), dtype=float)
        affine_prediction = np.empty_like(mean_prediction)
        hpp_prediction = np.empty_like(mean_prediction)
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            calibration_residual = (
                observed_animal[calibration_rows, endpoint_index]
                - baseline_animal[calibration_rows, endpoint_index]
            )
            mean_prediction[:, endpoint_index] = (
                baseline_source[evaluation, endpoint_index] + np.mean(calibration_residual)
            )
            affine_prediction[:, endpoint_index] = ridge_residual_update(
                baseline_animal[calibration_rows, endpoint_index],
                observed_animal[calibration_rows, endpoint_index],
                baseline_source[evaluation, endpoint_index],
                hyper[endpoint]["center"],
                hyper[endpoint]["scale"],
                hyper[endpoint]["sigma"],
            )
            hpp_prediction[:, endpoint_index] = ridge_residual_update(
                baseline_animal[calibration_rows, endpoint_index],
                observed_animal[calibration_rows, endpoint_index],
                baseline_source[evaluation, endpoint_index],
                hyper[endpoint]["center"],
                hyper[endpoint]["scale"],
                hyper[endpoint]["sigma"],
                calibration_domains,
                evaluation_domains,
                all_domains,
            )
        predictions["Mean"] = mean_prediction
        predictions["Affine"] = affine_prediction
        predictions["CG-HPP"] = hpp_prediction

        full_prediction = np.empty((len(evaluation), 2), dtype=float)
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            combined_x = np.vstack([history_x, animal_x[calibration_rows]])
            combined_y = np.concatenate(
                [history_y[endpoint], observed_animal[calibration_rows, endpoint_index]]
            )
            weights = np.concatenate(
                [np.ones(len(historical)), np.full(budget, len(historical) / budget)]
            )
            config = selected_config[endpoint]
            model = RandomForestRegressor(
                n_estimators=FULLRF_TREES,
                min_samples_leaf=int(config["min_samples_leaf"]),
                max_features=float(config["max_features"]),
                max_depth=config.get("max_depth"),
                random_state=FULLRF_SEED + endpoint_index * 100,
                n_jobs=1,
            )
            model.fit(combined_x, combined_y, sample_weight=weights)
            individual_prediction = model.predict(animal_x)
            for output_index, source_index in enumerate(evaluation):
                rows = animal_by_source[source_order[source_index]]
                full_prediction[output_index, endpoint_index] = float(
                    np.mean(individual_prediction[rows])
                )
        predictions["TargetWeightedFullRF"] = full_prediction

        local_predictions = {
            "TargetOnlyRidge": np.empty((len(evaluation), 2), dtype=float),
            "TargetOnlyElasticNet": np.empty((len(evaluation), 2), dtype=float),
            "TargetOnlyRF": np.empty((len(evaluation), 2), dtype=float),
        }
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            x_calibration = animal_x[calibration_rows]
            y_calibration = observed_animal[calibration_rows, endpoint_index]
            scaler = StandardScaler().fit(x_calibration)
            x_calibration_scaled = scaler.transform(x_calibration)
            x_all_scaled = scaler.transform(animal_x)
            ridge = Ridge(alpha=1.0).fit(x_calibration_scaled, y_calibration)
            elastic = ElasticNet(
                alpha=0.1, l1_ratio=0.5, max_iter=100000, random_state=LOCAL_RF_SEED
            ).fit(x_calibration_scaled, y_calibration)
            local_rf = RandomForestRegressor(
                n_estimators=LOCAL_RF_TREES,
                min_samples_leaf=1,
                max_features=1.0,
                random_state=LOCAL_RF_SEED + endpoint_index * 100,
                n_jobs=1,
            ).fit(x_calibration, y_calibration)
            individual_by_method = {
                "TargetOnlyRidge": ridge.predict(x_all_scaled),
                "TargetOnlyElasticNet": elastic.predict(x_all_scaled),
                "TargetOnlyRF": local_rf.predict(animal_x),
            }
            for method, individual_prediction in individual_by_method.items():
                for output_index, source_index in enumerate(evaluation):
                    rows = animal_by_source[source_order[source_index]]
                    local_predictions[method][output_index, endpoint_index] = float(
                        np.mean(individual_prediction[rows])
                    )

        rows = []
        common = {
            "Realization": realization,
            "Block": realization // 6,
            "Rotation": realization % 6,
            "Sampling": sampling,
            "Budget": budget,
            "n_calibration_sources": budget,
            "n_evaluation_sources": len(evaluation),
        }
        for method_index, method in enumerate(FORMAL):
            prediction = predictions[method]
            formal_prediction_cube[task_id, method_index, evaluation, :] = prediction.astype(np.float32)
            rows.append(
                {
                    **common,
                    "Method": method,
                    "Role": "formal_updater",
                    **metric_values(observed_source[evaluation], prediction, reference_sd),
                }
            )
        rows.append(
            {
                **common,
                "Method": "NoUpdate",
                "Role": "zero_shot_control",
                **metric_values(
                    observed_source[evaluation], baseline_source[evaluation], reference_sd
                ),
            }
        )
        for method in ("TargetOnlyRidge", "TargetOnlyElasticNet", "TargetOnlyRF"):
            rows.append(
                {
                    **common,
                    "Method": method,
                    "Role": "local_only_boundary",
                    **metric_values(
                        observed_source[evaluation], local_predictions[method], reference_sd
                    ),
                }
            )
        return rows

    nested_rows = Parallel(n_jobs=4, prefer="threads", batch_size=1, verbose=10)(
        delayed(evaluate_task)(*task) for task in tasks
    )
    metric_rows = [row for group in nested_rows for row in group]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "FACTORIAL_504_REALIZATION_METRICS.csv", index=False)

    summary = (
        metrics.groupby(["Role", "Sampling", "Method", "Budget"])
        .agg(
            n_realizations=("Realization", "size"),
            Primary_mean=("Primary_NRMSE", "mean"),
            Primary_SD=("Primary_NRMSE", "std"),
            Primary_q025=("Primary_NRMSE", lambda values: values.quantile(0.025)),
            Primary_median=("Primary_NRMSE", "median"),
            Primary_q975=("Primary_NRMSE", lambda values: values.quantile(0.975)),
            FN_RMSE_mean=("FN_RMSE", "mean"),
            FN_R2_mean=("FN_R2", "mean"),
            UN_RMSE_mean=("UN_RMSE", "mean"),
            UN_R2_mean=("UN_R2", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "FACTORIAL_CELL_SUMMARY.csv", index=False)

    aubc_rows = []
    for (role, sampling, method, realization), group in metrics.groupby(
        ["Role", "Sampling", "Method", "Realization"]
    ):
        ordered = group.sort_values("Budget")
        if tuple(ordered["Budget"]) != BUDGETS:
            raise AssertionError("Incomplete budget curve")
        aubc_rows.append(
            {
                "Role": role,
                "Sampling": sampling,
                "Method": method,
                "Realization": realization,
                "AUBC_3_12": float(np.trapezoid(ordered["Primary_NRMSE"], ordered["Budget"]) / 9),
            }
        )
    aubc = pd.DataFrame(aubc_rows)
    aubc.to_csv(output_dir / "FACTORIAL_AUBC_BY_REALIZATION.csv", index=False)
    aubc_summary = (
        aubc.groupby(["Role", "Sampling", "Method"])
        .agg(
            n_realizations=("Realization", "size"),
            AUBC_mean=("AUBC_3_12", "mean"),
            AUBC_SD=("AUBC_3_12", "std"),
            AUBC_q025=("AUBC_3_12", lambda values: values.quantile(0.025)),
            AUBC_median=("AUBC_3_12", "median"),
            AUBC_q975=("AUBC_3_12", lambda values: values.quantile(0.975)),
        )
        .reset_index()
        .sort_values(["Role", "AUBC_mean", "Sampling", "Method"])
    )
    aubc_summary.to_csv(output_dir / "FACTORIAL_AUBC_SUMMARY.csv", index=False)

    formal_summary = aubc_summary.loc[aubc_summary["Role"].eq("formal_updater")].copy()
    winner = formal_summary.sort_values(["AUBC_mean", "Sampling", "Method"]).iloc[0]
    selected_sampling = str(winner["Sampling"])
    selected_method = str(winner["Method"])

    task_lookup = {
        (realization, sampling, budget): task_id
        for task_id, realization, sampling, budget in tasks
    }
    common_rows = []
    for realization in range(504):
        for budget in BUDGETS:
            for left_index in range(len(SAMPLING)):
                for right_index in range(left_index + 1, len(SAMPLING)):
                    left, right = SAMPLING[left_index], SAMPLING[right_index]
                    left_task = task_lookup[(realization, left, budget)]
                    right_task = task_lookup[(realization, right, budget)]
                    common_mask = ~(selected_mask_cube[left_task] | selected_mask_cube[right_task])
                    evaluation = np.where(common_mask)[0]
                    for method_index, method in enumerate(FORMAL):
                        left_prediction = formal_prediction_cube[left_task, method_index, evaluation].astype(float)
                        right_prediction = formal_prediction_cube[right_task, method_index, evaluation].astype(float)
                        left_metric = metric_values(observed_source[evaluation], left_prediction, reference_sd)
                        right_metric = metric_values(observed_source[evaluation], right_prediction, reference_sd)
                        common_rows.append(
                            {
                                "Realization": realization,
                                "Budget": budget,
                                "Method": method,
                                "Left_sampling": left,
                                "Right_sampling": right,
                                "n_common_unmeasured": len(evaluation),
                                "Left_Primary": left_metric["Primary_NRMSE"],
                                "Right_Primary": right_metric["Primary_NRMSE"],
                                "Delta_left_minus_right": left_metric["Primary_NRMSE"]
                                - right_metric["Primary_NRMSE"],
                            }
                        )
    common = pd.DataFrame(common_rows)
    common.to_csv(output_dir / "COMMON_UNMEASURED_PAIRED_RESULTS.csv", index=False)
    common_summary = (
        common.groupby(["Method", "Left_sampling", "Right_sampling", "Budget"])
        .agg(
            n_realizations=("Realization", "size"),
            Delta_mean=("Delta_left_minus_right", "mean"),
            Delta_SD=("Delta_left_minus_right", "std"),
            Delta_q025=("Delta_left_minus_right", lambda values: values.quantile(0.025)),
            Delta_median=("Delta_left_minus_right", "median"),
            Delta_q975=("Delta_left_minus_right", lambda values: values.quantile(0.975)),
        )
        .reset_index()
    )
    common_summary.to_csv(output_dir / "COMMON_UNMEASURED_PAIRED_SUMMARY.csv", index=False)

    selected_method_index = FORMAL.index(selected_method)
    selected_rows = []
    for realization in range(504):
        for budget in BUDGETS:
            task_id = task_lookup[(realization, selected_sampling, budget)]
            prediction = formal_prediction_cube[task_id, selected_method_index].astype(float)
            evaluation = np.where(~selected_mask_cube[task_id])[0]
            for source_index in evaluation:
                selected_rows.append(
                    {
                        "Realization": realization,
                        "Block": realization // 6,
                        "Rotation": realization % 6,
                        "Budget": budget,
                        "Source_unit": source_order[source_index],
                        "Ingredient_Class": source_domains[source_index],
                        "FN_observed": observed_source[source_index, 0],
                        "UN_observed": observed_source[source_index, 1],
                        "FN_predicted": prediction[source_index, 0],
                        "UN_predicted": prediction[source_index, 1],
                    }
                )
    pd.DataFrame(selected_rows).to_csv(
        output_dir / "SELECTED_PROCEDURE_SOURCE_PREDICTIONS.csv.gz", index=False, compression="gzip"
    )

    metadata = {
        "status": "COMPLETE",
        "primary_prior": PRIMARY_MODEL,
        "budgets": list(BUDGETS),
        "sampling": list(SAMPLING),
        "formal_updaters": list(FORMAL),
        "controls_and_boundaries": list(BOUNDARY),
        "realizations": 504,
        "rotation_design": "84 independent source-wise pig-record permutations x 6 balanced rotations",
        "rotation_seed": ROTATION_SEED,
        "random_selection_seed": RANDOM_SELECTION_SEED,
        "selected_source_exit": True,
        "metric_denominator": {
            "FN_full_91_source_sample_SD": float(reference_sd[0]),
            "UN_full_91_source_sample_SD": float(reference_sd[1]),
        },
        "selector_features": selector_features_active,
        "starch_selector_rule": "excluded because all 546 values are structurally missing",
        "response_surface_selector": {
            "regions": "P2-PAM Voronoi regions",
            "surface": "all-906 MLR+RF FN/UN source-mean predictions",
            "kernel": "RBF",
            "length_scale": "median positive pairwise P2 distance",
            "ridge": 1e-5,
        },
        "updater_hyper": hyper,
        "fullrf": {
            "target_weight": "906/b per local animal; historical weight=1",
            "trees": FULLRF_TREES,
            "seed": FULLRF_SEED,
            "endpoint_config": selected_config,
        },
        "local_only": {
            "Ridge_alpha": 1.0,
            "ElasticNet_alpha": 0.1,
            "ElasticNet_l1_ratio": 0.5,
            "RF_trees": LOCAL_RF_TREES,
            "RF_seed": LOCAL_RF_SEED,
        },
        "procedure_selection": "minimum mean AUBC_3_12 across 504 realizations; point selection",
        "selected_sampling": selected_sampling,
        "selected_updater": selected_method,
        "selected_AUBC_mean": float(winner["AUBC_mean"]),
        "input_checksums": {
            "matrix_906_sha256": sha256(args.matrix_906),
            "predictions_546_sha256": sha256(args.predictions_546),
            "oof_673_sha256": sha256(args.oof_673),
            "model_spec_sha256": sha256(args.model_spec),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "leakage_assertions": {
            "sampling_uses_FN_UN_outcomes": False,
            "RSGS_uses_only_pre_measurement_predictions_and_P2": True,
            "one_outcome_record_per_selected_source": True,
            "other_five_selected_source_labels_visible": False,
            "selected_source_in_evaluation": False,
            "common_unmeasured_used_for_sampling_pairwise": True,
        },
    }
    metadata_path = output_dir / "target_updating_run_metadata.json"
    save_json(metadata_path, metadata)
    metadata["output_files"] = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path != metadata_path
    }
    save_json(metadata_path, metadata)
    print(
        f"FACTORIAL_546_COMPLETE sampling={selected_sampling} updater={selected_method} AUBC={float(winner['AUBC_mean']):.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
