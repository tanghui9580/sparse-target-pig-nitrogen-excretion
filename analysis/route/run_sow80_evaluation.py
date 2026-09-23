#!/usr/bin/env python3
"""Apply the selected 546 target-updating procedure to the SOW80 cross-physiology dataset.

The 546 factorial chooses the sampling philosophy and updater.  This script
does not reselect them on SOW outcomes.  It rebuilds target-local selections
from SOW P2/predictions, reveals only the selected labels, removes selected
records from evaluation, and tests four prespecified starting architectures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from run_546_sampling_updating_factorial import (
    BUDGETS,
    FULLRF_SEED,
    FULLRF_TREES,
    LOCAL_RF_SEED,
    LOCAL_RF_TREES,
    RANDOM_SELECTION_SEED,
    metric_values,
    response_surface_landmarks,
    pam,
    ridge_residual_update,
)


ARCHITECTURES = ("MLR", "RF", "TabM", "MLR+RF")
ENDPOINTS = ("FN", "UN")
TARGETS = {"FN": "Analysis_Fecal_N_g_d", "UN": "Analysis_Urinary_N_g_d"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def oof_hyper(primary_oof: pd.DataFrame, tabm_oof: pd.DataFrame) -> dict:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for architecture in ("MLR", "RF", "MLR+RF"):
        result[architecture] = {}
        for endpoint in ENDPOINTS:
            y = primary_oof[TARGETS[endpoint]].to_numpy(float)
            p = primary_oof[f"{architecture}_{endpoint}_pred"].to_numpy(float)
            result[architecture][endpoint] = {
                "center": float(np.mean(p)),
                "scale": float(np.std(p, ddof=1)),
                "sigma": float(np.std(y - p, ddof=1)),
            }
    result["TabM"] = {}
    for endpoint in ENDPOINTS:
        rows = tabm_oof.loc[tabm_oof.Endpoint.eq(endpoint)].copy()
        if len(rows) != 673 or rows.Record_ID.astype(str).nunique() != 673:
            raise AssertionError(f"TabM OOF analysis input mismatch for {endpoint}")
        y = rows.Observed.to_numpy(float)
        p = rows.Predicted.to_numpy(float)
        result["TabM"][endpoint] = {
            "center": float(np.mean(p)),
            "scale": float(np.std(p, ddof=1)),
            "sigma": float(np.std(y - p, ddof=1)),
        }
    return result


def update_prior(
    method: str,
    baseline: np.ndarray,
    observed: np.ndarray,
    selected: list[int],
    evaluation: list[int],
    domains: np.ndarray,
    all_domains: list[str],
    hyper: dict[str, dict[str, float]],
) -> np.ndarray:
    out = np.empty((len(evaluation), 2), dtype=float)
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        if method == "Mean":
            residual = observed[selected, endpoint_index] - baseline[selected, endpoint_index]
            out[:, endpoint_index] = baseline[evaluation, endpoint_index] + np.mean(residual)
        elif method == "Affine":
            hp = hyper[endpoint]
            out[:, endpoint_index] = ridge_residual_update(
                baseline[selected, endpoint_index],
                observed[selected, endpoint_index],
                baseline[evaluation, endpoint_index],
                hp["center"],
                hp["scale"],
                hp["sigma"],
            )
        elif method == "CG-HPP":
            hp = hyper[endpoint]
            out[:, endpoint_index] = ridge_residual_update(
                baseline[selected, endpoint_index],
                observed[selected, endpoint_index],
                baseline[evaluation, endpoint_index],
                hp["center"],
                hp["scale"],
                hp["sigma"],
                domains[selected],
                domains[evaluation],
                all_domains,
            )
        else:
            raise ValueError(method)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-906", type=Path, required=True)
    parser.add_argument("--sow-predictions", type=Path, required=True)
    parser.add_argument("--primary-oof", type=Path, required=True)
    parser.add_argument("--tabm-oof", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--target-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    historical = pd.read_csv(args.matrix_906)
    sow = pd.read_csv(args.sow_predictions)
    primary_oof = pd.read_csv(args.primary_oof)
    tabm_oof = pd.read_csv(args.tabm_oof)
    model_spec = json.loads(args.model_spec.read_text(encoding="utf-8"))
    target_spec = json.loads(args.target_spec.read_text(encoding="utf-8"))
    if target_spec.get("status") != "COMPLETE":
        raise AssertionError("Target-updating specification is incomplete")
    selected_sampling = target_spec["selected_sampling"]
    selected_updater = target_spec["selected_updater"]
    if selected_sampling not in ("Random", "P2-PAM", "RSGS"):
        raise AssertionError(f"Unknown selected sampling: {selected_sampling}")
    if selected_updater not in ("Mean", "Affine", "CG-HPP", "TargetWeightedFullRF"):
        raise AssertionError(f"Unknown selected updater: {selected_updater}")
    if len(sow) != 80 or sow.Study_ID.astype(str).nunique() != 8:
        raise AssertionError("Expected SOW80 analysis dataset")
    if set(historical.Study_ID.astype(str)) & set(sow.Study_ID.astype(str)):
        raise AssertionError("Historical/SOW Study_ID overlap")

    features = list(model_spec["features"])
    selector_candidates = list(target_spec["selector_features"])
    usable = []
    raw_columns = []
    for column in selector_candidates:
        values = pd.to_numeric(sow[column], errors="coerce")
        if values.notna().any():
            usable.append(column)
            raw_columns.append(values.to_numpy(float))
    selector_raw = np.column_stack(raw_columns)
    selector_imputed = SimpleImputer(strategy="median").fit_transform(selector_raw)
    variable = np.std(selector_imputed, axis=0, ddof=1) > 0
    selector_features = [column for column, keep in zip(usable, variable) if keep]
    z = StandardScaler().fit_transform(selector_imputed[:, variable])
    distance = cdist(z, z)
    names = sow.Record_ID.astype(str).tolist()
    primary_response = np.column_stack(
        [sow[f"MLR+RF_{endpoint}_pred"].to_numpy(float) for endpoint in ENDPOINTS]
    )
    primary_response = StandardScaler().fit_transform(primary_response)

    structured_sets: dict[int, list[int]] = {}
    selector_diagnostics = []
    if selected_sampling in ("P2-PAM", "RSGS"):
        for budget in BUDGETS:
            medoids = pam(distance, names, budget)
            if selected_sampling == "P2-PAM":
                selected = medoids
                reconstruction = np.nan
                length_scale = np.nan
            else:
                selected, reconstruction, length_scale = response_surface_landmarks(
                    z, distance, primary_response, medoids, names
                )
            structured_sets[budget] = selected
            selector_diagnostics.append(
                {
                    "Budget": budget,
                    "Sampling": selected_sampling,
                    "P2_total_distance": float(distance[:, medoids].min(axis=1).sum()),
                    "Model_shape_reconstruction_RMSE": reconstruction,
                    "RBF_length_scale": length_scale,
                }
            )

    n_realizations = 504 if selected_sampling == "Random" else 1
    random_sets = {}
    if selected_sampling == "Random":
        rng = np.random.default_rng(RANDOM_SELECTION_SEED)
        for realization in range(n_realizations):
            permutation = rng.permutation(len(sow))
            for budget in BUDGETS:
                random_sets[(realization, budget)] = permutation[:budget].tolist()

    selection_rows = []
    for realization in range(n_realizations):
        for budget in BUDGETS:
            selected = (
                random_sets[(realization, budget)]
                if selected_sampling == "Random"
                else structured_sets[budget]
            )
            for rank, index in enumerate(selected, 1):
                selection_rows.append(
                    {
                        "Realization": realization,
                        "Budget": budget,
                        "Rank": rank,
                        "Record_ID": names[index],
                        "Study_ID": sow.loc[index, "Study_ID"],
                    }
                )
    pd.DataFrame(selection_rows).to_csv(args.output_dir / "SOW80_SELECTION_LEDGER.csv", index=False)
    pd.DataFrame(selector_diagnostics).to_csv(
        args.output_dir / "SOW80_SELECTOR_DIAGNOSTICS.csv", index=False
    )

    reference_sd = np.array([sow[TARGETS[e]].std(ddof=1) for e in ENDPOINTS], dtype=float)
    observed = np.column_stack([sow[TARGETS[e]].to_numpy(float) for e in ENDPOINTS])
    domains = sow.Study_ID.astype(str).to_numpy()
    all_domains = sorted(set(domains))
    prior = {
        architecture: np.column_stack(
            [sow[f"{architecture}_{endpoint}_pred"].to_numpy(float) for endpoint in ENDPOINTS]
        )
        for architecture in ARCHITECTURES
    }
    hyper = oof_hyper(primary_oof, tabm_oof)

    history_imputer = SimpleImputer(strategy="median").fit(historical[features].to_numpy(float))
    historical_x = history_imputer.transform(historical[features].to_numpy(float))
    sow_x = history_imputer.transform(sow[features].to_numpy(float))
    historical_y = {e: historical[TARGETS[e]].to_numpy(float) for e in ENDPOINTS}
    selected_config = {e: model_spec["selected"][e]["config"] for e in ENDPOINTS}

    metrics_rows = []
    prediction_rows = []
    for realization in range(n_realizations):
        for budget in BUDGETS:
            selected = (
                random_sets[(realization, budget)]
                if selected_sampling == "Random"
                else structured_sets[budget]
            )
            selected_set = set(selected)
            evaluation = [index for index in range(len(sow)) if index not in selected_set]
            common = {
                "Realization": realization,
                "Sampling": selected_sampling,
                "Budget": budget,
                "n_calibration_records": len(selected),
                "n_evaluation_records": len(evaluation),
            }
            if selected_updater == "TargetWeightedFullRF":
                full_prediction = np.empty((len(evaluation), 2), dtype=float)
                for endpoint_index, endpoint in enumerate(ENDPOINTS):
                    combined_x = np.vstack([historical_x, sow_x[selected]])
                    combined_y = np.concatenate([historical_y[endpoint], observed[selected, endpoint_index]])
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
                    full_prediction[:, endpoint_index] = model.predict(sow_x[evaluation])
                formal = {"HistoryRF+target": full_prediction}
            else:
                formal = {
                    architecture: update_prior(
                        selected_updater,
                        prior[architecture],
                        observed,
                        selected,
                        evaluation,
                        domains,
                        all_domains,
                        hyper[architecture],
                    )
                    for architecture in ARCHITECTURES
                }

            for architecture, prediction in formal.items():
                metrics_rows.append(
                    {
                        **common,
                        "Role": "selected_updater",
                        "Architecture": architecture,
                        "Method": selected_updater,
                        **metric_values(observed[evaluation], prediction, reference_sd),
                    }
                )
                for output_index, row_index in enumerate(evaluation):
                    prediction_rows.append(
                        {
                            **common,
                            "Role": "selected_updater",
                            "Architecture": architecture,
                            "Method": selected_updater,
                            "Record_ID": sow.loc[row_index, "Record_ID"],
                            "Study_ID": sow.loc[row_index, "Study_ID"],
                            "FN_observed": observed[row_index, 0],
                            "UN_observed": observed[row_index, 1],
                            "FN_predicted": prediction[output_index, 0],
                            "UN_predicted": prediction[output_index, 1],
                        }
                    )

            for architecture in ARCHITECTURES:
                zero = prior[architecture][evaluation]
                metrics_rows.append(
                    {
                        **common,
                        "Role": "matched_no_update",
                        "Architecture": architecture,
                        "Method": "NoUpdate",
                        **metric_values(observed[evaluation], zero, reference_sd),
                    }
                )

            local_prediction = {
                "TargetOnlyRidge": np.empty((len(evaluation), 2), dtype=float),
                "TargetOnlyElasticNet": np.empty((len(evaluation), 2), dtype=float),
                "TargetOnlyRF": np.empty((len(evaluation), 2), dtype=float),
            }
            for endpoint_index, endpoint in enumerate(ENDPOINTS):
                x_cal = sow_x[selected]
                y_cal = observed[selected, endpoint_index]
                scaler = StandardScaler().fit(x_cal)
                x_cal_scaled = scaler.transform(x_cal)
                x_eval_scaled = scaler.transform(sow_x[evaluation])
                ridge = Ridge(alpha=1.0).fit(x_cal_scaled, y_cal)
                elastic = ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5,
                    max_iter=100000,
                    random_state=LOCAL_RF_SEED,
                ).fit(x_cal_scaled, y_cal)
                local_rf = RandomForestRegressor(
                    n_estimators=LOCAL_RF_TREES,
                    min_samples_leaf=1,
                    max_features=1.0,
                    random_state=LOCAL_RF_SEED + endpoint_index * 100,
                    n_jobs=1,
                ).fit(x_cal, y_cal)
                local_prediction["TargetOnlyRidge"][:, endpoint_index] = ridge.predict(x_eval_scaled)
                local_prediction["TargetOnlyElasticNet"][:, endpoint_index] = elastic.predict(x_eval_scaled)
                local_prediction["TargetOnlyRF"][:, endpoint_index] = local_rf.predict(sow_x[evaluation])
            for method, prediction in local_prediction.items():
                metrics_rows.append(
                    {
                        **common,
                        "Role": "local_only_boundary",
                        "Architecture": "local_only",
                        "Method": method,
                        **metric_values(observed[evaluation], prediction, reference_sd),
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "SOW80_SELECTED_PROCEDURE_METRICS.csv", index=False)
    predictions.to_csv(
        args.output_dir / "SOW80_SELECTED_PROCEDURE_PREDICTIONS.csv.gz",
        index=False,
        compression="gzip",
    )
    summary = (
        metrics.groupby(["Role", "Architecture", "Method", "Budget"])
        .agg(
            n_realizations=("Realization", "size"),
            Primary_mean=("Primary_NRMSE", "mean"),
            Primary_SD=("Primary_NRMSE", "std"),
            Primary_q025=("Primary_NRMSE", lambda x: x.quantile(0.025)),
            Primary_median=("Primary_NRMSE", "median"),
            Primary_q975=("Primary_NRMSE", lambda x: x.quantile(0.975)),
            FN_RMSE_mean=("FN_RMSE", "mean"),
            FN_R2_mean=("FN_R2", "mean"),
            UN_RMSE_mean=("UN_RMSE", "mean"),
            UN_R2_mean=("UN_R2", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "SOW80_SELECTED_PROCEDURE_SUMMARY.csv", index=False)

    aubc_rows = []
    for keys, group in metrics.groupby(["Role", "Architecture", "Method", "Realization"]):
        ordered = group.sort_values("Budget")
        if tuple(ordered.Budget) != tuple(BUDGETS):
            raise AssertionError(f"Incomplete SOW budget curve: {keys}")
        aubc_rows.append(
            {
                "Role": keys[0],
                "Architecture": keys[1],
                "Method": keys[2],
                "Realization": keys[3],
                "AUBC_3_12": float(np.trapezoid(ordered.Primary_NRMSE, ordered.Budget) / 9),
            }
        )
    aubc = pd.DataFrame(aubc_rows)
    aubc.to_csv(args.output_dir / "SOW80_AUBC_BY_REALIZATION.csv", index=False)
    aubc_summary = (
        aubc.groupby(["Role", "Architecture", "Method"])
        .agg(
            n_realizations=("Realization", "size"),
            AUBC_mean=("AUBC_3_12", "mean"),
            AUBC_SD=("AUBC_3_12", "std"),
            AUBC_q025=("AUBC_3_12", lambda x: x.quantile(0.025)),
            AUBC_median=("AUBC_3_12", "median"),
            AUBC_q975=("AUBC_3_12", lambda x: x.quantile(0.975)),
        )
        .reset_index()
        .sort_values(["Role", "AUBC_mean", "Architecture", "Method"])
    )
    aubc_summary.to_csv(args.output_dir / "SOW80_AUBC_SUMMARY.csv", index=False)

    metadata = {
        "status": "COMPLETE_STRESS_TEST_ONLY",
        "dataset": "SOW80",
        "data_note": "Renteria2008 mapping restored from the study record; 23 structurally missing starch values were filled outcome-blind using the 906-record training median",
        "selected_from_546": {
            "sampling": selected_sampling,
            "updater": selected_updater,
            "target_spec_sha256": sha256(args.target_spec),
        },
        "rows": int(len(sow)),
        "studies": int(sow.Study_ID.astype(str).nunique()),
        "budgets": list(BUDGETS),
        "realizations": n_realizations,
        "architectures": list(ARCHITECTURES),
        "selection_unit": "SOW treatment-stage mean Record_ID",
        "selected_record_exit": True,
        "selector_features_available": selector_features,
        "selector_features_dropped_in_SOW": sorted(set(selector_candidates) - set(selector_features)),
        "response_surface": "all-906 MLR+RF predictions generated before SOW outcome use",
        "HPP_domain": "Study_ID",
        "architecture_hyper_source": "673-record Study_ID-grouped OOF estimates from model development",
        "metric_denominator": {
            "FN_full_SOW80_sample_SD": float(reference_sd[0]),
            "UN_full_SOW80_sample_SD": float(reference_sd[1]),
        },
        "local_only_update": target_spec["local_only"],
        "outcomes_used_to_reselect_sampling_or_updater": False,
        "inputs": {
            "matrix_906_sha256": sha256(args.matrix_906),
            "sow_predictions_sha256": sha256(args.sow_predictions),
            "primary_oof_sha256": sha256(args.primary_oof),
            "tabm_oof_sha256": sha256(args.tabm_oof),
            "model_spec_sha256": sha256(args.model_spec),
        },
        "output_hashes": {},
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "sow80_run_metadata.json":
            metadata["output_hashes"][path.name] = sha256(path)
    (args.output_dir / "sow80_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        aubc_summary.loc[
            aubc_summary.Role.eq("selected_updater"),
            ["Architecture", "Method", "AUBC_mean", "AUBC_q025", "AUBC_q975"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
