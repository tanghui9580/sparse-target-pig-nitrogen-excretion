#!/usr/bin/env python3
"""Execute the reported MLR+RF development, confirmation, and refit route.

The analysis sequence is:
1. use only the 673 development rows for 24-grid RF HPO;
2. select the RF configuration endpoint-wise by pooled Study-GroupKFold OOF RMSE;
3. regenerate selected-head OOF predictions with three prespecified seeds and estimate clipped alpha;
4. fit the prespecified MLR, RF, and MLR+RF on all 673 and evaluate the 233 confirmation set once;
5. refit the unchanged architecture on all 906 records and predict the independent 546-record target matrix.

No 233 or 546 outcome participates in HPO, alpha estimation, or model identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


N_ESTIMATORS = 200
TARGETS = {"FN": "Analysis_Fecal_N_g_d", "UN": "Analysis_Urinary_N_g_d"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_mlr(x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x, y)
    return model


def fit_rf(x: np.ndarray, y: np.ndarray, config: dict, seed: int) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=int(config["min_samples_leaf"]),
                    max_features=float(config["max_features"]),
                    max_depth=config.get("max_depth"),
                    random_state=int(seed),
                    n_jobs=4,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def grouped_oof_rf(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    config: dict,
    seeds: list[int],
) -> np.ndarray:
    x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = frame[target].to_numpy(float)
    groups = frame["Study_ID"].astype(str).to_numpy()
    prediction_by_seed = []
    for seed in seeds:
        oof = np.full(len(frame), np.nan)
        for train_index, test_index in GroupKFold(n_splits=5).split(x, y, groups):
            model = fit_rf(x[train_index], y[train_index], config, seed)
            oof[test_index] = model.predict(x[test_index])
        if np.isnan(oof).any():
            raise AssertionError("RF OOF prediction contains missing values")
        prediction_by_seed.append(oof)
    return np.mean(np.stack(prediction_by_seed), axis=0)


def grouped_oof_mlr(frame: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = frame[target].to_numpy(float)
    groups = frame["Study_ID"].astype(str).to_numpy()
    oof = np.full(len(frame), np.nan)
    for train_index, test_index in GroupKFold(n_splits=5).split(x, y, groups):
        model = fit_mlr(x[train_index], y[train_index])
        oof[test_index] = model.predict(x[test_index])
    if np.isnan(oof).any():
        raise AssertionError("MLR OOF prediction contains missing values")
    return oof


def clipped_alpha(y: np.ndarray, base: np.ndarray, head: np.ndarray) -> float:
    delta = head - base
    denominator = float(np.dot(delta, delta))
    if denominator <= 0:
        return 0.0
    return float(np.clip(np.dot(y - base, delta) / denominator, 0.0, 1.0))


def endpoint_stats(y: np.ndarray, prediction: np.ndarray) -> dict:
    residual = prediction - y
    return {
        "n": int(len(y)),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "Bias_pred_minus_obs": float(np.mean(residual)),
        "R2": float(r2_score(y, prediction)),
        "Observed_sample_SD": float(np.std(y, ddof=1)),
    }


def metrics_table(
    frame: pd.DataFrame,
    models: list[str],
    scope: str,
    grain: str,
    reference_sd: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for model in models:
        by_endpoint = {}
        for endpoint, target in TARGETS.items():
            stats = endpoint_stats(
                frame[target].to_numpy(float), frame[f"{model}_{endpoint}_pred"].to_numpy(float)
            )
            by_endpoint[endpoint] = stats
            rows.append(
                {
                    "Scope": scope,
                    "Grain": grain,
                    "Model": model,
                    "Endpoint": endpoint,
                    **stats,
                    "NRMSE_target_sample_SD": stats["RMSE"] / stats["Observed_sample_SD"],
                    "NRMSE_673_reference_SD": stats["RMSE"] / reference_sd[endpoint],
                    "Primary_target_sample_SD": np.nan,
                    "Primary_673_reference_SD": np.nan,
                }
            )
        rows.append(
            {
                "Scope": scope,
                "Grain": grain,
                "Model": model,
                "Endpoint": "PRIMARY",
                "n": int(len(frame)),
                "RMSE": np.nan,
                "MAE": np.nan,
                "Bias_pred_minus_obs": np.nan,
                "R2": np.nan,
                "Observed_sample_SD": np.nan,
                "NRMSE_target_sample_SD": np.nan,
                "NRMSE_673_reference_SD": np.nan,
                "Primary_target_sample_SD": float(
                    np.mean(
                        [
                            by_endpoint[e]["RMSE"] / by_endpoint[e]["Observed_sample_SD"]
                            for e in ("FN", "UN")
                        ]
                    )
                ),
                "Primary_673_reference_SD": float(
                    np.mean([by_endpoint[e]["RMSE"] / reference_sd[e] for e in ("FN", "UN")])
                ),
            }
        )
    return pd.DataFrame(rows)


def source_mean(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = list(TARGETS.values()) + [
        f"{model}_{endpoint}_pred"
        for model in ("MLR", "RF", "MLR+RF")
        for endpoint in TARGETS
    ]
    keep = ["Source_unit", "Ingredient_Class"]
    return frame.groupby(keep, as_index=False)[numeric].mean()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-906", type=Path, required=True)
    parser.add_argument("--matrix-546", type=Path, required=True)
    parser.add_argument("--model-selection-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(args.model_selection_spec.read_text(encoding="utf-8"))
    matrix_hash = sha256(args.matrix_906)
    if matrix_hash != spec["data_sha256"]:
        raise AssertionError(f"906 matrix hash mismatch: {matrix_hash} != {spec['data_sha256']}")

    data = pd.read_csv(args.matrix_906)
    target = pd.read_csv(args.matrix_546)
    development = data.loc[data["Analysis_Split"].eq("development")].reset_index(drop=True)
    confirmation = data.loc[data["Analysis_Split"].eq("confirmation")].reset_index(drop=True)
    if (len(development), len(confirmation), len(target)) != (673, 233, 546):
        raise AssertionError("Expected 673/233/546 rows")
    if development["Study_ID"].isin(confirmation["Study_ID"]).any():
        raise AssertionError("Development and confirmation Study_ID overlap")
    if target["Source_unit"].nunique() != 91 or not target.groupby("Source_unit").size().eq(6).all():
        raise AssertionError("546 source structure must be 91 x 6")

    features = list(spec["features"])
    missing_features = [name for name in features if name not in target.columns]
    if missing_features:
        raise AssertionError(f"546 matrix lacks required analysis features: {missing_features}")
    tuning_seed = int(spec["hpo_tuning_seed"])
    final_seeds = [int(seed) for seed in spec["final_model_seeds"]]
    rf_grid = list(spec["grids"]["RF"])
    if len(rf_grid) != 24:
        raise AssertionError("RF grid must contain 24 prespecified configurations")

    selected = {}
    hpo_rows = []
    oof_frame = development[["Record_ID", "Study_ID", *TARGETS.values()]].copy()
    for endpoint, target_column in TARGETS.items():
        y = development[target_column].to_numpy(float)
        best = None
        for index, config in enumerate(rf_grid):
            prediction = grouped_oof_rf(development, features, target_column, config, [tuning_seed])
            rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
            row = {
                "Endpoint": endpoint,
                "Config_index": index,
                "min_samples_leaf": config["min_samples_leaf"],
                "max_features": config["max_features"],
                "max_depth": config.get("max_depth"),
                "Tuning_seed": tuning_seed,
                "n_estimators": N_ESTIMATORS,
                "OOF_RMSE": rmse,
            }
            hpo_rows.append(row)
            if best is None or (rmse, index) < (best["OOF_RMSE"], best["Config_index"]):
                best = {**row, "config": config}
        assert best is not None
        mlr_oof = grouped_oof_mlr(development, features, target_column)
        rf_oof = grouped_oof_rf(development, features, target_column, best["config"], final_seeds)
        alpha = clipped_alpha(y, mlr_oof, rf_oof)
        hybrid_oof = mlr_oof + alpha * (rf_oof - mlr_oof)
        selected[endpoint] = {
            "config_index": int(best["Config_index"]),
            "config": best["config"],
            "tuning_seed_oof_rmse": float(best["OOF_RMSE"]),
            "alpha": alpha,
            "three_seed_head_oof_rmse": float(np.sqrt(np.mean((rf_oof - y) ** 2))),
            "hybrid_oof_rmse": float(np.sqrt(np.mean((hybrid_oof - y) ** 2))),
        }
        oof_frame[f"MLR_{endpoint}_pred"] = mlr_oof
        oof_frame[f"RF_{endpoint}_pred"] = rf_oof
        oof_frame[f"MLR+RF_{endpoint}_pred"] = hybrid_oof
        print(f"{endpoint}: config={best['Config_index']} alpha={alpha:.10f}", flush=True)

    pd.DataFrame(hpo_rows).to_csv(output_dir / "RF_24GRID_673_HPO_LEDGER.csv", index=False)
    oof_frame.to_csv(output_dir / "mlr_rf_673_oof.csv", index=False)

    ref_sd = {e: float(development[t].std(ddof=1)) for e, t in TARGETS.items()}
    oof_metrics = metrics_table(
        oof_frame, ["MLR", "RF", "MLR+RF"], "673_refit_OOF", "record", ref_sd
    )
    oof_metrics.to_csv(output_dir / "MLR_RF_673_REFIT_OOF_METRICS.csv", index=False)

    x_dev = development[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x_val = confirmation[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    predictions_233 = confirmation[["Record_ID", "Study_ID", *TARGETS.values()]].copy()
    model_673_metadata = {}
    for endpoint, target_column in TARGETS.items():
        y_dev = development[target_column].to_numpy(float)
        mlr = fit_mlr(x_dev, y_dev)
        rf_models = [fit_rf(x_dev, y_dev, selected[endpoint]["config"], seed) for seed in final_seeds]
        mlr_prediction = mlr.predict(x_val)
        rf_prediction = np.mean(np.stack([model.predict(x_val) for model in rf_models]), axis=0)
        hybrid_prediction = mlr_prediction + selected[endpoint]["alpha"] * (rf_prediction - mlr_prediction)
        predictions_233[f"MLR_{endpoint}_pred"] = mlr_prediction
        predictions_233[f"RF_{endpoint}_pred"] = rf_prediction
        predictions_233[f"MLR+RF_{endpoint}_pred"] = hybrid_prediction
        mlr_path = model_dir / f"673_MLR_{endpoint}.joblib"
        rf_path = model_dir / f"673_RF_{endpoint}_3seed.joblib"
        joblib.dump(mlr, mlr_path)
        joblib.dump(rf_models, rf_path)
        model_673_metadata[endpoint] = {
            "MLR_model": mlr_path.name,
            "MLR_sha256": sha256(mlr_path),
            "RF_model": rf_path.name,
            "RF_sha256": sha256(rf_path),
        }
    predictions_233.to_csv(output_dir / "mlr_rf_confirmation233_predictions.csv", index=False)
    metrics_233 = metrics_table(
        predictions_233, ["MLR", "RF", "MLR+RF"], "confirmation_233", "record", ref_sd
    )
    metrics_233.to_csv(output_dir / "CONFIRMATION_233_MLR_RF_METRICS.csv", index=False)

    x_all = data[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x_546 = target[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    predictions_546 = target.copy()
    model_906_metadata = {}
    for endpoint, target_column in TARGETS.items():
        y_all = data[target_column].to_numpy(float)
        mlr = fit_mlr(x_all, y_all)
        rf_models = [fit_rf(x_all, y_all, selected[endpoint]["config"], seed) for seed in final_seeds]
        mlr_prediction = mlr.predict(x_546)
        rf_prediction = np.mean(np.stack([model.predict(x_546) for model in rf_models]), axis=0)
        hybrid_prediction = mlr_prediction + selected[endpoint]["alpha"] * (rf_prediction - mlr_prediction)
        predictions_546[f"MLR_{endpoint}_pred"] = mlr_prediction
        predictions_546[f"RF_{endpoint}_pred"] = rf_prediction
        predictions_546[f"MLR+RF_{endpoint}_pred"] = hybrid_prediction
        mlr_path = model_dir / f"ALL906_MLR_{endpoint}.joblib"
        rf_path = model_dir / f"ALL906_RF_{endpoint}_3seed.joblib"
        joblib.dump(mlr, mlr_path)
        joblib.dump(rf_models, rf_path)
        model_906_metadata[endpoint] = {
            "MLR_model": mlr_path.name,
            "MLR_sha256": sha256(mlr_path),
            "RF_model": rf_path.name,
            "RF_sha256": sha256(rf_path),
        }

    predictions_546.to_csv(output_dir / "ALL906_TO_546_MLR_RF_ZERO_SHOT_PREDICTIONS.csv", index=False)
    source_predictions_546 = source_mean(predictions_546)
    source_predictions_546.to_csv(
        output_dir / "ALL906_TO_546_MLR_RF_ZERO_SHOT_SOURCE_MEANS.csv", index=False
    )
    metrics_546_record = metrics_table(
        predictions_546, ["MLR", "RF", "MLR+RF"], "all906_to_546_zero_shot", "animal", ref_sd
    )
    metrics_546_source = metrics_table(
        source_predictions_546,
        ["MLR", "RF", "MLR+RF"],
        "all906_to_546_zero_shot",
        "source_mean",
        ref_sd,
    )
    metrics_546 = pd.concat([metrics_546_record, metrics_546_source], ignore_index=True)
    metrics_546.to_csv(output_dir / "ALL906_TO_546_MLR_RF_ZERO_SHOT_METRICS.csv", index=False)

    metadata = {
        "status": "COMPLETE",
        "analysis_role": "selected MLR+RF route after model-development evaluation",
        "selection_score_reference": 0.5230654982536393,
        "matrix_906": str(args.matrix_906),
        "matrix_906_sha256": matrix_hash,
        "matrix_546": str(args.matrix_546),
        "matrix_546_sha256": sha256(args.matrix_546),
        "model_selection_specification": str(args.model_selection_spec),
        "model_selection_specification_sha256": sha256(args.model_selection_spec),
        "rows": {"development": 673, "confirmation": 233, "target_546": 546},
        "groups": {
            "development_studies": int(development["Study_ID"].nunique()),
            "confirmation_studies": int(confirmation["Study_ID"].nunique()),
            "target_sources": int(target["Source_unit"].nunique()),
        },
        "features": features,
        "inactive_feature": spec["inactive_P2"],
        "n_estimators": N_ESTIMATORS,
        "n_estimators_note": "The model-selection specification recorded 24 structural RF configurations without an explicit tree count; 200 trees matches the implementation used for the reported analysis and is recorded here explicitly.",
        "tuning_seed": tuning_seed,
        "final_seeds": final_seeds,
        "selected": selected,
        "reference_sd_673": ref_sd,
        "model_673": model_673_metadata,
        "model_906": model_906_metadata,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "leakage_guards": {
            "233_used_for_selection": False,
            "546_outcomes_used_for_selection": False,
            "development_confirmation_study_overlap": 0,
            "selected_sources_exit_later_evaluation": "enforced in the target-updating analysis; not applicable to direct transfer",
        },
    }
    metadata_path = output_dir / "primary_model_specification.json"
    save_json(metadata_path, metadata)
    metadata["output_files"] = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path != metadata_path
    }
    save_json(metadata_path, metadata)
    print(f"PRIMARY_ROUTE_COMPLETE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
