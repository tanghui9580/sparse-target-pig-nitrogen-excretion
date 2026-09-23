#!/usr/bin/env python3
"""Paired cluster bootstrap for the 233-record confirmation set, 546 target source means, and SOW80.

The bootstrap is descriptive and cannot alter the nested-S17 model choice.
Clusters are Study_ID for 233/SOW and Source_unit for 546.  Every replicate
uses the same resampled rows for all architectures, so model deltas are paired.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


MODELS = ("MLR", "RF", "TabM", "MLR+RF")
ENDPOINTS = ("FN", "UN")
TARGETS = {"FN": "Analysis_Fecal_N_g_d", "UN": "Analysis_Urinary_N_g_d"}
N_BOOT = 10_000
SEED = 20260815


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(
    frame: pd.DataFrame,
    model: str,
    reference_sd: dict[str, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    nrmse_fixed = []
    nrmse_resampled = []
    for endpoint in ENDPOINTS:
        y = frame[TARGETS[endpoint]].to_numpy(float)
        p = frame[f"{model}_{endpoint}_pred"].to_numpy(float)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        sd = float(np.std(y, ddof=1))
        result[f"{endpoint}_RMSE"] = rmse
        result[f"{endpoint}_R2"] = float(r2_score(y, p))
        result[f"{endpoint}_Bias"] = float(np.mean(p - y))
        nrmse_fixed.append(rmse / reference_sd[endpoint])
        nrmse_resampled.append(rmse / sd)
    result["Primary_NRMSE"] = float(np.mean(nrmse_fixed))
    result["Primary_NRMSE_resampledSD"] = float(np.mean(nrmse_resampled))
    return result


def bootstrap_scope(
    frame: pd.DataFrame,
    group_column: str,
    scope: str,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = sorted(frame[group_column].astype(str).unique())
    row_lookup = {
        group: np.where(frame[group_column].astype(str).to_numpy() == group)[0] for group in groups
    }
    rng = np.random.default_rng(SEED + seed_offset)
    reference_sd = {
        endpoint: float(frame[TARGETS[endpoint]].std(ddof=1)) for endpoint in ENDPOINTS
    }
    point_rows = []
    for model in MODELS:
        point_rows.append(
            {"Scope": scope, "Model": model, **stats(frame, model, reference_sd)}
        )
    point = pd.DataFrame(point_rows)

    replicate_rows = []
    for replicate in range(N_BOOT):
        selected = rng.choice(groups, size=len(groups), replace=True)
        positions = np.concatenate([row_lookup[group] for group in selected])
        sampled = frame.iloc[positions]
        primary = {}
        row = {"Scope": scope, "Replicate": replicate}
        for model in MODELS:
            values = stats(sampled, model, reference_sd)
            primary[model] = values["Primary_NRMSE"]
            for metric, value in values.items():
                row[f"{model}__{metric}"] = value
        best = min(primary, key=lambda name: (primary[name], name))
        row["Best_model"] = best
        replicate_rows.append(row)
    replicates = pd.DataFrame(replicate_rows)

    summary_rows = []
    for model in MODELS:
        for metric in (
            "Primary_NRMSE",
            "Primary_NRMSE_resampledSD",
            "FN_RMSE",
            "FN_R2",
            "FN_Bias",
            "UN_RMSE",
            "UN_R2",
            "UN_Bias",
        ):
            values = replicates[f"{model}__{metric}"]
            point_value = float(point.loc[point.Model.eq(model), metric].iloc[0])
            summary_rows.append(
                {
                    "Scope": scope,
                    "Type": "model_metric",
                    "Model_or_comparison": model,
                    "Metric": metric,
                    "Point": point_value,
                    "Bootstrap_mean": float(values.mean()),
                    "CI_2.5%": float(values.quantile(0.025)),
                    "CI_50%": float(values.quantile(0.5)),
                    "CI_97.5%": float(values.quantile(0.975)),
                    "P_lt_0": np.nan,
                    "Best_frequency": float((replicates.Best_model == model).mean())
                    if metric == "Primary_NRMSE"
                    else np.nan,
                }
            )
    for left, right in itertools.combinations(MODELS, 2):
        values = replicates[f"{left}__Primary_NRMSE"] - replicates[f"{right}__Primary_NRMSE"]
        point_delta = float(
            point.loc[point.Model.eq(left), "Primary_NRMSE"].iloc[0]
            - point.loc[point.Model.eq(right), "Primary_NRMSE"].iloc[0]
        )
        summary_rows.append(
            {
                "Scope": scope,
                "Type": "paired_primary_delta",
                "Model_or_comparison": f"{left} minus {right}",
                "Metric": "Delta_Primary_NRMSE",
                "Point": point_delta,
                "Bootstrap_mean": float(values.mean()),
                "CI_2.5%": float(values.quantile(0.025)),
                "CI_50%": float(values.quantile(0.5)),
                "CI_97.5%": float(values.quantile(0.975)),
                "P_lt_0": float((values < 0).mean()),
                "Best_frequency": np.nan,
            }
        )
    return point, replicates, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-233", type=Path, required=True)
    parser.add_argument("--predictions-546-source", type=Path, required=True)
    parser.add_argument("--predictions-sow80", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specifications = [
        (args.predictions_233, "Study_ID", "confirmation_233_record", 0),
        (args.predictions_546_source, "Source_unit", "546_source_mean_zero_shot", 1000),
        (args.predictions_sow80, "Study_ID", "SOW80_treatment_stage_mean_zero_shot", 2000),
    ]
    all_point = []
    all_replicates = []
    all_summary = []
    inputs = {}
    for path, group, scope, seed_offset in specifications:
        frame = pd.read_csv(path)
        required = [TARGETS[e] for e in ENDPOINTS]
        required += [f"{m}_{e}_pred" for m in MODELS for e in ENDPOINTS]
        missing = [column for column in required if column not in frame]
        if missing:
            raise AssertionError(f"{scope} missing columns: {missing}")
        if frame[group].isna().any():
            raise AssertionError(f"{scope} has missing cluster IDs")
        point, replicates, summary = bootstrap_scope(frame, group, scope, seed_offset)
        all_point.append(point)
        all_replicates.append(replicates)
        all_summary.append(summary)
        inputs[scope] = {
            "file": str(path),
            "sha256": sha256(path),
            "rows": int(len(frame)),
            "cluster_column": group,
            "clusters": int(frame[group].astype(str).nunique()),
        }

    point = pd.concat(all_point, ignore_index=True)
    replicates = pd.concat(all_replicates, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    point_path = args.output_dir / "EXTERNAL_POINT_METRICS.csv"
    replicate_path = args.output_dir / "EXTERNAL_CLUSTER_BOOTSTRAP_10000.csv.gz"
    summary_path = args.output_dir / "EXTERNAL_CLUSTER_BOOTSTRAP_SUMMARY.csv"
    point.to_csv(point_path, index=False)
    replicates.to_csv(replicate_path, index=False, compression="gzip")
    summary.to_csv(summary_path, index=False)

    metadata = {
        "status": "COMPLETE_DESCRIPTIVE_ONLY",
        "cannot_reselect_nested_S17_winner": True,
        "n_bootstrap": N_BOOT,
        "base_seed": SEED,
        "resampling": "sample whole clusters with replacement; paired across all architectures",
        "metric_denominator": {
            "Primary_NRMSE": "full-scope endpoint sample SD held fixed across bootstrap replicates",
            "Primary_NRMSE_resampledSD": "secondary sensitivity with sample SD recomputed in each replicate",
        },
        "inputs": inputs,
        "outputs": {
            point_path.name: sha256(point_path),
            replicate_path.name: sha256(replicate_path),
            summary_path.name: sha256(summary_path),
        },
    }
    (args.output_dir / "EXTERNAL_BOOTSTRAP_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        summary.loc[
            summary.Metric.eq("Primary_NRMSE"),
            ["Scope", "Model_or_comparison", "Point", "CI_2.5%", "CI_97.5%", "Best_frequency"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
