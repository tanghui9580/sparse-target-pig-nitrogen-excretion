#!/usr/bin/env python3
"""84-block paired bootstrap for the 546 factorial AUBC surface.

Six pig-record rotations share a permutation block and are not treated as six
independent experiments.  The bootstrap resamples 84 whole blocks and keeps
all sampling/updater cells paired. The intervals are descriptive because the same
546-record target dataset is also used to evaluate the selected measurement-and-updating procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_BOOT = 10_000
SEED = 20260815


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aubc", type=Path, required=True)
    parser.add_argument("--target-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aubc = pd.read_csv(args.aubc)
    spec = json.loads(args.target_spec.read_text(encoding="utf-8"))
    formal = aubc.loc[aubc.Role.eq("formal_updater")].copy()
    if len(formal) != 504 * 3 * 4:
        raise AssertionError("Expected 504 realizations x 3 sampling x 4 formal updaters")
    formal["Block"] = formal.Realization.astype(int) // 6
    if formal.Block.nunique() != 84:
        raise AssertionError("Expected 84 independent permutation blocks")
    block = (
        formal.groupby(["Block", "Sampling", "Method"], as_index=False)
        .AUBC_3_12.mean()
    )
    cells = sorted(
        {(row.Sampling, row.Method) for row in block.itertuples()},
        key=lambda item: (item[0], item[1]),
    )
    blocks = sorted(block.Block.unique())
    matrix = np.empty((len(blocks), len(cells)), dtype=float)
    for block_index, block_id in enumerate(blocks):
        rows = block.loc[block.Block.eq(block_id)].set_index(["Sampling", "Method"])
        for cell_index, cell in enumerate(cells):
            matrix[block_index, cell_index] = float(rows.loc[cell, "AUBC_3_12"])
    rng = np.random.default_rng(SEED)
    replicate_means = np.empty((N_BOOT, len(cells)), dtype=float)
    for replicate in range(N_BOOT):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        replicate_means[replicate] = matrix[selected].mean(axis=0)

    point = matrix.mean(axis=0)
    selected_cell = (spec["selected_sampling"], spec["selected_updater"])
    winner_index = cells.index(selected_cell)
    winner_frequency = np.mean(
        replicate_means == replicate_means.min(axis=1, keepdims=True), axis=0
    )
    summary_rows = []
    for index, (sampling, method) in enumerate(cells):
        values = replicate_means[:, index]
        delta = values - replicate_means[:, winner_index]
        summary_rows.append(
            {
                "Sampling": sampling,
                "Method": method,
                "Point_AUBC": float(point[index]),
                "Bootstrap_mean": float(values.mean()),
                "CI_2.5%": float(np.quantile(values, 0.025)),
                "CI_50%": float(np.quantile(values, 0.5)),
                "CI_97.5%": float(np.quantile(values, 0.975)),
                "Winner_frequency": float(winner_frequency[index]),
                "Delta_vs_selected_point": float(point[index] - point[winner_index]),
                "Delta_vs_selected_CI_2.5%": float(np.quantile(delta, 0.025)),
                "Delta_vs_selected_CI_97.5%": float(np.quantile(delta, 0.975)),
                "P_cell_better_than_selected": float(np.mean(delta < 0)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["Point_AUBC", "Sampling", "Method"])
    summary_path = args.output_dir / "FACTORIAL_84BLOCK_BOOTSTRAP_SUMMARY.csv"
    summary.to_csv(summary_path, index=False)

    pair_rows = []
    for left_index, right_index in itertools.combinations(range(len(cells)), 2):
        left = cells[left_index]
        right = cells[right_index]
        values = replicate_means[:, left_index] - replicate_means[:, right_index]
        pair_rows.append(
            {
                "Left_sampling": left[0],
                "Left_method": left[1],
                "Right_sampling": right[0],
                "Right_method": right[1],
                "Point_delta_left_minus_right": float(point[left_index] - point[right_index]),
                "CI_2.5%": float(np.quantile(values, 0.025)),
                "CI_50%": float(np.quantile(values, 0.5)),
                "CI_97.5%": float(np.quantile(values, 0.975)),
                "P_left_better": float(np.mean(values < 0)),
            }
        )
    pair_path = args.output_dir / "FACTORIAL_84BLOCK_PAIRED_COMPARISONS.csv"
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)

    replicate_frame = pd.DataFrame(
        replicate_means,
        columns=[f"{sampling}__{method}" for sampling, method in cells],
    )
    replicate_frame.insert(0, "Replicate", np.arange(N_BOOT))
    replicate_path = args.output_dir / "FACTORIAL_84BLOCK_BOOTSTRAP_10000.csv.gz"
    replicate_frame.to_csv(replicate_path, index=False, compression="gzip")

    metadata = {
        "status": "COMPLETE_DESCRIPTIVE_CONDITIONAL_ON_546_SELECTION",
        "n_blocks": 84,
        "rotations_per_block": 6,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "resampling": "84 permutation blocks with replacement; all 12 formal cells paired",
        "selected_cell": {"sampling": selected_cell[0], "method": selected_cell[1]},
        "warning": "The selected cell was identified using the same 546-record target dataset; these intervals are descriptive and are not an independent confirmation.",
        "inputs": {
            "aubc_sha256": sha256(args.aubc),
            "target_spec_sha256": sha256(args.target_spec),
        },
        "outputs": {
            summary_path.name: sha256(summary_path),
            pair_path.name: sha256(pair_path),
            replicate_path.name: sha256(replicate_path),
        },
    }
    (args.output_dir / "FACTORIAL_84BLOCK_BOOTSTRAP_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
