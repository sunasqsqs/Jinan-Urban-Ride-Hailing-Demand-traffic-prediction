"""Generate deterministic frontend datasets from model experiment reports.

The experiment reports are kept as immutable algorithm outputs.  This script
adds the per-grid time series needed by the realtime and dispatch views and
writes browser-friendly datasets under ``static/data``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "static" / "data"

DATASETS = {
    "short": {
        "label": "短周期预测",
        "source": ROOT / "new results" / "experiment_report.json",
        "output": OUTPUT_DIR / "short_cycle.json",
    },
    "long": {
        "label": "长周期预测",
        "source": ROOT / "new results2" / "experiment_report.json",
        "output": OUTPUT_DIR / "long_cycle.json",
    },
}

GRID_NAMES = {
    81: "济南西站片区",
    82: "泉城广场—趵突泉核心区",
    83: "高新万达—奥体中心片区",
    84: "唐冶—章丘西片区",
    86: "章丘百脉泉片区",
}


def _scaled_series(values: list[float], target_sum: float, phase: float) -> list[float]:
    """Create a spatially varied series whose rounded sum equals target_sum."""
    raw = [
        max(0.0, float(value) * (1 + 0.10 * math.sin(i * 0.37 + phase) +
                                 0.04 * math.cos(i * 0.13 + phase * 0.7)))
        for i, value in enumerate(values)
    ]
    raw_sum = sum(raw) or 1.0
    scaled = [round(value * target_sum / raw_sum, 3) for value in raw]
    if scaled:
        scaled[-1] = round(scaled[-1] + target_sum - sum(scaled), 3)
    return scaled


def _region_series(report: dict) -> list[dict]:
    prediction = report["predictions_time_series"]["prediction_mamba_gnn"]
    ground_truth = report["predictions_time_series"]["ground_truth"]
    prediction_total = sum(float(value) for value in prediction) or 1.0
    regions = []

    for index, hotspot in enumerate(report.get("hotspots", [])):
        grid_id = int(hotspot["grid_id"])
        predicted_volume = float(hotspot["predicted_volume"])
        share = predicted_volume / prediction_total
        predicted_values = _scaled_series(prediction, predicted_volume, grid_id * 0.17)

        # Ground-truth values retain the experiment's global curve while adding
        # a small deterministic local pattern.  This is display simulation data,
        # clearly marked as derived in the metadata below.
        true_target = sum(float(value) for value in ground_truth) * share
        true_values = _scaled_series(ground_truth, true_target, grid_id * 0.17 + 0.45)
        lon_range = hotspot["lon_range"]
        lat_range = hotspot["lat_range"]
        lon_center = round(sum(lon_range) / 2, 6)
        lat_center = round(sum(lat_range) / 2, 6)
        errors = [predicted - actual for predicted, actual in zip(predicted_values, true_values)]
        mae = sum(abs(error) for error in errors) / len(errors) if errors else 0
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0

        regions.append({
            "rank": int(hotspot.get("rank", index + 1)),
            "grid_id": grid_id,
            "name": GRID_NAMES.get(grid_id, f"网格 {grid_id}"),
            "lon_center": lon_center,
            "lat_center": lat_center,
            "center": {"lng": lon_center, "lat": lat_center},
            "lon_range": lon_range,
            "lat_range": lat_range,
            "predicted_volume": hotspot["predicted_volume"],
            "metrics": {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "total_true_volume": round(sum(true_values), 3),
                "total_predicted_volume": round(sum(predicted_values), 3),
            },
            "time_series_data": {
                "true_values": true_values,
                "predicted_values": predicted_values,
            },
        })

    return regions


def build_dataset(dataset_id: str, config: dict) -> dict:
    with config["source"].open("r", encoding="utf-8") as source_file:
        report = json.load(source_file)

    steps = len(report.get("predictions_time_series", {}).get("ground_truth", []))
    report["frontend_metadata"] = {
        "dataset_id": dataset_id,
        "label": config["label"],
        "source_report": str(config["source"].relative_to(ROOT)).replace("\\", "/"),
        "simulation_version": 1,
        "time_steps": steps,
        "regional_series": "deterministically derived from experiment totals",
    }
    report["top_predicted_regions_analysis"] = _region_series(report)
    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id, config in DATASETS.items():
        dataset = build_dataset(dataset_id, config)
        with config["output"].open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(dataset, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        print(f"generated {config['output'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()
