"""Run seeded random, deadband, and scheduled-setback comparisons."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.evaluation import (
    CONTROLLER_DISPLAY_NAMES,
    CONTROLLER_NAMES,
    controller_baselines,
    summarize_results,
)
from thermostat_rl.env import ThermostatEnv

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty results file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_cost_plot(results: list[dict[str, object]], seed_count: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    values = [
        [
            float(result["energy_cost_usd"])
            for result in results
            if result["controller_type"] == name
        ]
        for name in CONTROLLER_NAMES
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        values,
        tick_labels=[CONTROLLER_DISPLAY_NAMES[name] for name in CONTROLLER_NAMES],
    )
    axis.set_title(f"Baseline weekly energy cost ({seed_count} evaluation seeds)")
    axis.set_xlabel("Controller")
    axis.set_ylabel("Energy cost (USD per simulated week)")
    axis.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_plot.png", dpi=160)
    plt.close(fig)


def run(seed_start: int = 1000, seed_count: int = 30) -> None:
    if seed_start < 0 or seed_count < 2:
        raise ValueError("seed_start must be non-negative and seed_count at least 2")
    config = ExperimentConfig.from_env()
    seeds = list(range(seed_start, seed_start + seed_count))
    results = controller_baselines(ThermostatEnv(config), config, seeds)
    summaries = summarize_results(results)
    _write_csv(RESULTS_DIR / "baseline_table.csv", results)
    _write_csv(RESULTS_DIR / "baseline_summary.csv", summaries)

    _write_cost_plot(results, seed_count)
    print(f"Saved {len(results)} baseline episodes across {seed_count} seeds.")
    for row in summaries:
        print(
            f"{row['controller_type']}: "
            f"cost=${row['energy_cost_usd_mean']:.2f} "
            f"(SD ${row['energy_cost_usd_sd']:.2f}), "
            f"comfort={row['comfort_degree_hours_mean']:.1f} degree-hours, "
            f"occupied comfort={row['occupied_comfort_percentage_mean']:.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=30)
    arguments = parser.parse_args()
    run(arguments.seed_start, arguments.seed_count)


if __name__ == "__main__":
    main()
