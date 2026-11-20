"""Train Q-learning policies and run the predefined reward-weight experiment."""

import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.controllers import (
    DeadbandController,
    RandomController,
    ScheduledSetbackController,
)
from thermostat_rl.env import HVACAction, ThermostatEnv
from thermostat_rl.evaluation import (
    CONTROLLER_DISPLAY_NAMES,
    EpisodeResult,
    evaluate_controller,
    summarize_results,
)
from thermostat_rl.q_learning import train_q_learning
from thermostat_rl.state import reachable_state_upper_bound

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
COMFORT_WEIGHTS = (2.5, 5.0, 10.0)
EVALUATION_SEED_START = 2000
METRICS = (
    "energy_cost_usd",
    "energy_kwh",
    "comfort_degree_hours",
    "occupied_comfort_percentage",
    "mode_changes",
    "mean_arrival_temperature_deficit_f",
    "pre_transition_energy_kwh",
    "pre_transition_energy_percentage",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty results file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_policy_seeds(
    results: list[EpisodeResult], evaluation_seed_count: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[EpisodeResult]] = {}
    for row in results:
        key = (float(row["comfort_weight"]), int(row["training_seed"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (comfort_weight, training_seed), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "comfort_weight": comfort_weight,
            "controller_type": "QLearningAgent",
            "training_seed": training_seed,
            "evaluation_episodes": len(rows),
        }
        for metric in METRICS:
            summary[f"{metric}_mean"] = mean(float(row[metric]) for row in rows)
        if len(rows) != evaluation_seed_count:
            raise RuntimeError(
                f"training seed {training_seed} evaluated on {len(rows)} seeds, "
                f"expected {evaluation_seed_count}"
            )
        summaries.append(summary)
    return summaries


def _summarize_policies(
    policy_seed_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in policy_seed_summaries:
        grouped.setdefault(float(row["comfort_weight"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for comfort_weight, rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "comfort_weight": comfort_weight,
            "controller_type": "QLearningAgent",
            "independent_training_seeds": len(rows),
            "evaluation_episodes_per_policy": rows[0]["evaluation_episodes"],
        }
        for metric in METRICS:
            values = [float(row[f"{metric}_mean"]) for row in rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_training_seed_sd"] = (
                stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def _write_learning_plot(records: list[dict[str, Any]]) -> None:
    grouped: dict[float, dict[int, list[float]]] = {}
    for row in records:
        grouped.setdefault(float(row["comfort_weight"]), {}).setdefault(
            int(row["episode"]), []
        ).append(float(row["episode_return"]))

    fig, axis = plt.subplots(figsize=(8, 5))
    for comfort_weight, episodes in sorted(grouped.items()):
        x_values = sorted(episodes)
        means = [mean(episodes[episode]) for episode in x_values]
        deviations = [
            stdev(episodes[episode]) if len(episodes[episode]) > 1 else 0.0
            for episode in x_values
        ]
        axis.plot(x_values, means, label=f"comfort weight={comfort_weight:g}")
        axis.fill_between(
            x_values,
            [value - sd for value, sd in zip(means, deviations)],
            [value + sd for value, sd in zip(means, deviations)],
            alpha=0.16,
        )
    axis.set_title("Training return (mean +/- SD across training seeds)")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Episode return (weighted reward units)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_curve.png", dpi=160)
    plt.close(fig)


def _write_tradeoff_plot(
    baseline_results: list[EpisodeResult],
    policy_summaries: list[dict[str, Any]],
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    baseline_results = [
        row
        for row in baseline_results
        if float(row["comfort_weight"]) == COMFORT_WEIGHTS[0]
    ]
    for controller_name in (
        "RandomController",
        "DeadbandController",
        "ScheduledSetbackController",
    ):
        rows = [
            row for row in baseline_results if row["controller_type"] == controller_name
        ]
        cost = [float(row["energy_cost_usd"]) for row in rows]
        comfort = [float(row["comfort_degree_hours"]) for row in rows]
        axis.errorbar(
            mean(cost),
            mean(comfort),
            xerr=stdev(cost) if len(cost) > 1 else 0.0,
            yerr=stdev(comfort) if len(comfort) > 1 else 0.0,
            fmt="o",
            capsize=3,
            label=f"{CONTROLLER_DISPLAY_NAMES[controller_name]} (evaluation SD)",
        )
    for row in policy_summaries:
        comfort_weight = float(row["comfort_weight"])
        axis.errorbar(
            float(row["energy_cost_usd_mean"]),
            float(row["comfort_degree_hours_mean"]),
            xerr=float(row["energy_cost_usd_training_seed_sd"]),
            yerr=float(row["comfort_degree_hours_training_seed_sd"]),
            fmt="o",
            capsize=3,
            label=f"Q-learning (comfort={comfort_weight:g}; training-seed SD)",
        )
    axis.set_title("Weekly cost-comfort trade-off (mean +/- SD; lower is better)")
    axis.set_xlabel("Energy cost (USD per simulated week)")
    axis.set_ylabel("Occupied comfort violation (degree-hours)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "reward_tradeoff.png", dpi=160)
    plt.close(fig)


def run(
    training_seed_count: int = 5,
    evaluation_seed_count: int = 20,
    episodes: int | None = None,
) -> None:
    if training_seed_count < 2 or evaluation_seed_count < 2:
        raise ValueError("use at least two training and two held-out evaluation seeds")
    base_config = ExperimentConfig.from_env()
    if episodes is not None:
        base_config = replace(base_config, episodes=episodes)
        base_config.validate()

    training_seeds = list(
        range(base_config.seed, base_config.seed + training_seed_count)
    )
    evaluation_seeds = list(
        range(EVALUATION_SEED_START, EVALUATION_SEED_START + evaluation_seed_count)
    )
    experiment_rows: list[dict[str, Any]] = []
    learning_records: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    policy_evaluations: list[EpisodeResult] = []
    baseline_evaluations: list[EpisodeResult] = []

    for comfort_weight in COMFORT_WEIGHTS:
        sweep_config = replace(base_config, comfort_weight=comfort_weight)
        for training_seed in training_seeds:
            print(
                f"Training Q-learning: comfort_weight={comfort_weight:g}, "
                f"seed={training_seed}, episodes={sweep_config.episodes}"
            )
            train_config = replace(sweep_config, seed=training_seed)
            agent, history = train_q_learning(
                ThermostatEnv(train_config), train_config, seed=training_seed
            )
            state_upper_bound = reachable_state_upper_bound(train_config)
            for record in history:
                learning_records.append(
                    {
                        "comfort_weight": comfort_weight,
                        "training_seed": training_seed,
                        **record,
                    }
                )
            coverage_records.append(
                {
                    "comfort_weight": comfort_weight,
                    "training_seed": training_seed,
                    "episodes": train_config.episodes,
                    "visited_states": agent.visited_states,
                    "state_upper_bound": state_upper_bound,
                    "visited_state_action_pairs": agent.visited_state_action_pairs,
                    "state_action_upper_bound": (state_upper_bound * len(HVACAction)),
                    "state_coverage_percentage": (
                        100.0 * agent.visited_states / state_upper_bound
                    ),
                    "state_action_coverage_percentage": (
                        100.0
                        * agent.visited_state_action_pairs
                        / (state_upper_bound * len(HVACAction))
                    ),
                }
            )
            for evaluation_seed in evaluation_seeds:
                result = evaluate_controller(
                    ThermostatEnv(sweep_config), agent, seed=evaluation_seed
                )
                result.update(
                    {
                        "comfort_weight": comfort_weight,
                        "training_seed": training_seed,
                    }
                )
                policy_evaluations.append(result)
                experiment_rows.append(result)

        baseline_results = []
        for evaluation_seed in evaluation_seeds:
            for controller in (
                RandomController(seed=evaluation_seed),
                DeadbandController(sweep_config),
                ScheduledSetbackController(sweep_config),
            ):
                result = evaluate_controller(
                    ThermostatEnv(sweep_config),
                    controller,
                    seed=evaluation_seed,
                )
                result.update(
                    {
                        "comfort_weight": comfort_weight,
                        "training_seed": "",
                    }
                )
                baseline_results.append(result)
                experiment_rows.append(result)
        baseline_evaluations.extend(baseline_results)

    policy_seed_summaries = _aggregate_policy_seeds(
        policy_evaluations, evaluation_seed_count
    )
    policy_summaries = _summarize_policies(policy_seed_summaries)
    baseline_summaries = []
    for weight in COMFORT_WEIGHTS:
        rows = [row for row in baseline_evaluations if row["comfort_weight"] == weight]
        baseline_summaries.extend(
            {
                "comfort_weight": weight,
                **summary,
            }
            for summary in summarize_results(rows)
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS_DIR / "experiment_runs.csv", experiment_rows)
    _write_csv(RESULTS_DIR / "policy_seed_summary.csv", policy_seed_summaries)
    _write_csv(
        RESULTS_DIR / "experiment_summary.csv",
        baseline_summaries + policy_summaries,
    )
    _write_csv(RESULTS_DIR / "learning_curves.csv", learning_records)
    _write_csv(RESULTS_DIR / "training_coverage.csv", coverage_records)
    metadata = {
        "project": "Smart Thermostat Energy Saver",
        "experiment": "Q-learning cost–comfort reward-weight sweep",
        "comfort_weights": list(COMFORT_WEIGHTS),
        "training_seeds": training_seeds,
        "held_out_evaluation_seeds": evaluation_seeds,
        "evaluation_seed_start": EVALUATION_SEED_START,
        "episodes_per_policy": base_config.episodes,
        "steps_per_episode": base_config.steps_per_episode,
        "pre_transition_window_hours": 2,
        "baseline_policy_seed": "same seed as corresponding evaluation run",
        "q_learning_evaluation_epsilon": 0.0,
        "summary_rule": (
            "Q-learning summaries average each evaluation-seed set within "
            "training seed, then report mean and SD across independent training "
            "seeds; rule-based baselines report mean and sample SD across "
            "evaluation seeds."
        ),
        "config": asdict(base_config),
    }
    (RESULTS_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _write_learning_plot(learning_records)
    _write_tradeoff_plot(baseline_evaluations, policy_summaries)

    print(
        f"Completed {len(policy_evaluations)} Q-learning evaluations and "
        f"{len(baseline_evaluations)} baseline evaluations."
    )
    for row in policy_summaries:
        print(
            f"comfort_weight={row['comfort_weight']}: "
            f"Q-learning cost=${row['energy_cost_usd_mean']:.2f} "
            f"(training-seed SD "
            f"${row['energy_cost_usd_training_seed_sd']:.2f}), "
            f"comfort={row['comfort_degree_hours_mean']:.1f} "
            f"(SD {row['comfort_degree_hours_training_seed_sd']:.1f}) "
            "degree-hours"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seeds", type=int, default=5)
    parser.add_argument("--evaluation-seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int)
    arguments = parser.parse_args()
    run(
        training_seed_count=arguments.training_seeds,
        evaluation_seed_count=arguments.evaluation_seeds,
        episodes=arguments.episodes,
    )


if __name__ == "__main__":
    main()
