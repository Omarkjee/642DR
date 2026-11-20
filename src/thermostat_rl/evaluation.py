"""Fair, seed-controlled episode evaluation and baseline comparisons."""

from collections import defaultdict
from math import sqrt
from typing import TypeAlias

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.controllers import (
    Controller,
    DeadbandController,
    RandomController,
    ScheduledSetbackController,
)
from thermostat_rl.env import ThermostatEnv

MetricValue: TypeAlias = str | int | float
EpisodeResult: TypeAlias = dict[str, MetricValue]

CONTROLLER_NAMES = (
    "RandomController",
    "DeadbandController",
    "ScheduledSetbackController",
)
CONTROLLER_DISPLAY_NAMES = {
    "RandomController": "Random",
    "DeadbandController": "Deadband",
    "ScheduledSetbackController": "Scheduled setback",
}
EVALUATION_METRICS = (
    "energy_cost_usd",
    "energy_kwh",
    "comfort_degree_hours",
    "occupied_comfort_percentage",
    "mode_changes",
    "mean_arrival_temperature_deficit_f",
    "pre_transition_energy_kwh",
    "pre_transition_energy_percentage",
)


def evaluate_controller(
    environment: ThermostatEnv,
    controller: Controller,
    seed: int = 642,
    pre_transition_window_hours: int = 2,
) -> EpisodeResult:
    """Evaluate one episode; transitions use the next observation boundary."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if pre_transition_window_hours <= 0:
        raise ValueError("pre_transition_window_hours must be positive")

    observation, _ = environment.reset(seed=seed)
    energy_cost_usd = 0.0
    energy_kwh = 0.0
    comfort_degree_hours = 0.0
    occupied_steps = 0
    comfortable_occupied_steps = 0
    mode_changes = 0
    arrival_deficits_f: list[float] = []
    energy_by_step: dict[int, float] = {}
    pre_transition_steps: set[int] = set()

    for _ in range(environment.config.steps_per_episode):
        action = controller.select_action(observation)
        next_observation, _, terminated, truncated, info = environment.step(action)
        step_index = int(info["step"])
        step_energy = float(info["energy_kwh"])
        energy_by_step[step_index] = step_energy
        energy_cost_usd += float(info["energy_cost_usd"])
        energy_kwh += step_energy
        comfort_degree_hours += float(info["comfort_violation_degree_hours"])
        mode_changes += int(bool(info["mode_changed"]))

        if info["occupancy"]:
            occupied_steps += 1
            comfortable_occupied_steps += int(bool(info["occupied_comfortable"]))

        occupancy_arrival = (
            not observation["occupancy"] and next_observation["occupancy"]
        )
        if occupancy_arrival:
            indoor_temperature = float(next_observation["indoor_temperature_f"][0])
            arrival_deficits_f.append(
                max(environment.config.comfort_low_f - indoor_temperature, 0.0)
            )

        schedule_transition = (
            observation["occupancy"] != next_observation["occupancy"]
            or observation["price_tier"] != next_observation["price_tier"]
        )
        if schedule_transition:
            start = max(0, step_index - pre_transition_window_hours + 1)
            pre_transition_steps.update(range(start, step_index + 1))

        observation = next_observation
        if terminated or truncated:
            break
    else:
        raise RuntimeError("evaluation episode did not terminate within its step limit")

    pre_transition_energy_kwh = sum(
        energy_by_step[step] for step in pre_transition_steps
    )
    return {
        "seed": seed,
        "controller_type": type(controller).__name__,
        "energy_cost_usd": energy_cost_usd,
        "energy_kwh": energy_kwh,
        "comfort_degree_hours": comfort_degree_hours,
        "occupied_comfort_percentage": (
            100.0 * comfortable_occupied_steps / occupied_steps
            if occupied_steps
            else 0.0
        ),
        "mode_changes": mode_changes,
        "arrival_count": len(arrival_deficits_f),
        "mean_arrival_temperature_deficit_f": (
            sum(arrival_deficits_f) / len(arrival_deficits_f)
            if arrival_deficits_f
            else 0.0
        ),
        "pre_transition_count": len(pre_transition_steps),
        "pre_transition_energy_kwh": pre_transition_energy_kwh,
        "pre_transition_energy_percentage": (
            100.0 * pre_transition_energy_kwh / energy_kwh if energy_kwh else 0.0
        ),
    }


def controller_baselines(
    environment: ThermostatEnv,
    config: ExperimentConfig,
    seed_list: list[int],
) -> list[EpisodeResult]:
    """Run all three baseline policies on each identical evaluation seed."""
    results: list[EpisodeResult] = []
    for seed in seed_list:
        controllers: tuple[Controller, ...] = (
            RandomController(seed=seed),
            DeadbandController(config),
            ScheduledSetbackController(config),
        )
        results.extend(
            evaluate_controller(environment, controller, seed=seed)
            for controller in controllers
        )
    return results


def summarize_results(
    results: list[EpisodeResult],
    group_by: tuple[str, ...] = ("controller_type",),
) -> list[dict[str, MetricValue]]:
    """Summarize each controller with mean, sample SD, and 95% normal CI."""
    if not group_by:
        raise ValueError("group_by must name at least one result field")
    grouped: dict[tuple[MetricValue, ...], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        try:
            group_key = tuple(result[field] for field in group_by)
        except KeyError as exc:
            raise ValueError(f"unknown grouping field: {exc.args[0]}") from exc
        grouped[group_key].append(result)

    summaries: list[dict[str, MetricValue]] = []
    for group_key in sorted(grouped, key=lambda group: tuple(map(str, group))):
        rows = grouped[group_key]
        summary: dict[str, MetricValue] = dict(zip(group_by, group_key))
        summary["episodes"] = len(rows)
        for metric in EVALUATION_METRICS:
            values = [float(row[metric]) for row in rows]
            mean_value = sum(values) / len(values)
            standard_deviation = (
                sqrt(
                    sum((value - mean_value) ** 2 for value in values)
                    / (len(values) - 1)
                )
                if len(values) > 1
                else 0.0
            )
            margin = 1.96 * standard_deviation / sqrt(len(values))
            summary[f"{metric}_mean"] = mean_value
            summary[f"{metric}_sd"] = standard_deviation
            summary[f"{metric}_ci95_low"] = mean_value - margin
            summary[f"{metric}_ci95_high"] = mean_value + margin
        summaries.append(summary)
    return summaries
