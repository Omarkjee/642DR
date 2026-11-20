from dataclasses import replace

import numpy as np
import pytest

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.controllers import DeadbandController
from thermostat_rl.env import HVACAction, ThermostatEnv
from thermostat_rl.evaluation import (
    controller_baselines,
    evaluate_controller,
    summarize_results,
)
from thermostat_rl.q_learning import train_q_learning


class ConstantController:
    def __init__(self, action: HVACAction) -> None:
        self.action = action

    def select_action(self, observation: dict[str, object]) -> HVACAction:
        del observation
        return self.action


def test_evaluation_metrics_and_schedule_transitions_are_well_defined() -> None:
    config = replace(
        ExperimentConfig(),
        steps_per_episode=24,
        initial_temperature_mean_f=60.0,
        initial_temperature_std_f=0.0,
        temperature_noise_std_f=0.0,
    )
    result = evaluate_controller(
        ThermostatEnv(config),
        ConstantController(HVACAction.HEAT_LOW),
        seed=642,
    )

    assert result["controller_type"] == "ConstantController"
    assert result["seed"] == 642
    assert result["energy_kwh"] == 24 * config.heating_low_kw
    assert result["energy_cost_usd"] > 0
    assert result["comfort_degree_hours"] >= 0
    assert 0 <= result["occupied_comfort_percentage"] <= 100
    assert result["mode_changes"] == 1
    assert result["arrival_count"] == 1
    assert result["mean_arrival_temperature_deficit_f"] >= 0
    assert result["pre_transition_count"] == 8
    assert result["pre_transition_energy_kwh"] == 8 * config.heating_low_kw
    assert result["pre_transition_energy_percentage"] == pytest.approx(100 / 3)


def test_trained_q_learning_agent_is_evaluated_without_exploration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        ExperimentConfig(),
        episodes=4,
        steps_per_episode=24,
        initial_temperature_std_f=0.0,
        temperature_noise_std_f=0.0,
    )
    agent, history = train_q_learning(ThermostatEnv(config), config, seed=42)
    assert history
    assert agent.q_values

    selected_epsilons: list[float] = []
    select_action = agent.select_action

    def record_selection(
        observation: dict[str, object], epsilon: float = 0.0
    ) -> HVACAction:
        selected_epsilons.append(epsilon)
        return select_action(observation, epsilon=epsilon)

    monkeypatch.setattr(agent, "select_action", record_selection)
    result = evaluate_controller(ThermostatEnv(config), agent, seed=123)

    assert result["controller_type"] == "QLearningAgent"
    assert result["seed"] == 123
    assert result["energy_cost_usd"] >= 0
    assert selected_epsilons
    assert set(selected_epsilons) == {0.0}


def test_evaluation_rejects_invalid_seed_and_transition_window() -> None:
    environment = ThermostatEnv()
    controller = DeadbandController(environment.config)

    with pytest.raises(ValueError, match="seed"):
        evaluate_controller(environment, controller, seed=-1)
    with pytest.raises(ValueError, match="window"):
        evaluate_controller(environment, controller, pre_transition_window_hours=0)


def test_baselines_use_shared_seeds_and_return_aggregate_statistics() -> None:
    config = ExperimentConfig()
    environment = ThermostatEnv(config)
    seeds = [642, 689, 2026]

    results = controller_baselines(environment, config, seeds)
    summaries = summarize_results(results)

    assert len(results) == 3 * len(seeds)
    assert {row["seed"] for row in results} == set(seeds)
    assert {row["controller_type"] for row in results} == {
        "RandomController",
        "DeadbandController",
        "ScheduledSetbackController",
    }
    assert len(summaries) == 3
    for summary in summaries:
        assert summary["episodes"] == len(seeds)
        assert summary["energy_cost_usd_sd"] >= 0
        assert (
            summary["comfort_degree_hours_ci95_low"]
            <= summary["comfort_degree_hours_mean"]
        )
        assert (
            summary["comfort_degree_hours_mean"]
            <= summary["comfort_degree_hours_ci95_high"]
        )


def test_random_baseline_repeats_same_seed_and_rules_share_trajectory() -> None:
    config = ExperimentConfig()
    environment = ThermostatEnv(config)
    first = controller_baselines(environment, config, [1234])
    second = controller_baselines(environment, config, [1234])

    assert first == second
    deadband, setback = first[1], first[2]
    assert deadband["energy_cost_usd"] != setback["energy_cost_usd"] or (
        deadband["comfort_degree_hours"] != setback["comfort_degree_hours"]
    )
