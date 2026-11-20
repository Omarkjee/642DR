from dataclasses import replace

import numpy as np
import pytest

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.env import HVACAction, ThermostatEnv
from thermostat_rl.q_learning import QLearningAgent, train_q_learning
from thermostat_rl.state import discretize_observation


def small_config() -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        episodes=6,
        steps_per_episode=24,
        initial_temperature_std_f=0.0,
        temperature_noise_std_f=0.0,
    )


def test_q_learning_update_bootstraps_nonterminal_but_not_terminal() -> None:
    config = replace(ExperimentConfig(), learning_rate=0.5, discount_factor=0.9)
    agent = QLearningAgent(config, seed=3)
    state = (1, 2, 3, 1, 0, 0, 0)
    next_state = (1, 2, 4, 1, 0, 0, 0)
    agent.q_values[state] = np.asarray([4.0, 0, 0, 0, 0])
    agent.q_values[next_state] = np.asarray([1.0, 2.0, 10.0, 0, 0])

    agent.update(
        state, HVACAction.OFF, reward=2.0, next_state=next_state, terminated=False
    )
    assert agent.q_values[state][HVACAction.OFF] == pytest.approx(7.5)

    agent.q_values[state][HVACAction.OFF] = 4.0
    agent.update(
        state, HVACAction.OFF, reward=2.0, next_state=next_state, terminated=True
    )
    assert agent.q_values[state][HVACAction.OFF] == pytest.approx(3.0)
    assert agent.action_visits[(state, HVACAction.OFF)] == 2


def test_greedy_selection_uses_learned_values_and_unseen_state_defaults_to_off() -> (
    None
):
    config = small_config()
    environment = ThermostatEnv(config)
    observation, _ = environment.reset(seed=1)
    agent = QLearningAgent(config, seed=1)

    assert agent.select_action(observation) == HVACAction.OFF
    state = discretize_observation(observation, config)
    agent.q_values[state] = np.asarray([-1.0, 0.0, 0.0, -2.0, -3.0])
    assert agent.select_action(observation) == HVACAction.HEAT_LOW
    assert agent.select_action(observation, epsilon=1.0) in set(HVACAction)


def test_epsilon_and_update_reject_invalid_values() -> None:
    agent = QLearningAgent(seed=1)
    observation, _ = ThermostatEnv().reset(seed=1)
    with pytest.raises(ValueError, match="epsilon"):
        agent.select_action(observation, epsilon=1.1)
    with pytest.raises(ValueError, match="reward"):
        agent.update(
            (0, 0, 0, 0, 0, 0, 0), 0, float("nan"), (0, 0, 0, 0, 0, 0, 0), False
        )


def test_short_training_is_seeded_and_records_coverage() -> None:
    config = small_config()
    first_agent, first_history = train_q_learning(
        ThermostatEnv(config), config, seed=42
    )
    second_agent, second_history = train_q_learning(
        ThermostatEnv(config), config, seed=42
    )

    assert first_history == second_history
    assert first_agent.visited_states == second_agent.visited_states
    assert (
        first_agent.visited_state_action_pairs
        == second_agent.visited_state_action_pairs
    )
    assert len(first_history) == config.episodes
    assert first_history[0]["epsilon"] == config.epsilon_start
    assert first_history[-1]["epsilon"] < config.epsilon_start
    assert first_history[-1]["visited_states"] == first_agent.visited_states
    assert first_history[-1]["visited_state_action_pairs"] > 0
    for state, values in first_agent.q_values.items():
        np.testing.assert_array_equal(values, second_agent.q_values[state])


def test_training_rejects_negative_seed() -> None:
    config = small_config()
    with pytest.raises(ValueError, match="seed"):
        train_q_learning(ThermostatEnv(config), config, seed=-1)
