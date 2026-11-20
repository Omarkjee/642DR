from dataclasses import replace

import numpy as np

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.env import HVACAction, ThermostatEnv
from thermostat_rl.q_learning import TabularQLearning


def small_config() -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        episodes=6,
        steps_per_episode=24,
        initial_temperature_std_f=0.0,
        temperature_noise_std_f=0.0,
    )


def test_state_qvalues_initialize_and_preserve_action_values() -> None:
    agent = TabularQLearning(small_config())
    state = (0, 0, 0, 0, 0, 0, 0)

    np.testing.assert_array_equal(agent.state_qValues(state), np.zeros(len(HVACAction)))
    agent.q_table[state] = np.asarray([6.0, 4.0, 2.0, 642.0, 64.2])
    np.testing.assert_array_equal(
        agent.state_qValues(state), np.asarray([6.0, 4.0, 2.0, 642.0, 64.2])
    )


def test_epsilon_greedy_selection_uses_greedy_and_seeded_random_actions() -> None:
    config = small_config()
    state = (0, 0, 0, 0, 0, 0, 0)
    first = TabularQLearning(config)
    second = TabularQLearning(config)
    first.q_table[state] = np.asarray([6.0, 4.0, 2.0, 642.0, 64.2])
    second.q_table[state] = first.q_table[state].copy()

    first.epsilon = second.epsilon = 0.0
    assert first.select_action(state) == HVACAction.COOL_LOW
    assert second.select_action(state) == HVACAction.COOL_LOW

    first.epsilon = second.epsilon = 1.0
    first_actions = [first.select_action(state) for _ in range(20)]
    second_actions = [second.select_action(state) for _ in range(20)]
    assert first_actions == second_actions
    assert all(action in set(HVACAction) for action in first_actions)


def test_q_value_update_bootstraps_only_nonterminal_states() -> None:
    config = replace(small_config(), learning_rate=0.5, discount_factor=0.9)
    agent = TabularQLearning(config)
    state = (0, 0, 0, 0, 0, 0, 0)
    next_state = (1, 0, 0, 0, 0, 0, 0)
    agent.q_table[state] = np.asarray([6.0, 4.0, 2.0, 0.0, 0.0])
    agent.q_table[next_state] = np.asarray([6.0, 0.0, 10.0, 0.0, 0.0])

    agent.update_qValue(state, HVACAction.HEAT_LOW, 10.0, next_state)
    assert agent.q_table[state][HVACAction.HEAT_LOW] == 11.5

    agent.q_table[state][HVACAction.HEAT_LOW] = 4.0
    agent.update_qValue(
        state, HVACAction.HEAT_LOW, 10.0, next_state, terminal_state=True
    )
    assert agent.q_table[state][HVACAction.HEAT_LOW] == 7.0


def test_epsilon_decay_respects_configured_minimum() -> None:
    config = small_config()
    agent = TabularQLearning(config)
    initial_epsilon = agent.epsilon

    agent.update_epsilon_decay()
    assert config.epsilon_min < agent.epsilon < initial_epsilon

    for _ in range(10_000):
        agent.update_epsilon_decay()
    assert agent.epsilon == config.epsilon_min


def test_single_episode_training_updates_table_and_decays_epsilon() -> None:
    config = small_config()
    agent = TabularQLearning(config)

    episode_return = agent.train_QLearning_episode(ThermostatEnv(config), seed=642)

    assert isinstance(episode_return, float)
    assert agent.q_table
    assert agent.epsilon < config.epsilon_start


def test_multiple_episode_training_is_reproducible() -> None:
    config = small_config()
    first = TabularQLearning(config)
    second = TabularQLearning(config)

    first_returns = first.train_QLearning(ThermostatEnv(config))
    second_returns = second.train_QLearning(ThermostatEnv(config))

    assert len(first_returns) == config.episodes
    assert first_returns == second_returns
    assert first.q_table.keys() == second.q_table.keys()
    for state, values in first.q_table.items():
        np.testing.assert_array_equal(values, second.q_table[state])
    assert first.epsilon == second.epsilon
