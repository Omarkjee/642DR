"""Sparse tabular Q-learning for the discrete thermostat action space."""

from collections import defaultdict
from math import isfinite
from typing import TypeAlias

import numpy as np

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.env import HVACAction, ThermostatEnv, ThermostatObservation
from thermostat_rl.state import (
    StateKey,
    discretize_observation,
    reachable_state_upper_bound,
)

TrainingRecord: TypeAlias = dict[str, int | float]


class QLearningAgent:
    """Tabular off-policy Q-learning with seeded epsilon-greedy actions."""

    def __init__(
        self, config: ExperimentConfig | None = None, seed: int | None = None
    ) -> None:
        self.config = config or ExperimentConfig()
        self.config.validate()
        agent_seed = self.config.seed if seed is None else seed
        if agent_seed < 0:
            raise ValueError("seed must be non-negative")
        self._rng = np.random.default_rng(agent_seed)
        self.q_values: dict[StateKey, np.ndarray] = {}
        self.action_visits: defaultdict[tuple[StateKey, int], int] = defaultdict(int)

    @property
    def visited_state_action_pairs(self) -> int:
        return len(self.action_visits)

    @property
    def visited_states(self) -> int:
        return len(self.q_values)

    def select_action(
        self, observation: ThermostatObservation, epsilon: float = 0.0
    ) -> HVACAction:
        if not isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be finite and in [0, 1]")
        state = discretize_observation(observation, self.config)
        if self._rng.random() < epsilon:
            return HVACAction(int(self._rng.integers(len(HVACAction))))
        values = self.q_values.get(state)
        if values is None:
            return HVACAction.OFF
        return HVACAction(int(np.argmax(values)))

    def update(
        self,
        state: StateKey,
        action: HVACAction | int,
        reward: float,
        next_state: StateKey,
        terminated: bool,
    ) -> None:
        action_index = int(action)
        if action_index not in range(len(HVACAction)):
            raise ValueError(f"action must be in [0, {len(HVACAction)})")
        if not isfinite(reward):
            raise ValueError("reward must be finite")

        values = self.q_values.setdefault(state, np.zeros(len(HVACAction)))
        next_values = self.q_values.setdefault(next_state, np.zeros(len(HVACAction)))
        bootstrap = (
            0.0
            if terminated
            else self.config.discount_factor * float(np.max(next_values))
        )
        target = reward + bootstrap
        values[action_index] += self.config.learning_rate * (
            target - values[action_index]
        )
        self.action_visits[(state, action_index)] += 1


def train_q_learning(
    environment: ThermostatEnv,
    config: ExperimentConfig,
    seed: int | None = None,
) -> tuple[QLearningAgent, list[TrainingRecord]]:
    """Train over configured episodes; return agent and reproducible learning log."""
    training_seed = config.seed if seed is None else seed
    if training_seed < 0:
        raise ValueError("seed must be non-negative")

    agent = QLearningAgent(config, seed=training_seed)
    history: list[TrainingRecord] = []
    epsilon = config.epsilon_start
    state_upper_bound = reachable_state_upper_bound(config)
    pair_upper_bound = state_upper_bound * len(HVACAction)

    for episode in range(config.episodes):
        episode_seed = training_seed * config.episodes + episode
        observation, _ = environment.reset(seed=episode_seed)
        episode_return = 0.0
        for _ in range(config.steps_per_episode):
            state = discretize_observation(observation, config)
            action = agent.select_action(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, _ = environment.step(
                action
            )
            next_state = discretize_observation(next_observation, config)
            agent.update(
                state,
                action,
                reward,
                next_state,
                terminated=terminated or truncated,
            )
            episode_return += reward
            observation = next_observation
            if terminated or truncated:
                break
        else:
            raise RuntimeError(
                "training episode did not terminate within its step limit"
            )

        history.append(
            {
                "episode": episode + 1,
                "episode_return": episode_return,
                "epsilon": epsilon,
                "visited_states": agent.visited_states,
                "visited_state_action_pairs": agent.visited_state_action_pairs,
                "state_coverage_percentage": (
                    100.0 * agent.visited_states / state_upper_bound
                ),
                "state_action_coverage_percentage": (
                    100.0 * agent.visited_state_action_pairs / pair_upper_bound
                ),
            }
        )
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

    return agent, history


class TabularQLearning:
    """State-key based Q-learning API contributed for the Week 4 deliverable."""

    def __init__(self, config: ExperimentConfig) -> None:
        config.validate()
        self.config = config
        self.random_num = np.random.default_rng(config.seed)
        self.epsilon = config.epsilon_start
        self.q_table: dict[StateKey, np.ndarray] = {}

    def state_qValues(self, state: StateKey) -> np.ndarray:
        """Return the state's action values, initializing unseen states to zero."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(HVACAction))
        return self.q_table[state]

    def select_action(self, state: StateKey) -> HVACAction:
        """Select an action using this agent's epsilon-greedy policy."""
        q_values = self.state_qValues(state)
        if self.random_num.random() < self.epsilon:
            return HVACAction(int(self.random_num.integers(len(HVACAction))))
        return HVACAction(int(np.argmax(q_values)))

    def update_qValue(
        self,
        state: StateKey,
        action: HVACAction,
        reward: float,
        next_state: StateKey,
        terminal_state: bool = False,
    ) -> None:
        """Apply one tabular Q-learning update."""
        current_values = self.state_qValues(state)
        next_values = self.state_qValues(next_state)
        bootstrap = 0.0 if terminal_state else float(np.max(next_values))
        action_index = int(action)
        current_values[action_index] += self.config.learning_rate * (
            reward
            + self.config.discount_factor * bootstrap
            - current_values[action_index]
        )

    def update_epsilon_decay(self) -> None:
        """Decay exploration to its configured minimum."""
        self.epsilon = max(
            self.config.epsilon_min, self.epsilon * self.config.epsilon_decay
        )

    def train_QLearning_episode(
        self, environment: ThermostatEnv, seed: int | None = None
    ) -> float:
        """Train one episode and decay exploration once after the episode."""
        observation, _ = environment.reset(seed=seed)
        state = discretize_observation(observation, self.config)
        episode_return = 0.0

        for _ in range(self.config.steps_per_episode):
            action = self.select_action(state)
            observation, reward, terminated, truncated, _ = environment.step(action)
            next_state = discretize_observation(observation, self.config)
            self.update_qValue(
                state,
                action,
                reward,
                next_state,
                terminal_state=terminated or truncated,
            )
            episode_return += reward
            state = next_state
            if terminated or truncated:
                self.update_epsilon_decay()
                return episode_return

        raise RuntimeError("training episode did not terminate within its step limit")

    def train_QLearning(self, environment: ThermostatEnv) -> list[float]:
        """Train for the configured episode count using reproducible reset seeds."""
        return [
            self.train_QLearning_episode(environment, seed=self.config.seed + episode)
            for episode in range(self.config.episodes)
        ]
