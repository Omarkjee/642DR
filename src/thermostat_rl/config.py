"""Validated settings loaded from the process environment."""

from dataclasses import dataclass
from math import isfinite
import os


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {value!r}") from exc


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; received {value!r}") from exc


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible starting settings for training and reward design."""

    seed: int = 42
    episodes: int = 500
    steps_per_episode: int = 168
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    cost_weight: float = 1.0
    comfort_weight: float = 5.0
    switch_weight: float = 0.1

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Load settings from THERMOSTAT_* variables, using defaults if absent."""
        config = cls(
            seed=_read_int("THERMOSTAT_SEED", cls.seed),
            episodes=_read_int("THERMOSTAT_EPISODES", cls.episodes),
            steps_per_episode=_read_int(
                "THERMOSTAT_STEPS_PER_EPISODE", cls.steps_per_episode
            ),
            learning_rate=_read_float(
                "THERMOSTAT_LEARNING_RATE", cls.learning_rate
            ),
            discount_factor=_read_float(
                "THERMOSTAT_DISCOUNT_FACTOR", cls.discount_factor
            ),
            epsilon_start=_read_float(
                "THERMOSTAT_EPSILON_START", cls.epsilon_start
            ),
            epsilon_min=_read_float("THERMOSTAT_EPSILON_MIN", cls.epsilon_min),
            epsilon_decay=_read_float(
                "THERMOSTAT_EPSILON_DECAY", cls.epsilon_decay
            ),
            cost_weight=_read_float("THERMOSTAT_COST_WEIGHT", cls.cost_weight),
            comfort_weight=_read_float(
                "THERMOSTAT_COMFORT_WEIGHT", cls.comfort_weight
            ),
            switch_weight=_read_float(
                "THERMOSTAT_SWITCH_WEIGHT", cls.switch_weight
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject settings outside the domains required by the algorithms."""
        if self.seed < 0:
            raise ValueError("THERMOSTAT_SEED must be non-negative")
        if self.episodes <= 0:
            raise ValueError("THERMOSTAT_EPISODES must be greater than zero")
        if self.steps_per_episode <= 0:
            raise ValueError("THERMOSTAT_STEPS_PER_EPISODE must be greater than zero")
        numeric_settings = {
            "THERMOSTAT_LEARNING_RATE": self.learning_rate,
            "THERMOSTAT_DISCOUNT_FACTOR": self.discount_factor,
            "THERMOSTAT_EPSILON_START": self.epsilon_start,
            "THERMOSTAT_EPSILON_MIN": self.epsilon_min,
            "THERMOSTAT_EPSILON_DECAY": self.epsilon_decay,
            "THERMOSTAT_COST_WEIGHT": self.cost_weight,
            "THERMOSTAT_COMFORT_WEIGHT": self.comfort_weight,
            "THERMOSTAT_SWITCH_WEIGHT": self.switch_weight,
        }
        for name, value in numeric_settings.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("THERMOSTAT_LEARNING_RATE must be in (0, 1]")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("THERMOSTAT_DISCOUNT_FACTOR must be in [0, 1]")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError(
                "epsilon values must satisfy 0 <= EPSILON_MIN <= "
                "EPSILON_START <= 1"
            )
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("THERMOSTAT_EPSILON_DECAY must be in (0, 1]")
        if min(self.cost_weight, self.comfort_weight, self.switch_weight) < 0:
            raise ValueError("reward weights must be non-negative")
        if self.cost_weight + self.comfort_weight + self.switch_weight == 0:
            raise ValueError("at least one reward weight must be greater than zero")
