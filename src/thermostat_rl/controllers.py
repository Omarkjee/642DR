"""Reproducible reference controllers for the thermostat environment."""

from math import isfinite
from typing import Protocol

import numpy as np

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.env import HVACAction, ThermostatObservation


class Controller(Protocol):
    def select_action(self, observation: ThermostatObservation) -> HVACAction: ...


class RandomController:
    """Uniform random actions from an independently seeded generator."""

    def __init__(self, seed: int = 642) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: ThermostatObservation) -> HVACAction:
        del observation
        return HVACAction(int(self._rng.integers(len(HVACAction))))


class DeadbandController:
    """Heat or cool outside the occupied comfort band; otherwise turn off."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def select_action(self, observation: ThermostatObservation) -> HVACAction:
        indoor_temperature = float(observation["indoor_temperature_f"][0])
        if indoor_temperature < self.config.comfort_low_f:
            return HVACAction.HEAT_LOW
        if indoor_temperature > self.config.comfort_high_f:
            return HVACAction.COOL_LOW
        return HVACAction.OFF


class ScheduledSetbackController:
    """Widen the deadband while unoccupied and restore it when residents return."""

    def __init__(
        self, config: ExperimentConfig, comfort_setback_f: float = 4.0
    ) -> None:
        if not isfinite(comfort_setback_f) or comfort_setback_f < 0:
            raise ValueError("comfort_setback_f must be finite and non-negative")
        self.config = config
        self.comfort_setback_f = comfort_setback_f

    def select_action(self, observation: ThermostatObservation) -> HVACAction:
        indoor_temperature = float(observation["indoor_temperature_f"][0])
        setback = self.comfort_setback_f if not observation["occupancy"] else 0.0
        lower_bound = self.config.comfort_low_f - setback
        upper_bound = self.config.comfort_high_f + setback
        if indoor_temperature < lower_bound:
            return HVACAction.HEAT_LOW
        if indoor_temperature > upper_bound:
            return HVACAction.COOL_LOW
        return HVACAction.OFF
