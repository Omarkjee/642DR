"""A reproducible single-zone thermostat simulator with a Gymnasium API."""

from enum import IntEnum
from math import cos, pi
from typing import TypedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from thermostat_rl.config import ExperimentConfig


class HVACMode(IntEnum):
    OFF = 0
    HEATING = 1
    COOLING = 2


class HVACAction(IntEnum):
    OFF = 0
    HEAT_LOW = 1
    HEAT_HIGH = 2
    COOL_LOW = 3
    COOL_HIGH = 4


class ThermostatObservation(TypedDict):
    indoor_temperature_f: np.ndarray
    outdoor_temperature_f: np.ndarray
    hour: int
    occupancy: int
    day_type: int
    hvac_mode: int
    price_tier: int


class ThermostatEnv(gym.Env[ThermostatObservation, int]):
    """One-hour control steps over a repeating seven-day occupancy schedule."""

    metadata = {"render_modes": []}

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExperimentConfig()
        self.config.validate()
        self.action_space = spaces.Discrete(len(HVACAction))
        self.observation_space = spaces.Dict(
            {
                "indoor_temperature_f": spaces.Box(
                    low=0.0, high=120.0, shape=(), dtype=np.float32
                ),
                "outdoor_temperature_f": spaces.Box(
                    low=-60.0, high=120.0, shape=(), dtype=np.float32
                ),
                "hour": spaces.Discrete(24),
                "occupancy": spaces.Discrete(2),
                "day_type": spaces.Discrete(2),
                "hvac_mode": spaces.Discrete(len(HVACMode)),
                "price_tier": spaces.Discrete(2),
            }
        )
        self._elapsed_steps = 0
        self._indoor_temperature_f = self.config.initial_temperature_mean_f
        self._hvac_mode = HVACMode.OFF
        self._has_reset = False
        self._terminated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, float] | None = None,
    ) -> tuple[ThermostatObservation, dict[str, float | int]]:
        super().reset(seed=seed)
        options = options or {}
        unknown_options = set(options) - {"initial_temperature_f"}
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise ValueError(f"unsupported reset option(s): {names}")

        if "initial_temperature_f" in options:
            initial_temperature = float(options["initial_temperature_f"])
        else:
            initial_temperature = float(
                self.np_random.normal(
                    self.config.initial_temperature_mean_f,
                    self.config.initial_temperature_std_f,
                )
            )
        if not 0.0 <= initial_temperature <= 120.0:
            raise ValueError("initial_temperature_f must be in [0, 120]")

        self._elapsed_steps = 0
        self._indoor_temperature_f = initial_temperature
        self._hvac_mode = HVACMode.OFF
        self._has_reset = True
        self._terminated = False
        return self._observation(), {
            "initial_temperature_f": initial_temperature,
            "time_step_hours": self.config.time_step_hours,
        }

    def step(
        self, action: int
    ) -> tuple[ThermostatObservation, float, bool, bool, dict[str, float | int | bool]]:
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("episode is terminated; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {len(HVACAction)})")

        action = HVACAction(int(action))
        step_index = self._elapsed_steps
        hour = step_index % 24
        day_index = (step_index // 24) % 7
        day_type = int(day_index >= 5)
        occupied = self._is_occupied(hour, day_type)
        price_tier = self._price_tier(hour)
        price = (
            self.config.peak_price_usd_per_kwh
            if price_tier
            else self.config.off_peak_price_usd_per_kwh
        )
        outdoor_temperature = self._outdoor_temperature(hour)

        previous_mode = self._hvac_mode
        self._hvac_mode = self._mode_for_action(action)
        mode_changed = self._hvac_mode != previous_mode
        hvac_power_kw = self._power_for_action(action)
        energy_kwh = abs(hvac_power_kw) * self.config.time_step_hours
        energy_cost_usd = energy_kwh * price

        heat_flow_kw = (
            outdoor_temperature - self._indoor_temperature_f
        ) / self.config.thermal_resistance_f_per_kw + hvac_power_kw
        temperature_change_f = (
            self.config.time_step_hours
            / self.config.thermal_capacitance_kwh_per_f
            * heat_flow_kw
        )
        temperature_noise_f = float(
            self.np_random.normal(0.0, self.config.temperature_noise_std_f)
        )
        self._indoor_temperature_f += temperature_change_f + temperature_noise_f

        comfort_violation_degree_hours = 0.0
        if occupied:
            comfort_violation_f = max(
                self.config.comfort_low_f - self._indoor_temperature_f,
                self._indoor_temperature_f - self.config.comfort_high_f,
                0.0,
            )
            comfort_violation_degree_hours = (
                comfort_violation_f * self.config.time_step_hours
            )
        reward = -(
            self.config.cost_weight * energy_cost_usd
            + self.config.comfort_weight * comfort_violation_degree_hours
            + self.config.switch_weight * int(mode_changed)
        )

        self._elapsed_steps += 1
        self._terminated = self._elapsed_steps >= self.config.steps_per_episode
        observation = self._observation()
        info: dict[str, float | int | bool] = {
            "step": step_index,
            "hour": hour,
            "day_type": day_type,
            "occupancy": int(occupied),
            "outdoor_temperature_f": outdoor_temperature,
            "indoor_temperature_f": self._indoor_temperature_f,
            "hvac_mode": int(self._hvac_mode),
            "hvac_power_kw": hvac_power_kw,
            "energy_kwh": energy_kwh,
            "price_tier": price_tier,
            "price_usd_per_kwh": price,
            "energy_cost_usd": energy_cost_usd,
            "comfort_violation_degree_hours": comfort_violation_degree_hours,
            "occupied_comfortable": bool(
                occupied and comfort_violation_degree_hours == 0.0
            ),
            "mode_changed": mode_changed,
        }
        return observation, float(reward), self._terminated, False, info

    def _observation(self) -> ThermostatObservation:
        step_index = self._elapsed_steps
        hour = step_index % 24
        day_index = (step_index // 24) % 7
        day_type = int(day_index >= 5)
        return {
            "indoor_temperature_f": np.asarray(
                self._indoor_temperature_f, dtype=np.float32
            ),
            "outdoor_temperature_f": np.asarray(
                self._outdoor_temperature(hour), dtype=np.float32
            ),
            "hour": hour,
            "occupancy": int(self._is_occupied(hour, day_type)),
            "day_type": day_type,
            "hvac_mode": int(self._hvac_mode),
            "price_tier": self._price_tier(hour),
        }

    def _outdoor_temperature(self, hour: int) -> float:
        phase = 2.0 * pi * (hour - self.config.outdoor_temperature_peak_hour) / 24.0
        return self.config.outdoor_temperature_mean_f + (
            self.config.outdoor_temperature_amplitude_f * cos(phase)
        )

    def _is_occupied(self, hour: int, day_type: int) -> bool:
        if day_type:
            return True
        return not (
            self.config.weekday_away_start_hour
            <= hour
            < self.config.weekday_away_end_hour
        )

    def _price_tier(self, hour: int) -> int:
        return int(
            self.config.peak_price_start_hour <= hour < self.config.peak_price_end_hour
        )

    @staticmethod
    def _mode_for_action(action: HVACAction) -> HVACMode:
        if action in (HVACAction.HEAT_LOW, HVACAction.HEAT_HIGH):
            return HVACMode.HEATING
        if action in (HVACAction.COOL_LOW, HVACAction.COOL_HIGH):
            return HVACMode.COOLING
        return HVACMode.OFF

    def _power_for_action(self, action: HVACAction) -> float:
        powers = {
            HVACAction.OFF: 0.0,
            HVACAction.HEAT_LOW: self.config.heating_low_kw,
            HVACAction.HEAT_HIGH: self.config.heating_high_kw,
            HVACAction.COOL_LOW: -self.config.cooling_low_kw,
            HVACAction.COOL_HIGH: -self.config.cooling_high_kw,
        }
        return powers[action]
