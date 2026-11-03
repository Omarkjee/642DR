"""State discretization helpers for the tabular controller."""

from math import ceil, floor
from typing import TypeAlias

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.env import ThermostatObservation

StateKey: TypeAlias = tuple[int, int, int, int, int, int, int]


def _temperature_bin(
    temperature_f: float, minimum_f: float, maximum_f: float, width_f: float
) -> int:
    bin_count = ceil((maximum_f - minimum_f) / width_f)
    return min(max(floor((temperature_f - minimum_f) / width_f), 0), bin_count - 1)


def discretize_observation(
    observation: ThermostatObservation, config: ExperimentConfig | None = None
) -> StateKey:
    """Map an observation to (indoor, outdoor, hour, occupancy, day, mode, price)."""
    config = config or ExperimentConfig()
    return (
        _temperature_bin(
            float(observation["indoor_temperature_f"]),
            config.indoor_temperature_bin_min_f,
            config.indoor_temperature_bin_max_f,
            config.indoor_temperature_bin_width_f,
        ),
        _temperature_bin(
            float(observation["outdoor_temperature_f"]),
            config.outdoor_temperature_bin_min_f,
            config.outdoor_temperature_bin_max_f,
            config.outdoor_temperature_bin_width_f,
        ),
        observation["hour"],
        observation["occupancy"],
        observation["day_type"],
        observation["hvac_mode"],
        observation["price_tier"],
    )


def reachable_state_upper_bound(config: ExperimentConfig | None = None) -> int:
    """Upper bound states under schedules that fix occupancy and price by hour."""
    config = config or ExperimentConfig()
    indoor_bins = ceil(
        (config.indoor_temperature_bin_max_f - config.indoor_temperature_bin_min_f)
        / config.indoor_temperature_bin_width_f
    )
    outdoor_bins = ceil(
        (config.outdoor_temperature_bin_max_f - config.outdoor_temperature_bin_min_f)
        / config.outdoor_temperature_bin_width_f
    )
    schedule_contexts = 24 * 2
    hvac_modes = 3
    return indoor_bins * outdoor_bins * schedule_contexts * hvac_modes
