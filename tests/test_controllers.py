import numpy as np
import pytest

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.controllers import (
    DeadbandController,
    RandomController,
    ScheduledSetbackController,
)
from thermostat_rl.env import HVACAction


def observation(temperature_f: float, occupancy: int = 1) -> dict[str, object]:
    return {
        "indoor_temperature_f": np.asarray([temperature_f], dtype=np.float32),
        "occupancy": occupancy,
    }


def test_random_controller_is_seeded_and_returns_valid_actions() -> None:
    first = RandomController(seed=642)
    second = RandomController(seed=642)

    first_actions = [first.select_action({}) for _ in range(20)]
    second_actions = [second.select_action({}) for _ in range(20)]

    assert first_actions == second_actions
    assert all(isinstance(action, HVACAction) for action in first_actions)
    assert set(first_actions).issubset(set(HVACAction))


@pytest.mark.parametrize(
    ("temperature_delta", "expected_action"),
    [
        (-0.1, HVACAction.HEAT_LOW),
        (0.0, HVACAction.OFF),
        (4.0, HVACAction.OFF),
        (8.0, HVACAction.OFF),
        (8.1, HVACAction.COOL_LOW),
    ],
)
def test_deadband_controller_actions_at_comfort_boundaries(
    temperature_delta: float, expected_action: HVACAction
) -> None:
    config = ExperimentConfig()
    controller = DeadbandController(config)
    temperature = config.comfort_low_f + temperature_delta

    assert controller.select_action(observation(temperature)) == expected_action


def test_scheduled_setback_widens_band_only_when_unoccupied() -> None:
    config = ExperimentConfig()
    controller = ScheduledSetbackController(config, comfort_setback_f=4.0)
    temperature = config.comfort_low_f - 2.0

    assert (
        controller.select_action(observation(temperature, occupancy=0))
        == HVACAction.OFF
    )
    assert (
        controller.select_action(observation(temperature, occupancy=1))
        == HVACAction.HEAT_LOW
    )


@pytest.mark.parametrize("setback", [-1.0, float("nan"), float("inf")])
def test_scheduled_setback_rejects_invalid_values(setback: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        ScheduledSetbackController(ExperimentConfig(), comfort_setback_f=setback)
