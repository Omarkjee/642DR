import os
import unittest
from unittest.mock import patch

from thermostat_rl.config import ExperimentConfig


class ExperimentConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = ExperimentConfig.from_env()

        self.assertEqual(config.seed, 42)
        self.assertEqual(config.episodes, 500)
        self.assertEqual(config.discount_factor, 0.99)

    def test_reads_environment_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "THERMOSTAT_SEED": "7",
                "THERMOSTAT_EPISODES": "20",
                "THERMOSTAT_COMFORT_WEIGHT": "3.5",
            },
            clear=True,
        ):
            config = ExperimentConfig.from_env()

        self.assertEqual(config.seed, 7)
        self.assertEqual(config.episodes, 20)
        self.assertEqual(config.comfort_weight, 3.5)

    def test_rejects_invalid_learning_rate(self) -> None:
        with patch.dict(os.environ, {"THERMOSTAT_LEARNING_RATE": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "LEARNING_RATE"):
                ExperimentConfig.from_env()

    def test_rejects_non_numeric_environment_value(self) -> None:
        with patch.dict(os.environ, {"THERMOSTAT_SEED": "many"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                ExperimentConfig.from_env()

    def test_rejects_negative_seed(self) -> None:
        with patch.dict(os.environ, {"THERMOSTAT_SEED": "-1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                ExperimentConfig.from_env()

    def test_rejects_non_finite_reward_weight(self) -> None:
        with patch.dict(os.environ, {"THERMOSTAT_COST_WEIGHT": "nan"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be finite"):
                ExperimentConfig.from_env()

    def test_reads_thermal_model_environment_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "THERMOSTAT_TIME_STEP_HOURS": "1.0",
                "THERMOSTAT_THERMAL_RESISTANCE_F_PER_KW": "3",
                "THERMOSTAT_COMFORT_LOW_F": "67",
            },
            clear=True,
        ):
            config = ExperimentConfig.from_env()

        self.assertEqual(config.time_step_hours, 1.0)
        self.assertEqual(config.thermal_resistance_f_per_kw, 3.0)
        self.assertEqual(config.comfort_low_f, 67.0)

    def test_rejects_invalid_physical_parameters(self) -> None:
        with patch.dict(os.environ, {"THERMOSTAT_TIME_STEP_HOURS": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "TIME_STEP_HOURS"):
                ExperimentConfig.from_env()

    def test_rejects_non_hourly_time_step(self) -> None:
        with patch.dict(os.environ, {"THERMOSTAT_TIME_STEP_HOURS": "0.5"}, clear=True):
            with self.assertRaisesRegex(ValueError, "schedules are hourly"):
                ExperimentConfig.from_env()

    def test_rejects_invalid_schedule(self) -> None:
        with patch.dict(
            os.environ,
            {"THERMOSTAT_WEEKDAY_AWAY_START_HOUR": "20"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "away start hour"):
                ExperimentConfig.from_env()

    def test_rejects_invalid_comfort_band(self) -> None:
        with patch.dict(
            os.environ,
            {"THERMOSTAT_COMFORT_LOW_F": "80", "THERMOSTAT_COMFORT_HIGH_F": "70"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "comfort band"):
                ExperimentConfig.from_env()


if __name__ == "__main__":
    unittest.main()
