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
        with patch.dict(
            os.environ, {"THERMOSTAT_SEED": "many"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                ExperimentConfig.from_env()

    def test_rejects_negative_seed(self) -> None:
        with patch.dict(os.environ, {"THERMOSTAT_SEED": "-1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                ExperimentConfig.from_env()

    def test_rejects_non_finite_reward_weight(self) -> None:
        with patch.dict(
            os.environ, {"THERMOSTAT_COST_WEIGHT": "nan"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "must be finite"):
                ExperimentConfig.from_env()


if __name__ == "__main__":
    unittest.main()
