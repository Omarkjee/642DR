from dataclasses import replace
import unittest

from gymnasium.utils.env_checker import check_env

from thermostat_rl.config import ExperimentConfig
from thermostat_rl.env import HVACAction, HVACMode, ThermostatEnv
from thermostat_rl.state import discretize_observation, reachable_state_upper_bound


class ThermostatEnvTests(unittest.TestCase):
    def make_config(self, **overrides: object) -> ExperimentConfig:
        values: dict[str, object] = {
            "initial_temperature_std_f": 0.0,
            "temperature_noise_std_f": 0.0,
            **overrides,
        }
        return replace(ExperimentConfig(), **values)

    def test_reset_and_step_follow_gymnasium_contract(self) -> None:
        env = ThermostatEnv(self.make_config(steps_per_episode=1))
        observation, reset_info = env.reset(seed=1)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(reset_info["time_step_hours"], 1.0)
        self.assertEqual(observation["hour"], 0)
        self.assertEqual(observation["hvac_mode"], HVACMode.OFF)

        next_observation, reward, terminated, truncated, info = env.step(HVACAction.OFF)

        self.assertTrue(env.observation_space.contains(next_observation))
        self.assertIsInstance(reward, float)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["step"], 0)
        self.assertIn("energy_cost_usd", info)
        self.assertIn("comfort_violation_degree_hours", info)

    def test_default_week_terminates_after_168_valid_steps(self) -> None:
        env = ThermostatEnv()
        observation, _ = env.reset(seed=42)

        for step in range(168):
            observation, _, terminated, truncated, info = env.step(HVACAction.OFF)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertFalse(truncated)
            self.assertEqual(terminated, step == 167)

        self.assertEqual(info["step"], 167)

    def test_gymnasium_environment_checker(self) -> None:
        check_env(
            ThermostatEnv(self.make_config(steps_per_episode=8)),
            skip_render_check=True,
        )

    def test_heating_and_cooling_have_expected_direction(self) -> None:
        config = self.make_config()
        env = ThermostatEnv(config)

        observation, _ = env.reset(seed=1, options={"initial_temperature_f": 70.0})
        passive_change = (
            config.time_step_hours
            / config.thermal_capacitance_kwh_per_f
            * (
                (observation["outdoor_temperature_f"] - 70.0)
                / config.thermal_resistance_f_per_kw
            )
        )
        heated, *_ = env.step(HVACAction.HEAT_LOW)
        expected_heating = passive_change + (
            3.0 * config.time_step_hours / config.thermal_capacitance_kwh_per_f
        )
        self.assertAlmostEqual(
            heated["indoor_temperature_f"] - 70.0, expected_heating, places=5
        )

        env.reset(seed=1, options={"initial_temperature_f": 70.0})
        cooled, *_ = env.step(HVACAction.COOL_LOW)
        expected_cooling = passive_change + (
            -3.0 * config.time_step_hours / config.thermal_capacitance_kwh_per_f
        )
        self.assertAlmostEqual(
            cooled["indoor_temperature_f"] - 70.0, expected_cooling, places=5
        )
        self.assertLess(cooled["indoor_temperature_f"], heated["indoor_temperature_f"])

    def test_passive_drift_matches_rc_equation(self) -> None:
        config = self.make_config()
        env = ThermostatEnv(config)
        observation, _ = env.reset(seed=2, options={"initial_temperature_f": 72.0})

        next_observation, _, _, _, info = env.step(HVACAction.OFF)
        expected_change = (
            config.time_step_hours
            / config.thermal_capacitance_kwh_per_f
            * (
                (info["outdoor_temperature_f"] - 72.0)
                / config.thermal_resistance_f_per_kw
            )
        )

        self.assertAlmostEqual(
            next_observation["indoor_temperature_f"]
            - observation["indoor_temperature_f"],
            expected_change,
            places=5,
        )
        self.assertLess(next_observation["indoor_temperature_f"], 72.0)

    def test_occupancy_aware_comfort_penalty_and_reward(self) -> None:
        config = self.make_config(
            steps_per_episode=1,
            comfort_low_f=68.0,
            comfort_weight=5.0,
            cost_weight=0.0,
            switch_weight=0.0,
        )
        env = ThermostatEnv(config)
        env.reset(seed=1, options={"initial_temperature_f": 70.0})

        _, reward, _, _, info = env.step(HVACAction.OFF)

        self.assertEqual(info["occupancy"], 1)
        expected_violation = max(
            config.comfort_low_f - info["indoor_temperature_f"], 0.0
        )
        self.assertAlmostEqual(
            info["comfort_violation_degree_hours"], expected_violation
        )
        self.assertFalse(info["occupied_comfortable"])
        self.assertAlmostEqual(reward, -config.comfort_weight * expected_violation)

    def test_unoccupied_comfort_does_not_reduce_reward(self) -> None:
        config = self.make_config(comfort_weight=5.0)
        env = ThermostatEnv(config)
        env.reset(seed=1, options={"initial_temperature_f": 60.0})
        for _ in range(8):
            env.step(HVACAction.OFF)

        _, reward, _, _, info = env.step(HVACAction.OFF)

        self.assertEqual(info["occupancy"], 0)
        self.assertEqual(info["comfort_violation_degree_hours"], 0.0)
        self.assertEqual(reward, 0.0)

    def test_energy_cost_uses_power_duration_and_time_of_use_price(self) -> None:
        config = self.make_config(
            cost_weight=2.0, comfort_weight=0.0, switch_weight=0.0
        )
        env = ThermostatEnv(config)
        env.reset(seed=1, options={"initial_temperature_f": 70.0})
        for _ in range(16):
            env.step(HVACAction.OFF)

        _, reward, _, _, info = env.step(HVACAction.HEAT_LOW)

        self.assertEqual(info["price_tier"], 1)
        self.assertEqual(info["price_usd_per_kwh"], config.peak_price_usd_per_kwh)
        self.assertEqual(info["energy_kwh"], 3.0)
        self.assertAlmostEqual(
            info["energy_cost_usd"], 3.0 * config.peak_price_usd_per_kwh
        )
        self.assertAlmostEqual(reward, -2.0 * info["energy_cost_usd"])

    def test_mode_switches_count_mode_changes_not_power_levels(self) -> None:
        config = self.make_config(
            cost_weight=0.0, comfort_weight=0.0, switch_weight=0.25
        )
        env = ThermostatEnv(config)
        env.reset(seed=1)

        _, first_reward, _, _, first = env.step(HVACAction.HEAT_LOW)
        _, second_reward, _, _, second = env.step(HVACAction.HEAT_HIGH)
        _, third_reward, _, _, third = env.step(HVACAction.COOL_LOW)

        self.assertTrue(first["mode_changed"])
        self.assertFalse(second["mode_changed"])
        self.assertTrue(third["mode_changed"])
        self.assertEqual(third["hvac_mode"], HVACMode.COOLING)
        self.assertEqual(first_reward, -config.switch_weight)
        self.assertEqual(second_reward, 0.0)
        self.assertEqual(third_reward, -config.switch_weight)

    def test_weekday_and_weekend_occupancy_and_peak_schedule(self) -> None:
        env = ThermostatEnv(self.make_config())
        observation, _ = env.reset(seed=1)
        self.assertEqual(observation["occupancy"], 1)

        for _ in range(8):
            observation, _, _, _, _ = env.step(HVACAction.OFF)
        self.assertEqual(observation["hour"], 8)
        self.assertEqual(observation["occupancy"], 0)

        for _ in range(112):
            observation, _, _, _, _ = env.step(HVACAction.OFF)
        self.assertEqual(observation["day_type"], 1)
        self.assertEqual(observation["occupancy"], 1)

        for _ in range(16):
            observation, _, _, _, _ = env.step(HVACAction.OFF)
        self.assertEqual(observation["hour"], 16)
        self.assertEqual(observation["price_tier"], 1)

    def test_outdoor_temperature_follows_daily_peak_and_trough(self) -> None:
        config = self.make_config()
        env = ThermostatEnv(config)
        observation, _ = env.reset(seed=1)

        for _ in range(config.outdoor_temperature_peak_hour):
            observation, _, _, _, _ = env.step(HVACAction.OFF)
        self.assertAlmostEqual(
            float(observation["outdoor_temperature_f"]),
            config.outdoor_temperature_mean_f + config.outdoor_temperature_amplitude_f,
            places=5,
        )

        for _ in range(12):
            observation, _, _, _, _ = env.step(HVACAction.OFF)
        self.assertAlmostEqual(
            float(observation["outdoor_temperature_f"]),
            config.outdoor_temperature_mean_f - config.outdoor_temperature_amplitude_f,
            places=5,
        )

    def test_seeded_runs_are_reproducible(self) -> None:
        config = self.make_config(steps_per_episode=4, temperature_noise_std_f=0.2)
        first_env = ThermostatEnv(config)
        second_env = ThermostatEnv(config)
        first_observation, _ = first_env.reset(seed=27)
        second_observation, _ = second_env.reset(seed=27)
        self.assertEqual(first_observation, second_observation)

        for action in (HVACAction.HEAT_LOW, HVACAction.OFF, HVACAction.COOL_HIGH):
            first = first_env.step(action)
            second = second_env.step(action)
            self.assertEqual(first, second)

    def test_discretized_state_and_state_count(self) -> None:
        config = self.make_config()
        env = ThermostatEnv(config)
        observation, _ = env.reset(
            seed=1,
            options={"initial_temperature_f": config.indoor_temperature_bin_min_f},
        )

        state = discretize_observation(observation, config)

        self.assertEqual(state, (0, 12, 0, 1, 0, HVACMode.OFF, 0))
        self.assertEqual(reachable_state_upper_bound(config), 74_880)

    def test_step_requires_reset_and_rejects_steps_after_termination(self) -> None:
        env = ThermostatEnv(self.make_config(steps_per_episode=1))
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(HVACAction.OFF)

        env.reset(seed=1)
        env.step(HVACAction.OFF)
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            env.step(HVACAction.OFF)


if __name__ == "__main__":
    unittest.main()
