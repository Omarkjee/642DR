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
    """Reproducible settings for the thermal model, reward, and experiments."""

    seed: int = 42
    episodes: int = 500
    steps_per_episode: int = 168
    time_step_hours: float = 1.0
    thermal_resistance_f_per_kw: float = 2.0
    thermal_capacitance_kwh_per_f: float = 5.0
    heating_low_kw: float = 3.0
    heating_high_kw: float = 6.0
    cooling_low_kw: float = 3.0
    cooling_high_kw: float = 6.0
    initial_temperature_mean_f: float = 72.0
    initial_temperature_std_f: float = 1.0
    outdoor_temperature_mean_f: float = 50.0
    outdoor_temperature_amplitude_f: float = 12.0
    outdoor_temperature_peak_hour: int = 15
    weekday_away_start_hour: int = 8
    weekday_away_end_hour: int = 18
    peak_price_start_hour: int = 16
    peak_price_end_hour: int = 21
    off_peak_price_usd_per_kwh: float = 0.12
    peak_price_usd_per_kwh: float = 0.36
    comfort_low_f: float = 68.0
    comfort_high_f: float = 76.0
    temperature_noise_std_f: float = 0.05
    indoor_temperature_bin_width_f: float = 2.0
    indoor_temperature_bin_min_f: float = 40.0
    indoor_temperature_bin_max_f: float = 92.0
    outdoor_temperature_bin_width_f: float = 5.0
    outdoor_temperature_bin_min_f: float = -20.0
    outdoor_temperature_bin_max_f: float = 80.0
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
            time_step_hours=_read_float(
                "THERMOSTAT_TIME_STEP_HOURS", cls.time_step_hours
            ),
            thermal_resistance_f_per_kw=_read_float(
                "THERMOSTAT_THERMAL_RESISTANCE_F_PER_KW",
                cls.thermal_resistance_f_per_kw,
            ),
            thermal_capacitance_kwh_per_f=_read_float(
                "THERMOSTAT_THERMAL_CAPACITANCE_KWH_PER_F",
                cls.thermal_capacitance_kwh_per_f,
            ),
            heating_low_kw=_read_float("THERMOSTAT_HEATING_LOW_KW", cls.heating_low_kw),
            heating_high_kw=_read_float(
                "THERMOSTAT_HEATING_HIGH_KW", cls.heating_high_kw
            ),
            cooling_low_kw=_read_float("THERMOSTAT_COOLING_LOW_KW", cls.cooling_low_kw),
            cooling_high_kw=_read_float(
                "THERMOSTAT_COOLING_HIGH_KW", cls.cooling_high_kw
            ),
            initial_temperature_mean_f=_read_float(
                "THERMOSTAT_INITIAL_TEMPERATURE_MEAN_F",
                cls.initial_temperature_mean_f,
            ),
            initial_temperature_std_f=_read_float(
                "THERMOSTAT_INITIAL_TEMPERATURE_STD_F",
                cls.initial_temperature_std_f,
            ),
            outdoor_temperature_mean_f=_read_float(
                "THERMOSTAT_OUTDOOR_TEMPERATURE_MEAN_F",
                cls.outdoor_temperature_mean_f,
            ),
            outdoor_temperature_amplitude_f=_read_float(
                "THERMOSTAT_OUTDOOR_TEMPERATURE_AMPLITUDE_F",
                cls.outdoor_temperature_amplitude_f,
            ),
            outdoor_temperature_peak_hour=_read_int(
                "THERMOSTAT_OUTDOOR_TEMPERATURE_PEAK_HOUR",
                cls.outdoor_temperature_peak_hour,
            ),
            weekday_away_start_hour=_read_int(
                "THERMOSTAT_WEEKDAY_AWAY_START_HOUR",
                cls.weekday_away_start_hour,
            ),
            weekday_away_end_hour=_read_int(
                "THERMOSTAT_WEEKDAY_AWAY_END_HOUR",
                cls.weekday_away_end_hour,
            ),
            peak_price_start_hour=_read_int(
                "THERMOSTAT_PEAK_PRICE_START_HOUR", cls.peak_price_start_hour
            ),
            peak_price_end_hour=_read_int(
                "THERMOSTAT_PEAK_PRICE_END_HOUR", cls.peak_price_end_hour
            ),
            off_peak_price_usd_per_kwh=_read_float(
                "THERMOSTAT_OFF_PEAK_PRICE_USD_PER_KWH",
                cls.off_peak_price_usd_per_kwh,
            ),
            peak_price_usd_per_kwh=_read_float(
                "THERMOSTAT_PEAK_PRICE_USD_PER_KWH",
                cls.peak_price_usd_per_kwh,
            ),
            comfort_low_f=_read_float("THERMOSTAT_COMFORT_LOW_F", cls.comfort_low_f),
            comfort_high_f=_read_float("THERMOSTAT_COMFORT_HIGH_F", cls.comfort_high_f),
            temperature_noise_std_f=_read_float(
                "THERMOSTAT_TEMPERATURE_NOISE_STD_F",
                cls.temperature_noise_std_f,
            ),
            indoor_temperature_bin_width_f=_read_float(
                "THERMOSTAT_INDOOR_TEMPERATURE_BIN_WIDTH_F",
                cls.indoor_temperature_bin_width_f,
            ),
            indoor_temperature_bin_min_f=_read_float(
                "THERMOSTAT_INDOOR_TEMPERATURE_BIN_MIN_F",
                cls.indoor_temperature_bin_min_f,
            ),
            indoor_temperature_bin_max_f=_read_float(
                "THERMOSTAT_INDOOR_TEMPERATURE_BIN_MAX_F",
                cls.indoor_temperature_bin_max_f,
            ),
            outdoor_temperature_bin_width_f=_read_float(
                "THERMOSTAT_OUTDOOR_TEMPERATURE_BIN_WIDTH_F",
                cls.outdoor_temperature_bin_width_f,
            ),
            outdoor_temperature_bin_min_f=_read_float(
                "THERMOSTAT_OUTDOOR_TEMPERATURE_BIN_MIN_F",
                cls.outdoor_temperature_bin_min_f,
            ),
            outdoor_temperature_bin_max_f=_read_float(
                "THERMOSTAT_OUTDOOR_TEMPERATURE_BIN_MAX_F",
                cls.outdoor_temperature_bin_max_f,
            ),
            learning_rate=_read_float("THERMOSTAT_LEARNING_RATE", cls.learning_rate),
            discount_factor=_read_float(
                "THERMOSTAT_DISCOUNT_FACTOR", cls.discount_factor
            ),
            epsilon_start=_read_float("THERMOSTAT_EPSILON_START", cls.epsilon_start),
            epsilon_min=_read_float("THERMOSTAT_EPSILON_MIN", cls.epsilon_min),
            epsilon_decay=_read_float("THERMOSTAT_EPSILON_DECAY", cls.epsilon_decay),
            cost_weight=_read_float("THERMOSTAT_COST_WEIGHT", cls.cost_weight),
            comfort_weight=_read_float("THERMOSTAT_COMFORT_WEIGHT", cls.comfort_weight),
            switch_weight=_read_float("THERMOSTAT_SWITCH_WEIGHT", cls.switch_weight),
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
        positive_settings = {
            "THERMOSTAT_TIME_STEP_HOURS": self.time_step_hours,
            "THERMOSTAT_THERMAL_RESISTANCE_F_PER_KW": (
                self.thermal_resistance_f_per_kw
            ),
            "THERMOSTAT_THERMAL_CAPACITANCE_KWH_PER_F": (
                self.thermal_capacitance_kwh_per_f
            ),
            "THERMOSTAT_HEATING_LOW_KW": self.heating_low_kw,
            "THERMOSTAT_HEATING_HIGH_KW": self.heating_high_kw,
            "THERMOSTAT_COOLING_LOW_KW": self.cooling_low_kw,
            "THERMOSTAT_COOLING_HIGH_KW": self.cooling_high_kw,
            "THERMOSTAT_INDOOR_TEMPERATURE_BIN_WIDTH_F": (
                self.indoor_temperature_bin_width_f
            ),
            "THERMOSTAT_OUTDOOR_TEMPERATURE_BIN_WIDTH_F": (
                self.outdoor_temperature_bin_width_f
            ),
        }
        for name, value in positive_settings.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.time_step_hours != 1.0:
            raise ValueError(
                "THERMOSTAT_TIME_STEP_HOURS must be 1.0 while schedules are hourly"
            )
        numeric_settings = {
            **positive_settings,
            "THERMOSTAT_INITIAL_TEMPERATURE_MEAN_F": (self.initial_temperature_mean_f),
            "THERMOSTAT_INITIAL_TEMPERATURE_STD_F": (self.initial_temperature_std_f),
            "THERMOSTAT_OUTDOOR_TEMPERATURE_MEAN_F": (self.outdoor_temperature_mean_f),
            "THERMOSTAT_OUTDOOR_TEMPERATURE_AMPLITUDE_F": (
                self.outdoor_temperature_amplitude_f
            ),
            "THERMOSTAT_OFF_PEAK_PRICE_USD_PER_KWH": (self.off_peak_price_usd_per_kwh),
            "THERMOSTAT_PEAK_PRICE_USD_PER_KWH": (self.peak_price_usd_per_kwh),
            "THERMOSTAT_COMFORT_LOW_F": self.comfort_low_f,
            "THERMOSTAT_COMFORT_HIGH_F": self.comfort_high_f,
            "THERMOSTAT_TEMPERATURE_NOISE_STD_F": (self.temperature_noise_std_f),
            "THERMOSTAT_INDOOR_TEMPERATURE_BIN_MIN_F": (
                self.indoor_temperature_bin_min_f
            ),
            "THERMOSTAT_INDOOR_TEMPERATURE_BIN_MAX_F": (
                self.indoor_temperature_bin_max_f
            ),
            "THERMOSTAT_OUTDOOR_TEMPERATURE_BIN_MIN_F": (
                self.outdoor_temperature_bin_min_f
            ),
            "THERMOSTAT_OUTDOOR_TEMPERATURE_BIN_MAX_F": (
                self.outdoor_temperature_bin_max_f
            ),
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
        if self.initial_temperature_std_f < 0:
            raise ValueError(
                "THERMOSTAT_INITIAL_TEMPERATURE_STD_F must be non-negative"
            )
        if self.outdoor_temperature_amplitude_f < 0:
            raise ValueError(
                "THERMOSTAT_OUTDOOR_TEMPERATURE_AMPLITUDE_F must be non-negative"
            )
        if self.temperature_noise_std_f < 0:
            raise ValueError("THERMOSTAT_TEMPERATURE_NOISE_STD_F must be non-negative")
        if self.heating_low_kw > self.heating_high_kw:
            raise ValueError(
                "THERMOSTAT_HEATING_LOW_KW must not exceed HEATING_HIGH_KW"
            )
        if self.cooling_low_kw > self.cooling_high_kw:
            raise ValueError(
                "THERMOSTAT_COOLING_LOW_KW must not exceed COOLING_HIGH_KW"
            )
        if min(self.off_peak_price_usd_per_kwh, self.peak_price_usd_per_kwh) < 0:
            raise ValueError("electricity prices must be non-negative")
        if self.comfort_low_f >= self.comfort_high_f:
            raise ValueError("comfort band must satisfy COMFORT_LOW_F < COMFORT_HIGH_F")
        if self.indoor_temperature_bin_min_f >= self.indoor_temperature_bin_max_f:
            raise ValueError("indoor temperature bin minimum must be below its maximum")
        if self.outdoor_temperature_bin_min_f >= self.outdoor_temperature_bin_max_f:
            raise ValueError(
                "outdoor temperature bin minimum must be below its maximum"
            )
        hourly_settings = {
            "THERMOSTAT_OUTDOOR_TEMPERATURE_PEAK_HOUR": (
                self.outdoor_temperature_peak_hour
            ),
            "THERMOSTAT_WEEKDAY_AWAY_START_HOUR": self.weekday_away_start_hour,
            "THERMOSTAT_WEEKDAY_AWAY_END_HOUR": self.weekday_away_end_hour,
            "THERMOSTAT_PEAK_PRICE_START_HOUR": self.peak_price_start_hour,
            "THERMOSTAT_PEAK_PRICE_END_HOUR": self.peak_price_end_hour,
        }
        for name, value in hourly_settings.items():
            if not 0 <= value < 24:
                raise ValueError(f"{name} must be in [0, 24)")
        if self.weekday_away_start_hour >= self.weekday_away_end_hour:
            raise ValueError("weekday away start hour must be before end hour")
        if self.peak_price_start_hour >= self.peak_price_end_hour:
            raise ValueError("peak price start hour must be before end hour")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("THERMOSTAT_LEARNING_RATE must be in (0, 1]")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("THERMOSTAT_DISCOUNT_FACTOR must be in [0, 1]")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError(
                "epsilon values must satisfy 0 <= EPSILON_MIN <= " "EPSILON_START <= 1"
            )
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("THERMOSTAT_EPSILON_DECAY must be in (0, 1]")
        if min(self.cost_weight, self.comfort_weight, self.switch_weight) < 0:
            raise ValueError("reward weights must be non-negative")
        if self.cost_weight + self.comfort_weight + self.switch_weight == 0:
            raise ValueError("at least one reward weight must be greater than zero")
