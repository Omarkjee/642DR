import numpy as np
from thermostat_rl.config import ExperimentConfig
from thermostat_rl.env import HVACAction, ThermostatObservation

# our projecct requires two controller classes for choosing what actions the thermostat should make

class RandomController:
    """Uniformly random thermostat actions"""

    def __init__(self, seed: int = 642) -> None:
        # choose random number
        # repeatable based on seed value
        self.random_num = np.random.default_rng(seed)

    def select_action(self, observation: ThermostatObservation) -> HVACAction:
        """Given current observation, what action should the controller take?"""
        # since random, don't use observation
        del observation

        # instead, choose a random action
        random_action = self.random_num.integers(0, len(HVACAction))

        # return the random HVAC action
        return HVACAction(random_action)

class DeadbandController:
    """Thermostat controller that takes into account a comfortable temperature range"""
    def __init__(self, config: ExperimentConfig) -> None:
        # save the comfortable temperature range values
        # project uses configuration as the environment
        self.config = config

    def select_action(self, observation: ThermostatObservation) -> HVACAction:
        """Based on the current indoor temperature, choose HVAC action"""
        # get the current indoor temp
        curr_indoor_temp = observation["indoor_temperature_f"]

        # if temperature is too low, need to turn on the heat
        if curr_indoor_temp < self.config.comfort_low_f:
            return HVACAction.HEAT_LOW

        # if temperature is too high, need to turn on the cooling
        elif curr_indoor_temp > self.config.comfort_high_f:
            return HVACAction.COOL_LOW

        # if temperature is within the comfortable range, turn off
        else:
            return HVACAction.OFF
