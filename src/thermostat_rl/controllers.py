import numpy as np
from thermostat_rl.config import ExperimentConfig
from thermostat_rl.env import HVACAction, ThermostatObservation

# our project requires three controller classes for choosing what actions the thermostat should make

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

# need to implement scheduled setback
# is 4 degree difference good for "eco" non-occupied hours?
# off temperature range would go from 64 - 80 F
class ScheduledSetbackController:
    """Thermostat controller that is not as stringent with the comfortable range when there are no occupants"""
    def __init__(self, config: ExperimentConfig, comfort_setback_f: float = 4.0) -> None:
        # save the config and adjustable temperature range
        self.config = config
        self.comfort_setback_f = comfort_setback_f

    def select_action(self, observation: ThermostatObservation) -> HVACAction:
        """Based on the current indoor temperature and occupancy, choose HVAC action"""
        # get the current indoor temp and occupancy
        curr_indoor_temp = observation["indoor_temperature_f"]
        occupancy = observation["occupancy"]

        # first need to check if there is an occupant in the environment
        if occupancy == 1: 
            # someone is home, so need to stay within the comfortable range
            lowest_temp = self.config.comfort_low_f
            highest_temp = self.config.comfort_high_f

        else:
            # no one is home
            lowest_temp = self.config.comfort_low_f - self.comfort_setback_f
            highest_temp = self.config.comfort_high_f + self.comfort_setback_f

        # after potentially updating acceptable temperature range, continue like deadband controller
        # if temperature is too low, need to turn on the heat
        if curr_indoor_temp < lowest_temp:
            return HVACAction.HEAT_LOW

        # if temperature is too high, need to turn on the cooling
        elif curr_indoor_temp > highest_temp:
            return HVACAction.COOL_LOW

        # if temperature is within the comfortable range, turn off
        else:
            return HVACAction.OFF