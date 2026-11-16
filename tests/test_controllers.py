from thermostat_rl.controllers import RandomController
from thermostat_rl.env import HVACAction
from thermostat_rl.config import ExperimentConfig
from thermostat_rl.controllers import DeadbandController, ScheduledSetbackController

# need to test the two controller classes

def test_RandomController():
    """Test the RandomController class"""

    # create two random controllers
    randomController_A = RandomController(seed = 642)
    randomController_B = RandomController(seed = 642)

    # lists for storing the HVACActions
    HVAC_actions_A = []
    HVAC_actions_B = []

    # need to make sure that although the actions are random
    # they're also repeatable
    for action in range(10):
        HVAC_actions_A.append(randomController_A.select_action({}))
        HVAC_actions_B.append(randomController_B.select_action({}))

    # each controller should have the same actions
    assert HVAC_actions_A == HVAC_actions_B

    # need to check that each HVAC action is possible
    # just have to check one list of actions, since the first assert would've needed to pass
    assert all(isinstance(action, HVACAction) for action in HVAC_actions_A)

def test_DeadbandController_turnOnHeat():
    """Test the DeadbandController class when temperature is below the comfortable temperature range"""
    # use current environment config
    config = ExperimentConfig()
    test_controller = DeadbandController(config)

    # test using temp that is slightly below the comfortable range
    observation = {"indoor_temperature_f": config.comfort_low_f - 1.0}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.HEAT_LOW

def test_DeadbandController_turnOnCooling():
    """Test the DeadbandController class when temperature is above the comfortable temperature range"""
    # use current environment config
    config = ExperimentConfig()
    test_controller = DeadbandController(config)

    # test using temp that is slightly above the comfortable range
    observation = {"indoor_temperature_f": config.comfort_high_f + 1.0}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.COOL_LOW

def test_DeadbandController_turnOff():
    """Test the DeadbandController class when temperature is within the comfortable temperature range"""
    # use current environment config
    config = ExperimentConfig()
    test_controller = DeadbandController(config)

    # test using temp that is within the comfortable range
    comfy_temp_range = config.comfort_high_f - config.comfort_low_f
    observation = {"indoor_temperature_f": config.comfort_low_f + (comfy_temp_range / 2)}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.OFF

def test_DeadbandController_turnOff_atLowestTemp():
    """Test the DeadbandController class when temperature is within the comfortable temperature range,
       at the lowest comfortable temperature"""
    # use current environment config
    config = ExperimentConfig()
    test_controller = DeadbandController(config)

    # test using temp that is within the comfortable range, at the lowest temp
    observation = {"indoor_temperature_f": config.comfort_low_f}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.OFF

def test_DeadbandController_turnOff_atHighestTemp():
    """Test the DeadbandController class when temperature is within the comfortable temperature range,
       at the highest comfortable temperature"""
    # use current environment config
    config = ExperimentConfig()
    test_controller = DeadbandController(config)

    # test using temp that is within the comfortable range, at the highest temp
    observation = {"indoor_temperature_f": config.comfort_high_f}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.OFF

# need to test the Setback controller
def test_ScheduledSetbackController_turnOnHeat_occupied():
    """Test the ScheduledSetbackController class when temperature is below the comfortable temperature range when environment is occupied"""
    # use current environment config
    comfort_setback_f = 4.0
    config = ExperimentConfig()
    test_controller = ScheduledSetbackController(config, comfort_setback_f = comfort_setback_f)

    # test using temp that is slightly below the comfortable range
    observation = {"indoor_temperature_f": config.comfort_low_f - 1.0, "occupancy": 1}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.HEAT_LOW

def test_ScheduledSetbackController_dont_turnOnHeat_away():
    """Test the ScheduledSetbackController class when temperature is within the eco comfortable temperature range when environment is not occupied"""
    # use current environment config
    comfort_setback_f = 4.0
    config = ExperimentConfig()
    test_controller = ScheduledSetbackController(config, comfort_setback_f = comfort_setback_f)

    # test using temp that is within the extended eco comfortable range
    observation = {"indoor_temperature_f": config.comfort_low_f - 1.0, "occupancy": 0}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.OFF

def test_ScheduledSetbackController_do_turnOnHeat_away():
    """Test the ScheduledSetbackController class when temperature is below the comfortable temperature range when environment is not occupied"""
    # use current environment config
    comfort_setback_f = 4.0
    config = ExperimentConfig()
    test_controller = ScheduledSetbackController(config, comfort_setback_f = comfort_setback_f)

    # test using temp that is slightly below the extended eco comfortable range
    observation = {"indoor_temperature_f": config.comfort_low_f - comfort_setback_f - 1.0, "occupancy": 0}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.HEAT_LOW

# test ScheduleSetbackController cooling
def test_ScheduledSetbackController_turnOnCooling_occupied():
    """Test the ScheduledSetbackController class when temperature is above the comfortable temperature range when environment is occupied"""
    # use current environment config
    comfort_setback_f = 4.0
    config = ExperimentConfig()
    test_controller = ScheduledSetbackController(config, comfort_setback_f = comfort_setback_f)

    # test using temp that is slightly above the comfortable range
    observation = {"indoor_temperature_f": config.comfort_high_f + 1.0, "occupancy": 1}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.COOL_LOW

def test_ScheduledSetbackController_dont_turnOnCooling_away():
    """Test the ScheduledSetbackController class when temperature is within the eco comfortable temperature range when environment is not occupied"""
    # use current environment config
    comfort_setback_f = 4.0
    config = ExperimentConfig()
    test_controller = ScheduledSetbackController(config, comfort_setback_f = comfort_setback_f)

    # test using temp that is within the extended eco comfortable range
    observation = {"indoor_temperature_f": config.comfort_high_f + 1.0, "occupancy": 0}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.OFF

def test_ScheduledSetbackController_do_turnOnCooling_away():
    """Test the ScheduledSetbackController class when temperature is above the extended eco comfortable temperature range when environment is not occupied"""
    # use current environment config
    comfort_setback_f = 4.0
    config = ExperimentConfig()
    test_controller = ScheduledSetbackController(config, comfort_setback_f = comfort_setback_f)

    # test using temp that is slightly above the extended eco comfortable range
    observation = {"indoor_temperature_f": config.comfort_high_f + comfort_setback_f + 1.0, "occupancy": 0}
    HVAC_action = test_controller.select_action(observation)

    # check to see if the correct action was chosen
    assert HVAC_action == HVACAction.COOL_LOW