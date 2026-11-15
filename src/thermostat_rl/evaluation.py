from thermostat_rl.controllers import DeadbandController, RandomController
from thermostat_rl.env import ThermostatEnv
from thermostat_rl.config import ExperimentConfig

# need to calculate energy cost, comfort degree-hours, occupied comfort percentage, and HVAC mode changes

def evaluate_controller(environment: ThermostatEnv, controller: DeadbandController | RandomController, seed: int = 642) -> dict:
    """Calculate metrics to evaluate controller after one episode"""
    # reset environment, to make sure testing is accurate
    observation, _ = environment.reset(seed = seed)

    # need to calculate energy cost, comfort degree-hours, occupied comfort percentage, and HVAC mode changes
    # save each metric into its own variable
    energy_cost = 0.0
    comfort_degree_hours = 0.0
    occupied_steps = 0
    comfortable_occupied_steps = 0
    occupied_comfort_percentage = 0.0
    HVAC_mode_changes = 0

    # episode will continue until the environment is either terminated or truncated
    bool_terminated = False
    bool_truncated = False
    while (bool_terminated == False and bool_truncated == False):
        # select an action
        action_selected = controller.select_action(observation)

        # use the selected action
        observation, reward, bool_terminated, bool_truncated, info = environment.step(action_selected)

        # update metric variables
        energy_cost += info["energy_cost_usd"]
        comfort_degree_hours += info["comfort_violation_degree_hours"]
        if info["mode_changed"] == True:
            HVAC_mode_changes += 1

        # check occupancy comfort
        if info["occupancy"] == True:
            occupied_steps += 1

            # if occupied, was temp within the comfortable range
            if info["occupied_comfortable"] == True:
                comfortable_occupied_steps += 1

    # after episode ends, still need to calculate one more metric
    # calc occupied comfort percentage
    if occupied_steps > 0: # don't want to divide by zero
        occupied_comfort_percentage = (comfortable_occupied_steps / occupied_steps) * 100

    # return metric results
    evaluation_metrics_dict = {"seed": seed, "energy_cost_usd": energy_cost, "comfort_degree_hours": comfort_degree_hours,
               "occupied_comfort_percentage": occupied_comfort_percentage, "HVAC_mode_changes": HVAC_mode_changes}
    return evaluation_metrics_dict

# need to run baseline for random action and deadband controllers
# during testing, save a table to compare results
def controller_baselines(environment: ThermostatEnv, config: ExperimentConfig, seed_list: list[int]) -> list[dict]:
    """Run the baseline for list of random action and deadband controllers"""
    # running on list of different seeds to see reproducibility
    baseline_results = []
    for seed_i in seed_list:
        # create the controllers
        deadband_controller = DeadbandController(config)
        random_action_controller = RandomController(seed = seed_i)

        # save the results for each controller
        randomController_results = evaluate_controller(environment, random_action_controller, seed = seed_i)
        randomController_results["controller_type"] = "RandomController"
        baseline_results.append(randomController_results)

        deadbandController_results = evaluate_controller(environment, deadband_controller, seed = seed_i)
        deadbandController_results["controller_type"] = "DeadbandController"
        baseline_results.append(deadbandController_results)

    return baseline_results