from thermostat_rl.env import ThermostatEnv
from thermostat_rl.config import ExperimentConfig
from thermostat_rl.controllers import DeadbandController, RandomController
from thermostat_rl.evaluation import evaluate_controller, controller_baselines

# need to test accuracy of evaluation metrics
def test_metrics_evaluate_controller():
    """Test the evaluation metrics for controller"""
    # create controller using config
    config = ExperimentConfig()
    environment = ThermostatEnv(config = config)
    test_controller = DeadbandController(config)

    # calculate the metrics for one episode
    evaluation_results = evaluate_controller(environment, test_controller, seed = 642)

    # check that correct values were returned
    assert "seed" in evaluation_results
    assert "energy_cost_usd" in evaluation_results
    assert "comfort_degree_hours" in evaluation_results
    assert "occupied_comfort_percentage" in evaluation_results
    assert "HVAC_mode_changes" in evaluation_results

    # check that the values are reasonable
    assert evaluation_results["energy_cost_usd"] >= 0
    assert evaluation_results["comfort_degree_hours"] >= 0
    assert evaluation_results["occupied_comfort_percentage"] >= 0
    assert evaluation_results["occupied_comfort_percentage"] <= 100
    assert evaluation_results["HVAC_mode_changes"] >= 0

    # make sure arrival temp deficit is also included
    assert "arrival_temperature_deficit" in evaluation_results
    assert evaluation_results["arrival_temperature_deficit"] >= 0

# test the baselines
def test_controller_baselines():
    """Run each baseline in the same scenarios with the same evaluation seeds"""
    # create controllers using config
    config = ExperimentConfig()
    environment = ThermostatEnv(config = config)
    seed_list = [642, 689, 2026, 1115]

    # run each controller with the same seed_list
    baseline_eval_results = controller_baselines(environment, config, seed_list)

    # results should add one entry per controller for each number in the seed_list
    for seed_i in seed_list:
        seed_i_results = [result for result in baseline_eval_results if result["seed"] == seed_i]
        #assert len(seed_i_results) == 2

        # update test because added ScheduleSetbackController to baseline
        assert len(seed_i_results) == 3