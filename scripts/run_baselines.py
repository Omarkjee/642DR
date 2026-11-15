from thermostat_rl.env import ThermostatEnv
from thermostat_rl.config import ExperimentConfig
from thermostat_rl.controllers import DeadbandController, RandomController
from thermostat_rl.evaluation import evaluate_controller, controller_baselines
import pandas as pd
import csv
import matplotlib.pyplot as plt

# similar to testing baseline, but need to save the results into a table
# create controllers using config
config = ExperimentConfig()
environment = ThermostatEnv(config = config)
seed_list = [642, 689, 2026, 1115]

# run each controller with the same seed_list
baseline_eval_results = controller_baselines(environment, config, seed_list)

# save as table and csv
baseline_table = pd.DataFrame(baseline_eval_results)
baseline_table.to_csv("results/baseline_table.csv", index = False)
#print(baseline_table)

# create sanity-check plot
# compare energy cost per controller
baseline_table.boxplot("energy_cost_usd", "controller_type")
plt.title("Baseline Sanity Plot: Energy Cost by Controller")
plt.xlabel("Controller")
plt.ylabel("Energy Cost (USD)")
plt.savefig("results/baseline_plot.png")