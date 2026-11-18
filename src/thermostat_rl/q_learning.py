from thermostat_rl.env import HVACAction
from thermostat_rl.config import ExperimentConfig
from thermostat_rl.state import StateKey
import numpy as np

# for tabular Q-learning, we need a new class
class TabularQLearning:
    """Tabular Q-Learning agent, used for controlling the thermostat"""
    def __init__(self, config: ExperimentConfig) -> None:
        # save the config and repeatable random number
        self.config = config
        self.random_num = np.random.default_rng(config.seed)

        # save important variables and initialize table
        self.epsilon = config.epsilon_start
        self.q_table = {}

    def state_qValues(self, state: StateKey) -> np.ndarray:
        """Get Q-Values for given state"""
        # if state not in Q table yet, all actions will be zero
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(HVACAction))
        return self.q_table[state]

    def select_action(self, state:StateKey) -> HVACAction:
        """Based on epsilon-greedy selection, choose HVAC action"""
        # get the current q_values
        q_values = self.state_qValues(state)

        # if random num is less than epsilon, choose a random HVAC action
        if self.random_num.random() < self.epsilon:
            random_action = self.random_num.integers(0, len(HVACAction))
            return HVACAction(random_action)

        # otherwise, choose best greedy action
        best_greedy_action = np.argmax(q_values)
        return HVACAction(best_greedy_action)
