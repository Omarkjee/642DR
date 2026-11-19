from thermostat_rl.env import HVACAction
from thermostat_rl.config import ExperimentConfig
from thermostat_rl.state import StateKey
import numpy as np
from thermostat_rl.env import ThermostatEnv
from thermostat_rl.state import discretize_observation

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

    def update_qValue(self, state:StateKey, action: HVACAction, reward: float, next_state: StateKey, terminal_state: bool = False) -> None:
        """Update Q-value using the Q-learning equation"""
        # get Q-values for current & next state
        curr_q_values = self.state_qValues(state)
        next_q_values = self.state_qValues(next_state)

        # use given action to select Q-value, and best Q-value for next state
        current_qValue = curr_q_values[action.value]
        if terminal_state == True:
            best_next_qValue = 0.0
        else:
            best_next_qValue = np.max(next_q_values)

        # calc updated Q-value
        # Q(s, a) ← Q(s, a) + η [r + γ*max_a′ Q(s′, a′) − Q(s, a)]
        update_Q = current_qValue + self.config.learning_rate * (reward + self.config.discount_factor * best_next_qValue - current_qValue)
        curr_q_values[action.value] = update_Q

    def update_epsilon_decay(self) -> None:
        """When finding optimal action, epsilon decays after each episode"""
        # just need to update the epsilon value
        # make it slightly smaller
        update_epsilon = self.epsilon * self.config.epsilon_decay

        # need to make sure it's not smaller than epsilon_min
        self.epsilon = max(self.config.epsilon_min, update_epsilon)
    
    def train_QLearning_episode(self, environment: ThermostatEnv, seed = None) -> float:
        """Train an episode of Tabular Q-Learning"""
        # reset environment, to make sure testing is accurate
        observation, _ = environment.reset(seed = seed)
        state = discretize_observation(observation, self.config)
        final_reward = 0.0
            
        # episode will continue until the environment is either terminated or truncated
        bool_terminated = False
        bool_truncated = False
        while (bool_terminated == False and bool_truncated == False):
            # select an action
            action_selected = self.select_action(state)
    
            # use the selected action
            observation, reward, bool_terminated, bool_truncated, info = environment.step(action_selected)
            
            # discretize observation
            next_state = discretize_observation(observation, self.config)

            # update Q-table value
            if bool_terminated == True or bool_truncated == True:
                self.update_qValue(state, action_selected, reward, next_state, terminal_state=True)
            else: 
                self.update_qValue(state, action_selected, reward, next_state)
                
            # update reward
            final_reward += reward

            # update to next state
            state = next_state

        # update epsilon decay after each episode
        self.update_epsilon_decay()

        return final_reward

