from thermostat_rl.env import HVACAction
from thermostat_rl.config import ExperimentConfig
from thermostat_rl.state import StateKey
from thermostat_rl.q_learning import TabularQLearning
import numpy as np

def test_state_qValues_newState():
    """Testing state_qValues() when state is not in the Q table yet"""
    # initialize values
    config = ExperimentConfig()
    state = (0, 0, 0, 0, 0, 0, 0)
    q_learning = TabularQLearning(config)
    q_values = q_learning.state_qValues(state)

    # if state not in Q table yet, all actions will be zero
    assert state in q_learning.q_table
    assert np.all(q_values == 0)

def test_state_qValues_existingState():
    """Testing state_qValues() when state is already in the Q table"""
    # initialize values
    config = ExperimentConfig()
    state = (0, 0, 0, 0, 0, 0, 0)
    q_learning = TabularQLearning(config)

    # add state to Q table before calling state_qValues()
    q_learning.q_table[state] = np.array([6.0, 4.0, 2.0, 642.0, 64.2])
    assert state in q_learning.q_table

    q_values = q_learning.state_qValues(state)
    assert np.array_equal(q_values, np.array([6.0, 4.0, 2.0, 642.0, 64.2]))