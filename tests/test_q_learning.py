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

def test_TabularQLearning_greedyEpsilon_zero():
    """Testing epsilon-greedy selection when epsilon is zero"""
    # initialize values
    config = ExperimentConfig()
    state = (0, 0, 0, 0, 0, 0, 0)
    q_learning = TabularQLearning(config)

    # using epsilon = 0, will make selection based on best Q-value
    q_learning.epsilon = 0.0

    # add state to Q table before calling state_qValues()
    # since fourth HVACAction has highest value, HVAC should select COOL_LOW
    q_learning.q_table[state] = np.array([6.0, 4.0, 2.0, 642.0, 64.2])
    assert state in q_learning.q_table
    best_greedy_action = q_learning.select_action(state)
    assert best_greedy_action == HVACAction.COOL_LOW

def test_TabularQLearning_greedyEpsilon_one():
    """Testing epsilon-greedy selection when epsilon is one"""
    # initialize values
    config = ExperimentConfig()
    state = (0, 0, 0, 0, 0, 0, 0)
    q_learning = TabularQLearning(config)

    # using epsilon = 1, will make selection based on random action
    q_learning.epsilon = 1.0

    # add state to Q table before calling state_qValues()
    q_learning.q_table[state] = np.array([6.0, 4.0, 2.0, 642.0, 64.2])
    assert state in q_learning.q_table

    # check potential random actions are valid
    random_actions = []
    for i in range(20):
        random_action_selected = q_learning.select_action(state)
        random_actions.append(random_action_selected)
        # make sure the action is valid
        assert isinstance(random_action_selected, HVACAction)
    
    # since we were selecting random actions, there should be more than one distinct HVACAction in the list
    # make sure it's not always just selecting the same action
    assert len(set(random_actions)) > 1