import math
import random

import numpy as np

from bonsai_ai import Simulator, Config, Brain

import move_a_point_sim as sim

def run_sim_episode(sim, policy):
    """
    Given a sim and a policy, step through some iterations 
    of the simulator, returning the history of states.
    
    Args:
        sim: a PointSimulator
        policy: a function (SimState -> action dictionary)
    """
    k = 0 # Count steps, break out of infinite loops
    state_history = []
    reward_history = []
    state = sim.episode_start()
    state_history.append(state)

    is_terminal = False
    
    while not is_terminal:
        action = policy(state)
        # convert to a [-1,1] action
        #action['direction_radians'] = action['direction_radians'] / math.pi - 1.0
        (state, reward, is_terminal) = sim.simulate(action)
        print(state, reward, is_terminal)
        state_history.append(state)
        reward_history.append(reward)
        k += 1
        if k > 1000:
            raise Exception("Simulation ran longer than 1000 steps. Stopping.")

    return state_history, reward_history

# Some silly policies
def random_policy(state):
    """
    Ignore the state, move randomly.
    """
    return {'direction_radians': random.random() * 2 * math.pi}

def go_up_policy(state):
    return {'direction_radians': math.pi / 2.0}

def run():
    config = Config()
    brain = Brain(config, "move-a-point")
    point_sim = sim.PointBonsaiBridge(brain, "move_a_point_sim")

    states, rewards = run_sim_episode(point_sim, random_policy)
#    print("states: ", states)
#    print("rewards: ", rewards)


if __name__ == "__main__":
    run()
