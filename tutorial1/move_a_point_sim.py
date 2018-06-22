"""
A simple simulator for learning to move a 2D point to a target location.
"""
from __future__ import print_function
import math
import sys
from random import random

import bonsai_ai

def debug(*args):
    # To debug, change this to True
    if False:
        print(*args, file=sys.stderr)

def distance(p1, p2):
    """Return the euclidean distance between (x1,y1) and (x2, y2)"""
    x1,y1 = p1
    x2,y2 = p2
    return math.hypot(x1-x2, y1-y2)

class PointSimulation:
    """
    Simulate an agent moving in the plane that contains a target point. 
    The simulation takes moves and computes the states that result from those moves. 
    """

    STEP_SIZE = 0.1  # how far to step each turn
    PRECISION = 0.15 # For the simple task, let's just get to the right
                      # area -- no need to make the AI learn to work
                      # around the fixed step size by detouring.

    def __init__(self, *args):
        super().__init__(*args)

   
    def reset(self):
        """
        Reset simulation state -- call before each episode.
        """
        debug('reset')
        def choose_points():
            # Target is a random point in [0,1]**2
            self.target = (random(), random())
            # same for starting point
            self.current = (random(), random())

        choose_points()
        # re-choose till we start far enough away
        while self.game_over():
            choose_points()

        self.steps = 0
        self.initial_distance = distance(self.current, self.target)
    
    def game_over(self):
        """
        Simulation ends if the agent gets close enough to the target
        position.
        """
        return distance(self.current, self.target) < self.PRECISION

    
    def _move_current(self, direction_radians):
        """
        Move the current point STEP_SIZE in the given direction.
        """
        new_x = self.current[0] + self.STEP_SIZE * math.cos(direction_radians)
        new_y = self.current[1] + self.STEP_SIZE * math.sin(direction_radians)
        self.current = (new_x, new_y)

        
    def step(self, direction_radians):
        """
        Take a step in the specified direction. 
        (Note: function name could be read as bo†h "step the simulation time one tick"
         and "take a step in a direction")

        Args:
            direction_radians: where to go
        """        
        debug("step", direction_radians)
        self._move_current(direction_radians)
        self.steps += 1

        if self.game_over():
            print("Initial distance: {:.3f}. Took {} steps.".format(self.initial_distance, self.steps))
    

class PointBonsaiBridge(bonsai_ai.Simulator):
    """
    This class connects the Bonsai brain to a PointSimulation.
    It also defines our reward function: getting closer to the target state.
    """

    # The simulation doesn't have a timeout, but we want to add one, to prevent
    # the BRAIN from wandering too far in the wrong direction.
    MAX_STEPS = 20    #  Need to be able to go a max distance of roughly
                      # 1.4, so this should be enough if the policy
                      # is good.

    def __init__(self, *args):
        self.num_episodes = 0
        self.simulation = PointSimulation()
        super().__init__(*args)

    def _is_terminal(self):
        """
        We're done either if the AI gets close enough to the target
        state, or if too many steps have passed.
        """
        return (self.simulation.game_over() or
                self.simulation.steps >= self.MAX_STEPS)


    def _reset_sim(self):
        """
        Reset simulator state before the next training episode.
        """
        self.simulation.reset()

        # to compute our reward, we need to track the previous point
        # as well as the current.
        self.previous = self.simulation.current
        self.num_episodes += 1
        if self.num_episodes % 100 == 0:
            print(".", file=sys.stderr)


    def episode_start(self, parameters=None):
        """ called at the start of every episode. should
        reset the simulation and return the initial state
        """
        self._reset_sim()
        return self._get_state()


    def simulate(self, action):
        """
        Simulate one step. Takes the action from the BRAIN as defined
        in the Inkling file.

        Args:
            action: a dictionary with one key: 'direction_radians'
        Returns:
            (state, reward, terminal) tuple, where state is a dict with keys 
                defined in the Inkling schema
        """
        direction = action["direction_radians"]
        
        # save where we are
        self.previous = self.simulation.current

        # take a step
        self.simulation.step(direction)

        # pull the current and target points from the simulation. 
        # We're tracking self.previous directly...
        current = self.simulation.current
        target = self.simulation.target

        # NOTE: self.objective_name, with value "reward_shaped" in this case, is available if
        # you have multiple objectives in your Inkling 
        reward = self.reward_shaped(current, self.previous, target)

        state = self._get_state()
        terminal = self._is_terminal()

        return (state, reward, terminal)


    def _get_state(self):
        """ Gets the state of the simulation, converting it to the form specified in Inkling.
        """
        current = self.simulation.current
        target = self.simulation.target

        state = {"dx": target[0] - current[0],
                 "dy": target[1] - current[1],}

        return state

    def _shape_reward(self, current, previous, target):
        """
        Return a reward for approaching the target. Max 1, min -2.
        """
        progress = distance(previous, target) - distance(current, target)

        # normalize by step size, so now in [-1,1]
        progress /= self.simulation.STEP_SIZE

        # if positive, square to encourage moving toward the target more directly
        if progress > 0:
            progress **= 2
        # if moving away, penalize by extra factor of 2 to discourage wandering toward 
        # and then away.
        else:
            progress *= 2
            # and subtract an extra 1.
            progress -= 1

        return progress

    def reward_shaped(self, current, previous, target):
        """Reward for approaching target"""
        if self._is_terminal():
            return self.MAX_STEPS - self.simulation.steps
        return self._shape_reward(current, previous, target)


if __name__ == "__main__":
    config = bonsai_ai.Config(sys.argv)
    brain = bonsai_ai.Brain(config)
    sim = PointSimulator(brain, "move_a_point_sim")
    print('starting...')
    while sim.run():
        continue
    
