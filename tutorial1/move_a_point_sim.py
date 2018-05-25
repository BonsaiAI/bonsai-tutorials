"""
A simple simulator for learning to move a 2D point to a target location.
"""
from __future__ import print_function
import math
import sys
from random import random

import bonsai_ai

def distance(p1, p2):
    """Return the euclidean distanceb between (x1,y1) and (x2, y2)"""
    x1,y1 = p1
    x2,y2 = p2
    return math.hypot(x1-x2, y1-y2)


class PointSimulator(bonsai_ai.Simulator):
    """
    Our simulator will take moves (eventually from our Inkling file),
    and return the states that result from those moves. It also defines our reward function:
    being close to the target state.
    """

    STEP_SIZE = 0.1  # how far to step each turn
    MAX_STEPS = 20   # Need to be able to go a max distance of roughly
                      # 1.4, so this should be enough if the policy
                      # is good.
    PRECISION = 0.15 # For the simple task, let's just get to the right
                      # area -- no need to make the AI learn to work
                      # around the fixed step size by detouring.

    def __init__(self, *args):
        self.num_episodes = 0
        super().__init__(*args)


    def _move_current(self, direction_radians):
        """
        Move the current point STEP_SIZE in the given direction.
        """
        self.previous = self.current
        new_x = self.current[0] + self.STEP_SIZE * math.cos(direction_radians)
        new_y = self.current[1] + self.STEP_SIZE * math.sin(direction_radians)
        self.current = (new_x, new_y)


    def episode_start(self, parameters=None):
        """ called at the start of every episode. should
        reset the simulation and return the initial state
        """
        # Ensure the points are further than PRECISION apart
        def pick_new_points():
            # Target is a random point in [0,1]**2
            self.target = (random(), random())
            # same for starting point
            self.current = (random(), random())

        pick_new_points()
        while distance(self.current, self.target) <= self.PRECISION:
            pick_new_points()

        self.previous = self.current
        self.steps = 0
        self.num_episodes += 1
        # Periodically print a dot so we know it's still going...
        if self.num_episodes % 100 == 0:
            print(".", file=sys.stderr)
        return self._get_state()

        
    def _is_terminal(self):
        """
        We're done either if the AI gets close enough to the target
        state, or if too much time passes.
        """
        return ((distance(self.current, self.target) < self.PRECISION) or
                self.steps > self.MAX_STEPS)


    def simulate(self, action):
        """ Function to make a move based on output from the BRAIN as defined
        in the Inkling file.
        Args:
            action: a dictionary with one key: 'direction_radians'
        """
        direction = action["direction_radians"]
        self._move_current(direction)
        self.steps += 1

        state = self._get_state()
        # Reward for getting closer, with a penalty of 1 for each step.
        # The penalty encourages the brain to finish faster
        if self.objective_name == "reward_closeness":
            reward = -distance(self.current, self.target) - 1
        elif self.predict:
            # no reward in prediction mode
            reward = 0
            print(self.steps)
        else:
            # not in prediction mode, and unexpected objective name
            raise ValueError("Unknown objective: {}".format(self.objective_name))
        return (state, reward, self._is_terminal())


    def _get_state(self):
        """ Gets the state of the simulator, whether it be a valid value or
        terminal ("game over") state.
        """
        # Save the state--we may reset the simulation and blow
        # everything away below.
        state = {"current_x": self.current[0],
                 "current_y": self.current[1],
                 "target_x": self.target[0],
                 "target_y": self.target[1]}

        return state


if __name__ == "__main__":
    config = bonsai_ai.Config(sys.argv)
    brain = bonsai_ai.Brain(config)
    sim = PointSimulator(brain, "move_a_point_sim")
    print('starting...')
    while sim.run():
        continue
