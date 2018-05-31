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

    def _is_terminal(self):
        """
        We're done either if the AI gets close enough to the target
        state, or if too much time passes.
        """
        return ((distance(self.current, self.target) < self.PRECISION) or
                self.steps >= self.MAX_STEPS)


    def _reset(self):
        """ Reset is called to reset simulator state before the next training session.
        """
        debug('_reset')
        def choose_points():
            # Target is a random point in [0,1]**2
            self.target = (random(), random())
            # same for starting point
            self.current = (random(), random())

        choose_points()
        # re-choose till we start far enough away
        while distance(self.current, self.target) <= self.PRECISION:
            choose_points()

        self.previous = self.current
        self.steps = 0
        self.num_episodes += 1
        self.initial_distance = distance(self.current, self.target)
        if self.num_episodes % 100 == 0:
            print(".", file=sys.stderr)


    def episode_start(self, parameters=None):
        """ called at the start of every episode. should
        reset the simulation and return the initial state
        """
        self._reset()
        return self._get_state()
        
    def simulate(self, action):
        """ Function to make a move based on output from the BRAIN as defined
        in the Inkling file.

        Args:
            action: a dictionary with one key: 'direction_radians'
        """
        direction = action["direction_radians"]
        
        debug("advance", direction)
        self._move_current(direction)
        self.steps += 1

        state = self._get_state()
        terminal = self._is_terminal()

        if self.objective_name == "reward_shaped":
            reward = self.reward_shaped()
        elif self.predict:
            # no reward in prediction mode
            reward = 0
            if terminal:
                print("Initial distance: {:.3f}. Took {} steps.".format(self.initial_distance, self.steps))
        else:
            # not in prediction mode, and unexpected objective name
            raise ValueError("Unknown objective: {}".format(self.objective_name))
        
        return (state, reward, terminal)


    def _get_state(self):
        """ Gets the state of the simulator, whether it be a valid value or
        terminal ("game over") state.
        """
        debug("get_state")
        debug("state", self.current, self.previous, self.target)

        state = {"dx": self.target[0] - self.current[0],
                 "dy": self.target[1] - self.current[1],
                 }

        return state

    def _shape_reward(self, current, previous, target):
        """
        Return a reward for approaching the target. Max 1, min -2.
        """
        progress = distance(previous, target) - distance(current, target)

        # normalize by step size, so now in [-1,1]
        progress /= self.STEP_SIZE

        # if positive, square to encourage moving toward the target more directly
        if progress > 0:
            progress **= 2
        # if moving away, penalize by extra factor of 2 to discourage wandering toward and away.
        else:
            progress *= 2
            # and subtract an extra 1.
            progress -= 1

        return progress

    def reward_shaped(self):
        """Reward for approaching target"""
        if self._is_terminal():
            return self.MAX_STEPS - self.steps
        return self._shape_reward(self.current, self.previous, self.target)


if __name__ == "__main__":
    config = bonsai_ai.Config(sys.argv)
    brain = bonsai_ai.Brain(config)
    sim = PointSimulator(brain, "move_a_point_sim")
    print('starting...')
    while sim.run():
        continue
    
